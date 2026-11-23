import bz2
import csv
import gzip
import logging
from io import BufferedWriter, BytesIO
from itertools import zip_longest
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple, TypedDict, Union

import duckdb
from duckdb import DuckDBPyConnection, DuckDBPyRelation

from countess import VERSION
from countess.core.parameters import (
    ArrayParam,
    BaseParam,
    BooleanParam,
    ChoiceParam,
    ColumnChoiceParam,
    DataTypeOrNoneChoiceParam,
    FileSaveParam,
    MultiParam,
    StringParam,
    TabularMultiParam,
)
from countess.core.plugins import DuckdbLoadFileWithTheLotPlugin, DuckdbSaveFilePlugin
from countess.utils.compress import compressed_output, open_for_reading
from countess.utils.duckdb import (
    duckdb_dtype_is_boolean,
    duckdb_dtype_is_numeric,
    duckdb_dtype_to_datatype_choice,
    duckdb_source_to_table,
)
from countess.utils.sql import Id, Lit, Raw, Sql, Type, join

CSV_FILE_TYPES: Sequence[Tuple[str, Union[str, List[str]]]] = [
    ("CSV", [".csv", ".csv.gz", ".csv.bz2"]),
    ("TSV", [".tsv", ".tsv.gz", ".tsv.bz2"]),
    ("TXT", [".txt", ".text", ".txt.gz", ".text.gz", ".txt.bz2", ".text.bz2"]),
]

logger = logging.getLogger(__name__)


class ColumnsMultiParam(MultiParam):
    name = StringParam("Column Name", "")
    type = DataTypeOrNoneChoiceParam("Column Type")


CSV_DELIMITER_CHOICES: dict[str, str] = {",": ",", ";": ";", "|": "|", "TAB": "\t", "SPACE": " "}


def csv_sql_type(choice: str) -> str:
    """The DuckDB type which a column is loaded as, for the type it's been given.  They're the same
    but for INTEGER: that's a 64 bit integer, as a Python int effectively is, since DuckDB's own INTEGER
    is only 32 bit and a number outside of that would quietly become a NULL (3000000000 does)."""
    return "BIGINT" if choice == "INTEGER" else choice


class ReadCsvOptions(TypedDict):
    sep: str
    null_padding: bool
    strict_mode: bool
    all_varchar: bool
    skiprows: int
    header: bool


class LoadCsvPlugin(DuckdbLoadFileWithTheLotPlugin):
    """Load CSV files"""

    name = "CSV Load"
    description = "Loads data from CSV or similar delimited text files and assigns types to columns"
    link = "https://countess-project.github.io/CountESS/included-plugins/#csv-reader"
    version = VERSION
    file_types = CSV_FILE_TYPES

    delimiter = ChoiceParam("Delimiter", ",", choices=CSV_DELIMITER_CHOICES.keys())
    header = BooleanParam("CSV file has header row?", True)
    columns = ArrayParam("Columns", ColumnsMultiParam("Column"))

    def load_file(
        self, cursor: duckdb.DuckDBPyConnection, filename: str, file_param: BaseParam, row_limit: Optional[int] = None
    ) -> duckdb.DuckDBPyRelation:
        options: ReadCsvOptions = {
            "sep": CSV_DELIMITER_CHOICES[self.delimiter.value],
            "null_padding": True,
            "strict_mode": False,
            "all_varchar": len(self.columns) > 0,
            "skiprows": 1 if self.header and len(self.columns) > 0 else 0,
            "header": self.header.value and len(self.columns) == 0,
        }
        if filename.lower().endswith((".bz2", ".xz")):
            with open_for_reading(filename) as fh:
                rel = cursor.read_csv(fh, **options)  # typing: disable=arg-type
        else:
            rel = cursor.read_csv(filename, **options)  # typing: disable=arg-type

        if row_limit is not None:
            rel = rel.limit(row_limit)

        if len(self.columns):
            # If there's a bonus filename column ignore it for now
            rel_columns = [c for c in rel.columns if not (self.filename_column and c == "filename")]
            # there's three cases here, either we've got both a column
            # name or a column parameter or both, depending on the relative
            # lengths of the column lists.
            proj = join(
                (
                    Id(cn)
                    if cp is None
                    else Sql.template(
                        "TRY_CAST({value} AS {type}) AS {name}",
                        value=Raw("NULL") if cn is None else Id(cn),
                        type=Type(csv_sql_type(cp.type.value)),
                        name=Id(cp.name.value or f"column{num}"),
                    )
                )
                for num, (cn, cp) in enumerate(zip_longest(rel_columns, self.columns))
                if cp is None or cp.type.is_not_none()
            )

            logger.debug("LoadCsvPlugin.load_file proj %s", proj)
            rel = rel.project(str(proj))

        # `rel = cursor.read_csv(fh)` returns a `DuckDBPyRelation`
        # with `rel.alias` starting `DUCKDB_INTERNAL_OBJECTSTORE:`
        # but limits / projections / views of that relation stop
        # working once `rel` falls out of scope.
        #
        # `rel = cursor.read_csv(filename)` returns a `DuckDBPyRelation`
        # with `rel.alias` starting `unnamed_relation_` and those
        # seem to work fine to limit / project / view.

        if rel.alias.startswith("DUCKDB_INTERNAL_OBJECTSTORE:"):
            rel = duckdb_source_to_table(cursor, rel)

        return rel

    def combine(
        self, ddbc: duckdb.DuckDBPyConnection, tables: Iterable[duckdb.DuckDBPyRelation]
    ) -> Optional[duckdb.DuckDBPyRelation]:
        combined = super().combine(ddbc, tables)
        if combined is None:
            return None

        # filename is always last column if it exists.
        combined_columns_and_dtypes = list(zip(combined.columns, combined.dtypes))
        if self.filename_column:
            combined_columns_and_dtypes.pop()

        for num, (column, dtype) in enumerate(combined_columns_and_dtypes):
            if num >= len(self.columns) and not (self.filename_column and column == "filename"):
                new_param = self.columns.add_row()
                new_param.name.value = column
                new_param.type.value = duckdb_dtype_to_datatype_choice(dtype)
        return combined


class SaveCsvOrderParameter(TabularMultiParam):
    order_by = ColumnChoiceParam("Order By")
    descending = BooleanParam("Descending")


def write_to_csv(
    ddbc: DuckDBPyConnection, source: DuckDBPyRelation, filename: str, sep: str, quoting: int, header: bool = True
) -> None:
    """Write `source` to `filename` with DuckDB, compressed (as .gz, .bz2 or .xz) if that's what the name says.

    `quoting` is csv.QUOTE_MINIMAL, which is what DuckDB does, or csv.QUOTE_NONNUMERIC.  DuckDB has no mode for
    that, but it can be told which columns to always quote, so that quotes every column which isn't a number
    (or a boolean).

    The compression is done as the file is written (see `countess.utils.compress`) rather than by DuckDB, which
    only does gzip and does it in one thread.  Whichever it's compressed with the uncompressed text is exactly the
    same, and the file isn't there until it's all been written.

    Some things about what's written:
    * with QUOTE_NONNUMERIC the names in the header row are quoted for the same columns as the values are.
    * a missing value is empty, whatever the column, and an empty string is `""`.
    * a FLOAT (32 bit) is written as the shortest text which reads back as the same number.
    * NaN and infinity are `nan` and `inf`, and booleans are `true` and `false`."""

    if quoting not in (csv.QUOTE_MINIMAL, csv.QUOTE_NONNUMERIC):
        raise ValueError(f"quoting must be csv.QUOTE_MINIMAL or csv.QUOTE_NONNUMERIC, not {quoting!r}")

    logger.debug("writing file %s", filename)

    options = [
        Raw("FORMAT CSV"),
        Raw("HEADER true" if header else "HEADER false"),
        Sql.template("DELIMITER {sep}", sep=Lit(sep)),
        Raw("COMPRESSION none"),  # (otherwise it would compress it as well, going by the name)
        Raw("USE_TMP_FILE false"),  # (it's written to a temporary file already, so it is all there or isn't)
    ]
    if quoting == csv.QUOTE_NONNUMERIC:
        quoted = [
            Id(name)
            for name, dtype in zip(source.columns, source.dtypes)
            if not (duckdb_dtype_is_numeric(dtype) or duckdb_dtype_is_boolean(dtype))
        ]
        if quoted:
            options.append(Sql.template("FORCE_QUOTE ({columns})", columns=join(quoted)))

    with compressed_output(ddbc, filename) as target:
        query = Sql.template(
            "COPY ({source}) TO {target} ({options})",
            source=Raw(source.sql_query()),  # (the SQL of the relation, which is DuckDB's own)
            target=Lit(target),
            options=join(options),
        )
        logger.debug("write_to_csv %s", query)
        ddbc.execute(str(query))


class SaveCsvPlugin(DuckdbSaveFilePlugin):
    name = "CSV Save"
    description = "Save data as CSV or similar delimited text files"
    link = "https://countess-project.github.io/CountESS/included-plugins/#csv-writer"
    version = VERSION
    file_types = CSV_FILE_TYPES

    header = BooleanParam("CSV header row?", True)
    filename = FileSaveParam("Filename", file_types=file_types)
    delimiter = ChoiceParam("Delimiter", ",", choices=[",", ";", "TAB", "|", "SPACE"])
    quoting = BooleanParam("Quote Non-numerics?", True)
    sorting = ArrayParam("Sorting", SaveCsvOrderParameter("sort"))

    filehandle: Optional[Union[BufferedWriter, BytesIO, gzip.GzipFile, bz2.BZ2File]] = None
    csv_columns = None

    SEPARATORS = {",": ",", ";": ";", "|": "|", "SPACE": " ", "TAB": "\t"}
    QUOTING = {False: csv.QUOTE_MINIMAL, True: csv.QUOTE_NONNUMERIC}

    show_preview = False

    def execute(
        self, ddbc: DuckDBPyConnection, source: Optional[DuckDBPyRelation], row_limit: Optional[int] = None
    ) -> None:
        if source is None or row_limit is not None or not self.filename:
            return

        if len(self.sorting):
            order_by = join(
                Sql.template("{column} DESC", column=Id(sp.order_by.value)) if sp.descending else Id(sp.order_by.value)
                for sp in self.sorting
            )
            logger.debug("SaveCsvPlugin.execute order_by %s", order_by)
            source = source.order(str(order_by))

        filename = self.filename.get_file_path()
        Path(filename).parent.mkdir(parents=True, exist_ok=True)
        write_to_csv(
            ddbc,
            source,
            filename,
            sep=self.SEPARATORS[self.delimiter.value],
            quoting=self.QUOTING[self.quoting.value],
            header=bool(self.header),
        )
