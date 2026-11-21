import bz2
import csv
import gzip
import logging
import lzma
from io import BufferedWriter, BytesIO, TextIOBase
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
from countess.utils.duckdb import duckdb_dtype_to_datatype_choice, duckdb_escape_identifier, duckdb_source_to_table

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


# How the types a column of a CSV file can be given are loaded:
#
# * INTEGER is a 64 bit integer.  (DuckDB's INTEGER is only 32 bit, which would silently turn anything larger
#   than 2,147,483,647 into a NULL.)
#
# * FLOAT is a DECIMAL, so that the numbers are exactly what was in the file: DuckDB's FLOAT is only single
#   precision, and a DOUBLE rounds, say, 0.1 to the nearest number a binary fraction can make.  A DECIMAL has to
#   have a fixed number of digits though, and one with a lot of them can't be multiplied (DECIMAL(38,18) times
#   itself is an overflow error, whatever's in it), so what's used is the smallest one which holds every number
#   which is in the column: see csv_decimal_sizes.

CSV_INTEGER_TYPE = "BIGINT"

# The most digits a DuckDB DECIMAL can have
DECIMAL_MAX_DIGITS = 38


def _decimal_digits_sql(identifier: str) -> Tuple[str, str]:
    """Returns SQL for how many digits before the decimal point and how many after it are needed to hold a
    value of the text column `identifier` exactly: NULL if it isn't a finite number.  It's allowed to be a bit
    too generous (it counts a sign, and zeros at the ends) but never too small.

    Most numbers are just digits and a point, which is looked at with plain string functions, much quicker
    than the regular expressions which are needed to follow an exponent, as in 1.5e-3, moving the point."""
    has_point = f"instr({identifier}, '.') > 0"
    has_exponent = f"(contains({identifier}, 'e') OR contains({identifier}, 'E'))"
    sign = f"(CASE WHEN {identifier}[1] IN ('-', '+') THEN 1 ELSE 0 END)"
    plain_before = f"(CASE WHEN {has_point} THEN instr({identifier}, '.') - 1 ELSE length({identifier}) END) - {sign}"
    plain_after = f"(CASE WHEN {has_point} THEN length({identifier}) - instr({identifier}, '.') ELSE 0 END)"

    text = f"trim({identifier})"
    before = f"length(ltrim(regexp_extract({text}, '^[+-]?(\\d*)', 1), '0'))"
    after = f"length(rtrim(regexp_extract({text}, '^[+-]?\\d*\\.?(\\d*)', 1), '0'))"
    exponent = f"coalesce(try_cast(regexp_extract({text}, '[eE]([+-]?\\d+)$', 1) as integer), 0)"

    is_finite = f"try_cast({identifier} as double) is not null and isfinite(try_cast({identifier} as double))"
    return (
        f"CASE WHEN {is_finite} THEN greatest(0, CASE WHEN {has_exponent} THEN {before} + {exponent} "
        f"ELSE {plain_before} END) END",
        f"CASE WHEN {is_finite} THEN greatest(0, CASE WHEN {has_exponent} THEN {after} - {exponent} "
        f"ELSE {plain_after} END) END",
    )


def csv_decimal_sizes(rel: DuckDBPyRelation, columns: Sequence[str]) -> dict[str, Optional[Tuple[int, int]]]:
    """For each of the named text columns of `rel`, the `(width, scale)` of the smallest DECIMAL which holds
    every number in it exactly, or None if there's one which is too big to be a DECIMAL at all (in which case
    it has to be a DOUBLE).  This reads all of `rel` once.  If there's nothing in a column which is a number
    its size doesn't matter, so it's the smallest there is."""
    if not columns:
        return {}
    aggregates = []
    for number, column in enumerate(columns):
        before, after = _decimal_digits_sql(duckdb_escape_identifier(column))
        aggregates += [f"max({before}) AS before_{number}", f"max({after}) AS after_{number}"]
    row = rel.aggregate(", ".join(aggregates)).fetchone()
    assert row is not None

    sizes: dict[str, Optional[Tuple[int, int]]] = {}
    for number, column in enumerate(columns):
        before, after = row[2 * number], row[2 * number + 1]
        if before is None or after is None:
            sizes[column] = (1, 0)
            continue
        before = max(before, 1)  # (there's always a digit before the point, even if it's a 0)
        if before > DECIMAL_MAX_DIGITS:
            logger.warning("Column %s has numbers too big to be a DECIMAL, loading it as a DOUBLE", column)
            sizes[column] = None
            continue
        if before + after > DECIMAL_MAX_DIGITS:
            logger.warning(
                "Column %s has numbers with more than %d digits, rounding them to fit", column, DECIMAL_MAX_DIGITS
            )
            after = DECIMAL_MAX_DIGITS - before
        sizes[column] = (before + after, after)
    return sizes


def csv_cast_sql(value_sql: str, choice: str, size: Optional[Tuple[int, int]] = None, decimal: bool = False) -> str:
    """The SQL for turning the text `value_sql` into what `choice` (INTEGER, FLOAT, STRING, BOOLEAN) of the
    types of a column means.  If `decimal`, a FLOAT is a DECIMAL of `size` (width, scale), or a DOUBLE if that's
    None."""
    if choice == "INTEGER":
        return f"TRY_CAST({value_sql} as {CSV_INTEGER_TYPE})"
    if choice == "FLOAT" and decimal:
        if size is None:
            return f"TRY_CAST({value_sql} as DOUBLE)"
        width, scale = size
        direct = f"TRY_CAST({value_sql} as DECIMAL({width},{scale}))"
        # DuckDB has trouble getting a number with an exponent (1.5e-3) into a DECIMAL of just the right size
        # (it needs room for all the digits of the part before the exponent as well, and without it the
        # number is a NULL, which in a test was one in ten of them), so those are put in a wide DECIMAL of the
        # same scale first.  That's much slower, hence only for the ones which have an exponent.
        via_wide = (
            f"TRY_CAST(TRY_CAST({value_sql} as DECIMAL({DECIMAL_MAX_DIGITS},{scale})) as DECIMAL({width},{scale}))"
        )
        if value_sql == "NULL":
            return direct
        return f"CASE WHEN contains({value_sql}, 'e') OR contains({value_sql}, 'E') THEN {via_wide} ELSE {direct} END"
    return f"TRY_CAST({value_sql} as {choice})"


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

    @staticmethod
    def _read_once(
        cursor: duckdb.DuckDBPyConnection, rel: duckdb.DuckDBPyRelation, row_limit: Optional[int]
    ) -> duckdb.DuckDBPyRelation:
        """What's read from a file object (as for the compressed files) can only be read the once, and has to
        be before the file is closed, so it goes straight into a table, which can be looked at as often as needed."""
        if row_limit is not None:
            rel = rel.limit(row_limit)
        return duckdb_source_to_table(cursor, rel)

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
        if filename.endswith(".bz2"):
            with bz2.open(filename) as fh:
                rel = self._read_once(cursor, cursor.read_csv(fh, **options), row_limit)  # typing: disable=arg-type
        elif filename.endswith(".xz"):
            with lzma.open(filename) as fh:
                rel = self._read_once(cursor, cursor.read_csv(fh, **options), row_limit)  # typing: disable=arg-type
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
            pairs = list(zip_longest(rel_columns, self.columns))

            # FLOAT columns are as big a DECIMAL as the numbers in them need, which means looking at them first
            float_columns = [cn for cn, cp in pairs if cn is not None and cp is not None and cp.type.value == "FLOAT"]
            sizes = csv_decimal_sizes(rel, float_columns)

            proj = ",".join(
                (
                    duckdb_escape_identifier(cn)
                    if cp is None
                    else (
                        csv_cast_sql(
                            "NULL" if cn is None else duckdb_escape_identifier(cn),
                            cp.type.value,
                            sizes.get(cn, (1, 0)) if cn is not None else (1, 0),
                            decimal=True,
                        )
                        + " AS "
                        + (duckdb_escape_identifier(cp.name.value) if cp.name.value else "column%d" % num)
                    )
                )
                for num, (cn, cp) in enumerate(pairs)
                if cp is None or cp.type.is_not_none()
            )

            logger.debug("LoadCsvPlugin.load_file proj %s", proj)
            rel = rel.project(proj)

        # (see _read_once for why it's done there now, for the files which need it)
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


def write_to_csv(source: DuckDBPyRelation, filename: str, sep: str, quoting: int, header: bool = True) -> None:
    logger.debug("writing file %s", filename)

    def _write(fh: TextIOBase | gzip.GzipFile | bz2.BZ2File | lzma.LZMAFile, **kwargs):
        # the type check supression is because the first parameter to
        # to_csv is:
        #     path_or_buf: str | PathLike[str] | WriteBuffer[bytes] | WriteBuffer[str]
        # but bz2.BZFile|lzma.LZMAFile don't seem to be compatible.
        chunk = source.fetch_df_chunk()
        chunk.to_csv(fh, header=header, index=False, **kwargs)  # type: ignore
        while len(chunk) > 0:
            chunk = source.fetch_df_chunk()
            chunk.to_csv(fh, header=False, index=False, **kwargs)  # type: ignore

    # If it's one of the formats that duckdb doesn't (yet) support, write via pandas
    # See also https://github.com/nickzoic/duckdb-compression/

    if filename.endswith(".bz2"):
        with bz2.open(filename, "w") as fh_bz2:
            _write(fh_bz2, sep=sep, quoting=quoting)
    elif filename.endswith(".xz"):
        with lzma.open(filename, "w") as fh_xz:
            _write(fh_xz, sep=sep, quoting=quoting)
    elif quoting not in (csv.QUOTE_MINIMAL, csv.QUOTE_ALL):
        # duckdb doesn't support this properly
        # https://github.com/duckdb/duckdb-python/discussions/491
        if filename.endswith(".gz"):
            with gzip.open(filename, "w") as fh_gz:
                _write(fh_gz, sep=sep, quoting=quoting)
        else:
            with open(filename, "w", encoding="utf-8") as fh_open:
                _write(fh_open, sep=sep, quoting=quoting)
    elif quoting == csv.QUOTE_ALL:
        # otherwise write it directly
        source.to_csv(filename, sep=sep, quoting=quoting, header=header)
    else:
        source.to_csv(filename, sep=sep, header=header)


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
            order_by = ",".join(
                duckdb_escape_identifier(sp.order_by.value) + (" desc" if sp.descending else "") for sp in self.sorting
            )
            logger.debug("SaveCsvPlugin.execute order_by %s", order_by)
            source = source.order(order_by)

        filename = self.filename.get_file_path()
        Path(filename).parent.mkdir(parents=True, exist_ok=True)
        write_to_csv(
            source,
            filename,
            sep=self.SEPARATORS[self.delimiter.value],
            quoting=self.QUOTING[self.quoting.value],
            header=bool(self.header),
        )
