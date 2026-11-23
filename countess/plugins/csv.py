import bz2
import csv
import gzip
import logging
import tempfile
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
from countess.utils.sql import Fragment, Id, Lit, Raw, Sql, Type, join

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


# How the types a column of a CSV file can be given are loaded:
#
# * INTEGER is a 64 bit integer (see csv_sql_type).
#
# * FLOAT is a DECIMAL, so that the numbers are exactly what was in the file: DuckDB's FLOAT is only single
#   precision, and a DOUBLE rounds, say, 0.1 to the nearest number a binary fraction can make.  A DECIMAL has to
#   have a fixed number of digits though, and one with a lot of them can't be multiplied (DECIMAL(38,18) times
#   itself is an overflow error, whatever's in it), so what's used is the smallest one which holds every number
#   which is in the column: see csv_decimal_sizes.

# The most digits a DuckDB DECIMAL can have
DECIMAL_MAX_DIGITS = 38


def _decimal_digits_sql(value: Fragment) -> Tuple[Sql, Sql]:
    """Returns SQL for how many digits before the decimal point and how many after it are needed to hold a
    value of the text column `value` exactly: NULL if it isn't a finite number.  It's allowed to be a bit
    too generous (it counts a sign, and zeros at the ends) but never too small.

    Most numbers are just digits and a point, which is looked at with plain string functions, much quicker
    than the regular expressions which are needed to follow an exponent, as in 1.5e-3, moving the point."""

    def sql(template: str, **fragments: Fragment) -> Sql:
        return Sql.template(template, value=value, **fragments)

    has_point = sql("instr({value}, '.') > 0")
    has_exponent = sql("(contains({value}, 'e') OR contains({value}, 'E'))")
    sign = sql("(CASE WHEN {value}[1] IN ('-', '+') THEN 1 ELSE 0 END)")
    plain_before = sql(
        "(CASE WHEN {has_point} THEN instr({value}, '.') - 1 ELSE length({value}) END) - {sign}",
        has_point=has_point,
        sign=sign,
    )
    plain_after = sql(
        "(CASE WHEN {has_point} THEN length({value}) - instr({value}, '.') ELSE 0 END)", has_point=has_point
    )

    text = sql("trim({value})")
    before = Sql.template(r"length(ltrim(regexp_extract({text}, '^[+-]?(\d*)', 1), '0'))", text=text)
    after = Sql.template(r"length(rtrim(regexp_extract({text}, '^[+-]?\d*\.?(\d*)', 1), '0'))", text=text)
    exponent = Sql.template(
        r"coalesce(try_cast(regexp_extract({text}, '[eE]([+-]?\d+)$', 1) as integer), 0)", text=text
    )

    is_finite = sql("try_cast({value} as double) is not null and isfinite(try_cast({value} as double))")

    def digits(with_exponent: Sql, without_exponent: Sql) -> Sql:
        return Sql.template(
            "CASE WHEN {is_finite} THEN greatest(0, CASE WHEN {has_exponent} THEN {with_exponent} "
            "ELSE {without_exponent} END) END",
            is_finite=is_finite,
            has_exponent=has_exponent,
            with_exponent=with_exponent,
            without_exponent=without_exponent,
        )

    return (
        digits(Sql.template("{before} + {exponent}", before=before, exponent=exponent), plain_before),
        digits(Sql.template("{after} - {exponent}", after=after, exponent=exponent), plain_after),
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
        before, after = _decimal_digits_sql(Id(column))
        aggregates += [
            Sql.template("max({digits}) AS {name}", digits=before, name=Id(f"before_{number}")),
            Sql.template("max({digits}) AS {name}", digits=after, name=Id(f"after_{number}")),
        ]
    row = rel.aggregate(str(join(aggregates))).fetchone()
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


def csv_cast_sql(value: Fragment, choice: str, size: Optional[Tuple[int, int]] = None, decimal: bool = False) -> Sql:
    """The SQL for turning the text `value` into what `choice` (INTEGER, FLOAT, STRING, BOOLEAN) of the
    types of a column means.  If `decimal`, a FLOAT is a DECIMAL of `size` (width, scale), or a DOUBLE if that's
    None."""
    if choice == "FLOAT" and decimal:
        if size is None:
            return Sql.template("TRY_CAST({value} AS DOUBLE)", value=value)
        width, scale = size
        decimal_type = Type(f"DECIMAL({width},{scale})")
        direct = Sql.template("TRY_CAST({value} AS {type})", value=value, type=decimal_type)
        # DuckDB has trouble getting a number with an exponent (1.5e-3) into a DECIMAL of just the right size
        # (it needs room for all the digits of the part before the exponent as well, and without it the
        # number is a NULL, which in a test was one in ten of them), so those are put in a wide DECIMAL of the
        # same scale first.  That's much slower, hence only for the ones which have an exponent.
        #
        # XXX This is a work-around for a bug in DuckDB, so CHECK WHETHER IT'S BEEN FIXED YET:
        #     https://github.com/duckdb/duckdb/issues/29189
        # and if it has, take out `via_wide` and the CASE (and keep the tests, they'll show it's still right).
        # That issue is about a width which is the same as the scale, where it's one digit short, but what was
        # found is that the cast needs room for (the digits before the point) + (the scale), as the number was
        # written, so 7642656515e-5 is a NULL as a DECIMAL(10,5) though it fits, and is five digits short, and
        # the work-around has a limit too: it fails when those add up to more than 38.  So when it's fixed,
        # check that those are, as well as 1e-5 as a DECIMAL(5,5), before taking this out.
        via_wide = Sql.template(
            "TRY_CAST(TRY_CAST({value} AS {wide}) AS {type})",
            value=value,
            wide=Type(f"DECIMAL({DECIMAL_MAX_DIGITS},{scale})"),
            type=decimal_type,
        )
        if value == Raw("NULL"):
            return direct
        return Sql.template(
            "CASE WHEN contains({value}, 'e') OR contains({value}, 'E') THEN {via_wide} ELSE {direct} END",
            value=value,
            via_wide=via_wide,
            direct=direct,
        )
    return Sql.template("TRY_CAST({value} AS {type})", value=value, type=Type(csv_sql_type(choice)))


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
        if filename.lower().endswith((".bz2", ".xz")):
            with open_for_reading(filename) as fh:
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

            proj = join(
                (
                    Id(cn)
                    if cp is None
                    else Sql.template(
                        "{cast} AS {name}",
                        cast=csv_cast_sql(
                            Raw("NULL") if cn is None else Id(cn),
                            cp.type.value,
                            sizes.get(cn, (1, 0)) if cn is not None else (1, 0),
                            decimal=True,
                        ),
                        name=Id(cp.name.value or f"column{num}"),
                    )
                )
                for num, (cn, cp) in enumerate(pairs)
                if cp is None or cp.type.is_not_none()
            )

            logger.debug("LoadCsvPlugin.load_file proj %s", proj)
            rel = rel.project(str(proj))

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


# how many rows of the data are shown in the preview of CSV Save, which is of what the file will look like
CSV_PREVIEW_ROWS = 100


def csv_text(ddbc: DuckDBPyConnection, source: DuckDBPyRelation, sep: str, quoting: int, header: bool = True) -> str:
    """The text which `write_to_csv` writes for `source`, which is the same whatever a file is compressed with if it
    is.  It has `write_to_csv` write a file and reads it back, so it's what the file is and not what some other way
    of writing it would make (how booleans, numbers, nulls and quotes come out are all up to the writer)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "preview.csv"
        write_to_csv(ddbc, source, str(path), sep=sep, quoting=quoting, header=header)
        return path.read_text(encoding="utf-8")


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

    def sorted(self, source: DuckDBPyRelation) -> DuckDBPyRelation:
        """The source in the order that the file is to have it in"""
        if len(self.sorting) == 0:
            return source
        order_by = join(
            Sql.template("{column} DESC", column=Id(sp.order_by.value)) if sp.descending else Id(sp.order_by.value)
            for sp in self.sorting
        )
        logger.debug("SaveCsvPlugin.sorted order_by %s", order_by)
        return source.order(str(order_by))

    def preview_text(self, ddbc: DuckDBPyConnection, source: DuckDBPyRelation) -> Optional[str]:
        """The start of the file as it will be written, with the delimiter, quoting, header and order that it
        has been set up to have, in place of the table of rows which something which writes a file hasn't got"""
        total = len(source)
        text = csv_text(
            ddbc,
            self.sorted(source).limit(CSV_PREVIEW_ROWS),
            sep=self.SEPARATORS[self.delimiter.value],
            quoting=self.QUOTING[self.quoting.value],
            header=bool(self.header),
        )
        hidden = total - CSV_PREVIEW_ROWS
        if hidden > 0:
            text += f"… and {hidden:,} more {'row' if hidden == 1 else 'rows'}\n"
        return text

    def execute(
        self, ddbc: DuckDBPyConnection, source: Optional[DuckDBPyRelation], row_limit: Optional[int] = None
    ) -> None:
        if source is None or row_limit is not None or not self.filename:
            return

        source = self.sorted(source)

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
