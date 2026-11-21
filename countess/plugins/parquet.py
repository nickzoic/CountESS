import logging
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

import duckdb
from duckdb import DuckDBPyConnection, DuckDBPyRelation

from countess import VERSION
from countess.core.parameters import (
    ArrayParam,
    BaseParam,
    BooleanParam,
    ChoiceParam,
    ColumnChoiceParam,
    FileSaveParam,
    TabularMultiParam,
)
from countess.core.plugins import DuckdbLoadFileWithTheLotPlugin, DuckdbSaveFilePlugin
from countess.utils.duckdb import duckdb_escape_identifier

PARQUET_FILE_TYPES: Sequence[Tuple[str, Union[str, List[str]]]] = [
    ("Parquet", [".parquet", ".pq"]),
]

# The compression methods DuckDB can write which are widely supported by other readers.
# (DuckDB also has a legacy "LZ4" which is the same as LZ4_RAW as far as it is concerned,
# but other software tends not to cope with it, so it isn't offered.)
PARQUET_COMPRESSION_CHOICES = ["SNAPPY", "ZSTD", "GZIP", "BROTLI", "LZ4_RAW", "UNCOMPRESSED"]

# Parquet has no 128 bit integers, and so DuckDB writes these as DOUBLE ... which is quietly
# lossy, and turns a column of counts into floating point numbers.  They arise easily, because
# DuckDB's SUM() of any integers is a HUGEINT.
HUGE_INTEGER_TYPES = {"hugeint": "BIGINT", "uhugeint": "UBIGINT"}

logger = logging.getLogger(__name__)


class LoadParquetPlugin(DuckdbLoadFileWithTheLotPlugin):
    """Load Parquet files"""

    name = "Parquet Load"
    description = "Loads data from Apache Parquet files"
    link = "https://countess-project.github.io/CountESS/included-plugins/#parquet-reader"
    version = VERSION
    file_types = PARQUET_FILE_TYPES

    # Unlike a CSV file a Parquet file knows what its columns are called and what type they
    # are, so there's nothing to configure here but the files.

    def load_file(
        self, cursor: duckdb.DuckDBPyConnection, filename: str, file_param: BaseParam, row_limit: Optional[int] = None
    ) -> duckdb.DuckDBPyRelation:
        rel = cursor.read_parquet(filename)
        if row_limit is not None:
            rel = rel.limit(row_limit)
        return rel


class SaveParquetOrderParameter(TabularMultiParam):
    order_by = ColumnChoiceParam("Order By")
    descending = BooleanParam("Descending")


def parquet_compatible(source: DuckDBPyRelation) -> DuckDBPyRelation:
    """Convert any columns which Parquet can't store (see HUGE_INTEGER_TYPES) to the
    nearest thing which it can.  A value which doesn't fit is an error: better to
    stop than to write something which isn't what was asked for."""
    projection = []
    converted = []
    for column, dtype in zip(source.columns, source.dtypes):
        identifier = duckdb_escape_identifier(column)
        if dtype.id in HUGE_INTEGER_TYPES:
            converted.append(column)
            projection.append(f"CAST({identifier} AS {HUGE_INTEGER_TYPES[dtype.id]}) AS {identifier}")
        else:
            projection.append(identifier)

    if not converted:
        return source
    logger.info("Parquet can't store 128 bit integers, so %s will be saved as 64 bit", ", ".join(converted))
    return source.project(", ".join(projection))


class SaveParquetPlugin(DuckdbSaveFilePlugin):
    """Save Parquet files"""

    name = "Parquet Save"
    description = "Save data as Apache Parquet files"
    link = "https://countess-project.github.io/CountESS/included-plugins/#parquet-writer"
    version = VERSION
    file_types = PARQUET_FILE_TYPES

    filename = FileSaveParam("Filename", file_types=file_types)
    compression = ChoiceParam("Compression", "SNAPPY", choices=PARQUET_COMPRESSION_CHOICES)
    sorting = ArrayParam("Sorting", SaveParquetOrderParameter("sort"))

    show_preview = False

    def execute(
        self, ddbc: DuckDBPyConnection, source: Optional[DuckDBPyRelation], row_limit: Optional[int] = None
    ) -> None:
        # (nothing is written if there's a row limit, as that means it's only a preview)
        if source is None or row_limit is not None or not self.filename:
            return

        # (this is done before sorting so the sorting doesn't depend on rows staying in order through a projection)
        source = parquet_compatible(source)

        if len(self.sorting):
            order_by = ",".join(
                duckdb_escape_identifier(sp.order_by.value) + (" desc" if sp.descending else "") for sp in self.sorting
            )
            logger.debug("SaveParquetPlugin.execute order_by %s", order_by)
            source = source.order(order_by)

        filename = self.filename.get_file_path()
        Path(filename).parent.mkdir(parents=True, exist_ok=True)
        logger.debug("SaveParquetPlugin.execute writing file %s", filename)

        # DuckDB writes to a temporary file if there's already a file of that name, so
        # something which goes wrong part way through leaves the old file as it was.
        source.to_parquet(filename, compression=self.compression.value.lower())
