import logging
from typing import Iterable, Optional

import duckdb
import oxbow

from countess import VERSION
from countess.core.parameters import BaseParam, BooleanParam, FloatParam
from countess.core.plugins import DuckdbLoadFileWithTheLotPlugin
from countess.utils.compress import open_for_reading

logger = logging.getLogger(__name__)


class LoadFastqPlugin(DuckdbLoadFileWithTheLotPlugin):
    """Load counts from one or more FASTQ files, by first building a dask dataframe of raw sequences
    with count=1 and then grouping by sequence and summing counts.  It supports counting
    in multiple columns."""

    name = "FASTQ Load"
    description = "Loads counts from FASTQ files containing either variant or barcodes"
    link = "https://countess-project.github.io/CountESS/included-plugins/#fastq-load"
    version = VERSION

    file_types = [("FASTQ", [".fastq", ".fastq.gz", ".fastq.bz2", ".fastq.xz", ".fq", ".fq.gz", ".fq.bz2", ".fq.xz"])]

    min_avg_quality = FloatParam("Minimum Average Quality", 10)
    min_low_quality = FloatParam("Minimum Lowest Quality", 0)
    header_column = BooleanParam("Header Column?", False)
    group = BooleanParam("Group by Sequence?", True)

    def load_file(
        self, cursor: duckdb.DuckDBPyConnection, filename: str, file_param: BaseParam, row_limit: Optional[int] = None
    ) -> duckdb.DuckDBPyRelation:
        logger.debug("Loading file %s row_limit %s", filename, row_limit)

        fields = ["sequence"]
        if self.min_avg_quality > 0 or self.min_low_quality > 0:
            fields.append("quality")
        if self.header_column:
            fields.append("name")

        if filename.lower().endswith((".bz2", ".xz")):
            reader = oxbow.from_fastq(lambda: open_for_reading(filename), None, fields=fields)
        else:
            reader = oxbow.from_fastq(filename, fields=fields)

        rel = reader.to_duckdb(cursor)

        if row_limit:
            rel = rel.limit(row_limit)

        if self.min_avg_quality > 0:
            filt = "list_avg(list_transform(split(quality,''), lambda x: ord(x))) >= %d" % (self.min_avg_quality + 33)
            rel = rel.filter(filt)

        if self.min_low_quality > 0:
            filt = "list_min(list_transform(split(quality,''), lambda x: ord(x))) >= %d" % (self.min_low_quality + 33)
            rel = rel.filter(filt)

        if self.header_column:
            rel = rel.project("sequence, name as header")
        else:
            rel = rel.project("sequence")

        if self.group:
            rel = rel.aggregate("sequence, count(*) as count")

        return rel

    def combine(
        self, ddbc: duckdb.DuckDBPyConnection, tables: Iterable[duckdb.DuckDBPyRelation]
    ) -> Optional[duckdb.DuckDBPyRelation]:
        combined_view = super().combine(ddbc, tables)
        if combined_view is None:
            return None
        elif not self.group or self.filename_column or self.header_column:
            return combined_view
        else:
            return combined_view.aggregate("sequence, sum(count) as count")


class LoadFastaPlugin(DuckdbLoadFileWithTheLotPlugin):
    name = "FASTA Load"
    description = "Loads sequences from FASTA files"
    link = "https://countess-project.github.io/CountESS/included-plugins/#fasta-load"
    version = VERSION

    file_types = [("FASTA", [".fasta", ".fa", ".fasta.gz", ".fa.gz", ".fasta.bz2", ".fa.bz2", ".fasta.xz", ".fa.xz"])]

    def load_file(
        self, cursor: duckdb.DuckDBPyConnection, filename: str, file_param: BaseParam, row_limit: Optional[int] = None
    ) -> duckdb.DuckDBPyRelation:
        if filename.lower().endswith((".bz2", ".xz")):
            reader = oxbow.from_fasta(lambda: open_for_reading(filename))
        else:
            reader = oxbow.from_fasta(filename)

        rel = reader.to_duckdb(cursor)
        return rel.limit(row_limit) if row_limit else rel
