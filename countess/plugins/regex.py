import logging
from typing import Optional

from duckdb import DuckDBPyConnection, DuckDBPyRelation

from countess import VERSION
from countess.core.parameters import (
    ArrayParam,
    BooleanParam,
    ColumnChoiceParam,
    DataTypeChoiceParam,
    MultiParam,
    StringParam,
)
from countess.core.plugins import DuckdbSimplePlugin
from countess.utils.sql import Fragment, Id, Lit, Raw, Sql, Type, join

logger = logging.getLogger(__name__)


class OutputColumnsMultiParam(MultiParam):
    name = StringParam("Column Name")
    datatype = DataTypeChoiceParam("Column Type", "STRING")


class RegexToolPlugin(DuckdbSimplePlugin):
    name = "Regex Tool"
    description = "Apply regular expressions to a column to make new column(s)"
    link = "https://countess-project.github.io/CountESS/included-plugins/#regex-tool"
    version = VERSION

    column = ColumnChoiceParam("Input Column")
    regex = StringParam("Regular Expression", ".*")
    output = ArrayParam("Output Columns", OutputColumnsMultiParam("Col"))
    drop_column = BooleanParam("Drop Column", False)
    drop_unmatch = BooleanParam("Drop Unmatched Rows", False)

    def execute(
        self, ddbc: DuckDBPyConnection, source: DuckDBPyRelation, row_limit: Optional[int] = None
    ) -> Optional[DuckDBPyRelation]:
        outputs = [o for o in self.output if o.name.value]

        proj: Fragment
        if self.drop_column:
            proj = join(Id(c) for c in source.columns if c != self.column.value)
        else:
            proj = Raw("*")

        if not (outputs or self.drop_unmatch):
            # nothing which follows needs the column or the pattern, so there's no need for them to have been
            # chosen yet (and until they have, the tool just passes everything on)
            logger.debug("RegexToolPlugin proj %s", proj)
            return source.project(str(proj))

        column = Id(self.column.value)
        pattern = Lit(self.regex.value)

        if outputs:
            extract = Sql.template(
                "unnest(try_cast(regexp_extract({column}, {pattern}, [{names}]) AS struct({types})))",
                column=column,
                pattern=pattern,
                names=join(Lit(o.name.value) for o in outputs),
                types=join(
                    Sql.template("{name} {type}", name=Id(o.name.value), type=Type(o.datatype.value)) for o in outputs
                ),
            )
            proj = join([part for part in (proj, extract) if part])

        logger.debug("RegexToolPlugin proj %s", proj)

        if self.drop_unmatch:
            filt = Sql.template("regexp_matches({column}, {pattern})", column=column, pattern=pattern)
            logger.debug("RegexToolPlugin filt %s", filt)
            return source.filter(str(filt)).project(str(proj))

        return source.project(str(proj))
