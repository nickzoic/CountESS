import logging
from typing import Iterable, Optional

from countess import VERSION
from countess.core.parameters import BooleanParam, PerColumnArrayParam, StringParam
from countess.core.plugins import DuckdbSqlPlugin
from countess.utils.sql import Id, Sql, join

logger = logging.getLogger(__name__)


class UnpivotPlugin(DuckdbSqlPlugin):
    """UnPivots (Melts) a Table"""

    name = "Unpivot Tool"
    description = "Unpivots (Melts) Tables"
    version = VERSION
    link = "https://countess-project.github.io/CountESS/included-plugins/#unpivot-tool"

    columns = PerColumnArrayParam("Columns", BooleanParam("Melt?", default=False))
    name_column = StringParam("Name Column")
    value_column = StringParam("Value Column")
    omit_blanks = BooleanParam("Omit Blanks?")

    def sql(self, table_name: str, columns: Iterable[str]) -> Optional[Sql]:
        column_set = set(columns)
        melt_cols = [p.label for p in self.columns if p.value and p.label in column_set]
        value_col = Id(self.value_column.value or "value")

        query = Sql.template(
            "UNPIVOT {table} ON {melt} INTO NAME {name} VALUE {value}",
            table=Id(table_name),
            melt=join(Id(c) for c in melt_cols),
            name=Id(self.name_column.value or "name"),
            value=value_col,
        )

        if self.omit_blanks:
            query = Sql.template("SELECT * FROM ({query}) WHERE {value}::string != ''", query=query, value=value_col)

        return query
