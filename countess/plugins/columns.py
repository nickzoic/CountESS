import logging
from typing import Iterable, Optional

from countess import VERSION
from countess.core.parameters import DataTypeOrNoneChoiceParam, PerColumnArrayParam, StringParam, TabularMultiParam
from countess.core.plugins import DuckdbSqlPlugin
from countess.utils.sql import Id, Sql, Type, join

logger = logging.getLogger(__name__)


class ColumnMultiParam(TabularMultiParam):
    name = StringParam("Rename")
    ttype = DataTypeOrNoneChoiceParam("Column Type")


class ColumnsPlugin(DuckdbSqlPlugin):
    name = "Columns"
    description = "Rename, remove or cast columns to a different type"
    version = VERSION

    columns = PerColumnArrayParam("Columns", ColumnMultiParam("Column"))

    def sql(self, table_name: str, columns: Iterable[str]) -> Optional[Sql]:
        cols = join(
            Sql.template(
                "TRY_CAST({column} AS {type}) AS {name}",
                column=Id(n),
                type=Type(p.ttype.value),
                name=Id(p.name.value or n),
            )
            for n, p in self.columns.get_column_params()
            if p.ttype.is_not_none()
        )
        if not cols:
            return None

        return Sql.template("SELECT {cols} FROM {table}", cols=cols, table=Id(table_name))
