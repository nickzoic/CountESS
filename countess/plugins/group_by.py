import logging
from typing import Iterable, Optional

from countess import VERSION
from countess.core.parameters import BooleanParam, PerColumnArrayParam, TabularMultiParam
from countess.core.plugins import DuckdbSqlPlugin
from countess.utils.sql import Fn, Id, Raw, Sql, join

logger = logging.getLogger(__name__)


class ColumnMultiParam(TabularMultiParam):
    index = BooleanParam("Index")
    count = BooleanParam("Count")
    nunique = BooleanParam("Count Distinct")
    min = BooleanParam("Min")
    max = BooleanParam("Max")
    sum = BooleanParam("Sum")
    mean = BooleanParam("Mean")
    median = BooleanParam("Median")
    var_pop = BooleanParam("Var")


def _op(op_name: str, col_name: str) -> Sql:
    column = Id(col_name)
    if op_name == "nunique":
        call = Sql.template("COUNT(DISTINCT {column})", column=column)
    else:
        call = Sql.template("{function}({column})", function=Fn(op_name.upper()), column=column)
    return Sql.template("{call} AS {output}", call=call, output=Id(col_name + "__" + op_name))


class GroupByPlugin(DuckdbSqlPlugin):
    """Groups by an arbitrary column and rolls up rows"""

    name = "Group By"
    description = "Group records by column(s) and calculate aggregates"
    version = VERSION
    link = "https://countess-project.github.io/CountESS/included-plugins/#group-by"
    additional = "Note: 'Var' is uncorrected population variance."

    columns = PerColumnArrayParam("Columns", ColumnMultiParam("Column"))
    join = BooleanParam("Join Back?")

    def sql(self, table_name: str, columns: Iterable[str]) -> Optional[Sql]:
        column_params = list(self.columns.get_column_params())
        aggregates = join(
            _op(op, col_name)
            for col_name, col_param in column_params
            for op, bp in col_param.params.items()
            if bp.value and op != "index"
        ) or Raw("count(*) AS count")
        group_by = join(Id(col_name) for col_name, col_param in column_params if col_param.params["index"].value)
        table = Id(table_name)

        if self.join:
            if group_by:
                return Sql.template(
                    "SELECT * FROM {table} JOIN (SELECT {group_by}, {aggregates} FROM {table} GROUP BY {group_by}) "
                    "USING ({group_by})",
                    table=table,
                    group_by=group_by,
                    aggregates=aggregates,
                )
            return Sql.template(
                "SELECT * FROM {table} CROSS JOIN (SELECT {aggregates} FROM {table})",
                table=table,
                aggregates=aggregates,
            )
        if group_by:
            return Sql.template(
                "SELECT {group_by}, {aggregates} FROM {table} GROUP BY {group_by}",
                group_by=group_by,
                aggregates=aggregates,
                table=table,
            )
        return Sql.template("SELECT {aggregates} FROM {table}", aggregates=aggregates, table=table)
