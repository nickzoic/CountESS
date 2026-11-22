import logging
from typing import Iterable, Optional

from countess import VERSION
from countess.core.parameters import ArrayParam, ColumnChoiceParam, NumericColumnChoiceParam, ScalarParam
from countess.core.plugins import DuckdbSqlPlugin
from countess.utils.sql import EMPTY, Id, Sql, join

logger = logging.getLogger(__name__)


class RankingPlugin(DuckdbSqlPlugin):
    name = "Ranking"
    description = "Rank rows by column value(s)"
    version = VERSION

    order_by = ArrayParam("Order By", NumericColumnChoiceParam("Column"))
    partition = ArrayParam("Partition By", ColumnChoiceParam("Column"))

    def sql(self, table_name: str, columns: Iterable[str]) -> Optional[Sql]:
        order_by = [Id(p.value) for p in self.order_by.params if isinstance(p, ScalarParam)]
        partition = [Id(p.value) for p in self.partition.params if isinstance(p, ScalarParam)]

        return Sql.template(
            "SELECT *, percent_rank({order_by}) OVER ({partition}) AS rank FROM {table}",
            order_by=Sql.template("ORDER BY {columns}", columns=join(order_by)) if order_by else EMPTY,
            partition=Sql.template("PARTITION BY {columns}", columns=join(partition)) if partition else EMPTY,
            table=Id(table_name),
        )
