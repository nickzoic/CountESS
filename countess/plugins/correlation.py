import logging
from typing import Iterable, Optional

from countess import VERSION
from countess.core.parameters import BooleanParam, ColumnOrNoneChoiceParam, PerNumericColumnArrayParam, ScalarParam
from countess.core.plugins import DuckdbSqlPlugin
from countess.utils.sql import EMPTY, Id, Lit, Sql, join

logger = logging.getLogger(__name__)


class CorrelationPlugin(DuckdbSqlPlugin):
    """Correlations"""

    name = "Correlation Tool"
    description = "Measures correlation coefficient, covariance and Pearsons r² correlation of columns"
    version = VERSION
    link = "https://countess-project.github.io/CountESS/included-plugins/#correlation-tool"

    columns = PerNumericColumnArrayParam("Columns", BooleanParam("Correlate?", False))
    group = ColumnOrNoneChoiceParam("Group")

    def sql(self, table_name: str, columns: Iterable[str]) -> Optional[Sql]:
        columns_params = [cp for cp in self.columns.params if isinstance(cp, ScalarParam)]
        grp = Id(self.group.value) if self.group.is_not_none() else None

        if sum(1 for cp in columns_params if cp.value) < 2:
            return None

        # (the total over all the groups, which rollup adds, has a NULL for its group)
        group_select = Sql.template("{group}, ", group=grp) if grp else EMPTY
        group_by = Sql.template("GROUP BY ROLLUP({group})", group=grp) if grp else EMPTY

        return join(
            (
                Sql.template(
                    "SELECT {group_select}{label_x} AS column_x, {label_y} AS column_y, "
                    "corr({y}, {x}) AS correlation_coefficient, covar_pop({y}, {x}) AS covariance_population "
                    "FROM {table} WHERE {x} IS NOT NULL AND {y} IS NOT NULL {group_by}",
                    group_select=group_select,
                    label_x=Lit(c1.label),
                    label_y=Lit(c2.label),
                    x=Id(c1.label),
                    y=Id(c2.label),
                    table=Id(table_name),
                    group_by=group_by,
                )
                for c1 in columns_params
                for c2 in columns_params
                if c1.value and c2.value and c1.label < c2.label
            ),
            sep=" UNION ALL ",
        )
