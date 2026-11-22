import logging
from typing import Iterable, Optional

from duckdb import DuckDBPyConnection, DuckDBPyRelation

from countess import VERSION
from countess.core.parameters import FloatParam, MultiColumnChoiceParam, PerNumericColumnArrayParam, TabularMultiParam
from countess.core.plugins import DuckdbSqlPlugin
from countess.utils.sql import EMPTY, Id, Lit, Raw, Sql, join

logger = logging.getLogger(__name__)


class CountColumnParam(TabularMultiParam):
    weight = FloatParam("Weight")


class VampSeqScorePlugin(DuckdbSqlPlugin):
    name = "VAMP-seq Scoring"
    description = "Calculate scores from weighed bin counts"
    version = VERSION

    columns = PerNumericColumnArrayParam("Columns", CountColumnParam("Column"))
    group_by = MultiColumnChoiceParam("Group By")
    threshold = FloatParam("Total Frequency Threshold", 1.78e-5)

    def prepare(self, ddbc: DuckDBPyConnection, source: Optional[DuckDBPyRelation]) -> None:
        super().prepare(ddbc, source)

        # set default values for weights on "count" columns
        if all(c.weight.value is None for c in self.columns):
            count_cols = [c for c in self.columns if c.label.startswith("count")]
            for n, c in enumerate(count_cols):
                c.weight.value = (n + 1) / len(count_cols)

        weight_cols = set(n for n, p in self.columns.get_column_params() if p.weight.value is not None)
        if source:
            self.group_by.set_choices([c for c in source.columns if c not in weight_cols])

    def sql(self, table_name: str, columns: Iterable[str]) -> Optional[Sql]:
        group_names = sorted(self.group_by.get_values())
        weights = {
            name: param.weight.value
            for name, param in self.columns.get_column_params()
            if param.weight.value is not None
        }

        if not weights:
            return None

        # Var is the table, Tot is the total of each of the counts over each group (or everything)
        inner_select = join(
            [Id.dotted("T0", name) for name in group_names]
            + [Sql.template("sum({count}) AS {total}", count=Id.dotted("T0", name), total=Id(name)) for name in weights]
        )

        total_freq = Sql.template(
            "({counts}) / ({totals})",
            counts=join((Id.dotted("Var", name) for name in weights), sep=" + "),
            totals=join((Id.dotted("Tot", name) for name in weights), sep=" + "),
        )

        weighted_freqs = join(
            (
                Sql.template(
                    "CASE WHEN {tot} > 0 THEN {var} * {weight} / {tot} ELSE 0 END",
                    tot=Id.dotted("Tot", name),
                    var=Id.dotted("Var", name),
                    weight=Lit(weight),
                )
                for name, weight in weights.items()
            ),
            sep=" + ",
        )
        unweighted_freqs = join(
            (
                Sql.template(
                    "CASE WHEN {tot} > 0 THEN {var} / {tot} ELSE 0 END",
                    tot=Id.dotted("Tot", name),
                    var=Id.dotted("Var", name),
                )
                for name in weights
            ),
            sep=" + ",
        )
        group_by = (
            Sql.template("GROUP BY {cols}", cols=join(Id(name) for name in group_names)) if group_names else EMPTY
        )
        join_on = (
            join(
                (
                    Sql.template("{var} = {tot}", var=Id.dotted("Var", name), tot=Id.dotted("Tot", name))
                    for name in group_names
                ),
                sep=" AND ",
            )
            if group_names
            else Raw("1=1")
        )

        return Sql.template(
            """
            SELECT Var.*, ({weighted_freqs}) / ({unweighted_freqs}) AS score, {total_freq} AS total_freq
            FROM {table} Var JOIN (
                SELECT {inner_select}
                FROM {table} T0
                {group_by}
            ) Tot ON {join_on}
            WHERE total_freq > {threshold}
            """,
            weighted_freqs=weighted_freqs,
            unweighted_freqs=unweighted_freqs,
            total_freq=total_freq,
            table=Id(table_name),
            inner_select=inner_select,
            group_by=group_by,
            join_on=join_on,
            threshold=Lit(self.threshold.value),
        )
