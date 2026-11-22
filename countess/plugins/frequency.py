import logging
from typing import Iterable, Mapping, Optional

from countess import VERSION
from countess.core.parameters import BooleanParam, MultiColumnChoiceParam, NumericMultiColumnChoiceParam
from countess.core.plugins import DuckdbSqlPlugin
from countess.utils.sql import Id, Sql, join

logger = logging.getLogger(__name__)


class FrequencyPlugin(DuckdbSqlPlugin):
    name = "Frequency Calculator"
    description = "Calculate frequencies from counts"
    version = VERSION

    columns = NumericMultiColumnChoiceParam("Count Columns")
    group_cols = MultiColumnChoiceParam("Group By")
    sigma = BooleanParam("Output Sigmas?")

    def set_column_choices(self, choices: Mapping[str, bool]):
        self.columns.set_column_choices(choices)
        count_cols = self.columns.get_values()
        if not count_cols:
            count_cols = set(c for c in choices if c.startswith("count"))
            self.columns.set_values(count_cols)

        self.group_cols.set_column_choices({c: n for c, n in choices.items() if c not in count_cols})

    def sql(self, table_name: str, columns: Iterable[str]) -> Optional[Sql]:
        if not self.columns.get_values():
            return None

        columns = sorted(self.columns.get_values())
        # whatever -> freq_whatever; count -> freq_count; count_N -> freq_N; etc
        suffixes = [cc.removeprefix("count_") for cc in columns]

        # X is the table, Y is the total of each count column (S0, S1, ...) over its group, or over everything
        totals = [Id.dotted("Y", f"S{n}") for n in range(len(columns))]
        sums = join(
            Sql.template("sum({count}) AS {total}", count=Id(cc), total=Id(f"S{n}")) for n, cc in enumerate(columns)
        )

        outputs = [
            Sql.template(
                "CASE WHEN {total} > 0 THEN {count} / {total} ELSE 0 END AS {freq}",
                total=total,
                count=Id.dotted("X", cc),
                freq=Id("freq_" + suffix),
            )
            for cc, suffix, total in zip(columns, suffixes, totals)
        ]
        if self.sigma.value:
            outputs += [
                Sql.template(
                    "SQRT({count} + 0.5) / {total} AS {sigma}",
                    count=Id.dotted("X", cc),
                    total=total,
                    sigma=Id("sigma_" + suffix),
                )
                for cc, suffix, total in zip(columns, suffixes, totals)
            ]

        group_by = join(Id(gc) for gc in self.group_cols.get_values() if gc not in columns)
        table = Id(table_name)

        if group_by:
            return Sql.template(
                "SELECT X.*, {outputs} FROM {table} X JOIN "
                "(SELECT {group_by}, {sums} FROM {table} GROUP BY {group_by}) Y USING ({group_by})",
                outputs=join(outputs),
                table=table,
                group_by=group_by,
                sums=sums,
            )
        return Sql.template(
            "SELECT X.*, {outputs} FROM {table} X CROSS JOIN (SELECT {sums} FROM {table}) Y",
            outputs=join(outputs),
            table=table,
            sums=sums,
        )
