import logging
from typing import Iterable, Optional

from countess import VERSION
from countess.core.parameters import (
    ArrayParam,
    ChoiceParam,
    ColumnChoiceParam,
    MultiColumnChoiceParam,
    NumericColumnChoiceParam,
    StringParam,
    TabularMultiParam,
)
from countess.core.plugins import DuckdbSqlPlugin
from countess.utils.sql import EMPTY, Fn, Fragment, Id, Lit, Raw, Sql, join

logger = logging.getLogger(__name__)


# how each of the operations makes a test of a column {col} against a value {val}
FILTER_TEMPLATES = {
    "Equals": "{col} = {val}",
    "Starts With": "starts_with({col},{val})",
    "Ends With": "ends_with({col},{val})",
    "Contains": "contains({col},{val})",
    "Matches": "regexp_full_match({col},{val})",
    "Greater Than": "({col} > {val})",
    "Less Than": "({col} < {val})",
}


class ScaleClassParam(TabularMultiParam):
    agg = ChoiceParam("Aggregation", "Median", ["Median", "Avg"])
    col = ColumnChoiceParam("Column")
    op = ChoiceParam(
        "Operation",
        "Equals",
        ["Equals", "Starts With", "Ends With", "Contains", "Matches", "Greater Than", "Less Than"],
    )
    st = StringParam("Value")

    def filter(self) -> Sql:
        try:
            template = FILTER_TEMPLATES[self.op.value]
        except KeyError as exc:
            # (before looking at the column and value: whatever they are, an operation which isn't known is a
            # mistake in the code and not something which is yet to be filled in)
            raise NotImplementedError(self.op.value) from exc
        return Sql.template(template, col=Id(self.col.value), val=Lit(self.st.value))


class ScoreScalingPlugin(DuckdbSqlPlugin):
    name = "Score Scaling"
    description = "Scaled Scores using variant classification"
    version = VERSION

    score_col = NumericColumnChoiceParam("Score Column")
    scaled_col = StringParam("Scaled Score Column", "scaled_score")
    classifiers = ArrayParam("Variant Classifiers", ScaleClassParam("Class"), min_size=2, max_size=2, read_only=True)
    group_col = MultiColumnChoiceParam("Group By")

    def __init__(self, *a, **k):
        super().__init__(*a, **k)

        # override classifiers labels
        self.classifiers[0].label = "Scale to 0.0"
        self.classifiers[1].label = "Scale to 1.0"

    def sql(self, table_name: str, columns: Iterable[str]) -> Optional[Sql]:
        score = Id(self.score_col.value)
        scaled = Id(self.scaled_col.value)

        all_columns = join(Id.dotted("T1", c) for c in columns if self.scaled_col != c)

        group_cols = join(Id(v) for v in sorted(self.group_col.get_values()))
        join_method: Fragment
        if group_cols:
            group_select = Sql.template("{cols}, ", cols=group_cols)
            group_by = Sql.template("GROUP BY {cols}", cols=group_cols)
            join_method = Sql.template("USING ({cols})", cols=group_cols)
        else:
            group_select = EMPTY
            group_by = EMPTY
            join_method = Raw("ON (1=1)")

        c0, c1 = self.classifiers

        # score_0 and score_1 are where the scores of the two classes are (and without any members of a class
        # they're 0 and 1), as T2 has for each group, and the scores are scaled so those are 0 and 1.
        return Sql.template(
            """
            SELECT {all_columns}, ({score} - T2.score_0) / (T2.score_1 - T2.score_0) AS {scaled}
            FROM {table} T1 JOIN (
                SELECT {group_select}
                    coalesce({agg0}({score}) FILTER ({filter0}), 0) AS score_0,
                    coalesce({agg1}({score}) FILTER ({filter1}), 1) AS score_1
                FROM {table}
                {group_by}
            ) T2 {join_method}
            """,
            all_columns=all_columns,
            score=score,
            scaled=scaled,
            table=Id(table_name),
            group_select=group_select,
            agg0=Fn(c0.agg.value),
            filter0=c0.filter(),
            agg1=Fn(c1.agg.value),
            filter1=c1.filter(),
            group_by=group_by,
            join_method=join_method,
        )
