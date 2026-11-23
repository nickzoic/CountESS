import logging
import secrets
from math import log, nan, sqrt
from typing import Optional

from duckdb import DuckDBPyConnection, DuckDBPyRelation
from duckdb.sqltypes import DuckDBPyType

from countess import VERSION
from countess.core.parameters import BooleanParam, ColumnGroupChoiceParam, ColumnOrNoneChoiceParam, StringParam
from countess.core.plugins import DuckdbSimplePlugin
from countess.utils.duckdb import duckdb_dtype_is_boolean, duckdb_dtype_is_numeric
from countess.utils.sql import EMPTY, Fn, Fragment, Id, Lit, Raw, Sql, join

logger = logging.getLogger(__name__)


def _wls_slope(xs: list[float], ys: list[float], ws: list[float]) -> tuple[float, float]:
    """Slope and standard error of the slope of the weighted least squares line y = slope * x + intercept.

    This is the closed form of weighted least squares for a straight line (two parameters): centre x and y on
    their weighted means, then

        slope = sum(w (x - xbar) (y - ybar)) / sum(w (x - xbar)^2)
        var(slope) = (sum(w residual^2) / (n - 2)) / sum(w (x - xbar)^2)

    Working from the centred values rather than the raw sums of w, wx, wxx, ... avoids catastrophic
    cancellation when the x values are large compared to their spread.  Building and decomposing a
    general purpose regression model for every variant was the single largest cost of scoring a typical
    experiment.

    There's no line to be had from fewer than two points, from points with no weight, or from points which
    all have the same x, so those give NaN for both.  The standard error also needs a degree of freedom to
    spare (n - 2 > 0), so two points give a slope but a NaN standard error.
    """

    if len(xs) < 2:
        return nan, nan
    sw = sum(ws)
    if not sw > 0.0:
        return nan, nan
    xbar = sum(w * x for w, x in zip(ws, xs)) / sw
    ybar = sum(w * y for w, y in zip(ws, ys)) / sw
    sxx = sum(w * (x - xbar) ** 2 for w, x in zip(ws, xs))
    if not sxx > 0.0:
        return nan, nan
    sxy = sum(w * (x - xbar) * (y - ybar) for w, x, y in zip(ws, xs, ys))
    slope = sxy / sxx
    if len(xs) < 3:
        return slope, nan
    intercept = ybar - slope * xbar
    rss = sum(w * (y - slope * x - intercept) ** 2 for w, x, y in zip(ws, xs, ys))
    return slope, sqrt(rss / (len(xs) - 2) / sxx)


def score_function(times: list[float], counts: list[float], populations: list[float]) -> Optional[tuple[float, float]]:
    """
    Called with three equal-length lists of floats:
    * `times` is the time points
    * `counts` is the variant count at that point
    * `populations` is the population count at that point

    Apply the formulae from the Enrich2 paper and then perform
    a linear least squares regression to xs and ys with option weights ws
    and return the 'slope' and the stderr of slope.
    """

    if None in times or None in counts or None in populations:
        return None
    assert len(times) == len(counts) == len(populations) > 1
    assert all(0 <= c <= p for c, p in zip(counts, populations))

    if len(times) == 2:
        # 2 time point estimate
        score = log((counts[1] + 0.5) / (populations[1] + 0.5)) - log((counts[0] + 0.5) / (populations[0] + 0.5))
        sigma = sqrt(
            (1 / (counts[0] + 0.5) + 1 / (counts[1] + 0.5) + 1 / (populations[0] + 0.5) + 1 / (populations[1] + 0.5))
        )
    else:
        # multi time point least squares regression
        ys = [log((c + 0.5) / (p + 0.5)) for c, p in zip(counts, populations)]
        ws = [1 / ((1 / (c + 0.5)) + (1 / (p + 0.5))) for c, p in zip(counts, populations)]
        # we only care about the slope, not the "intercept" of the line.
        score, sigma = _wls_slope(times, ys, ws)

    return score, sigma


class ScoringPlugin(DuckdbSimplePlugin):
    name = "Scoring"
    description = "Score variants using counts or frequencies"
    version = VERSION
    link = "https://countess-project.github.io/CountESS/included-plugins/#scoring"

    replicate = ColumnOrNoneChoiceParam("Replicate Column")
    columns = ColumnGroupChoiceParam("Input Columns")
    wildtype = ColumnOrNoneChoiceParam("Wildtype Indicator")
    output = StringParam("Score Column", "score")
    stddev = StringParam("Standard Deviation Column", "sigma")
    drop_input = BooleanParam("Drop Input Columns?", False)

    function_name = None

    def prepare(self, ddbc: DuckDBPyConnection, source: Optional[DuckDBPyRelation]) -> None:
        super().prepare(ddbc, source)

        if not self.function_name:
            self.function_name = "f_" + secrets.token_hex(16)
            logger.debug("Creating function %s for plugin %s", self.function_name, self.name)
            ddbc.create_function(  # type: ignore[call-overload]
                self.function_name,
                score_function,
                return_type=DuckDBPyType("DOUBLE[]"),
                null_handling="special",
            )

    def execute(
        self, ddbc: DuckDBPyConnection, source: DuckDBPyRelation, row_limit: Optional[int] = None
    ) -> Optional[DuckDBPyRelation]:
        yaxis_prefix = self.columns.get_column_prefix()
        suffix_set = {k.removeprefix(yaxis_prefix) for k in source.columns if k.startswith(yaxis_prefix)}
        suffixes = sorted(suffix_set)

        count_names = [yaxis_prefix + suffix for suffix in suffixes]

        logger.debug("ScoringPlugin.execute count_names %s", count_names)

        try:
            x_values = [float(x) for x in suffixes]
        except ValueError:
            x_values = list(range(0, len(suffixes)))
        x_max = max(x_values)

        x_cols = [Lit(x / x_max) for x in x_values]

        logger.debug("ScoringPlugin.execute x_cols %s", x_cols)

        # the counts are compared with those of the whole population, or only of those which are wildtype
        where_clause: Fragment = EMPTY
        if self.wildtype.is_not_none():
            wtcol = Id(self.wildtype.value)
            for col, dtype in zip(source.columns, source.dtypes):
                if col == self.wildtype.value:
                    if duckdb_dtype_is_boolean(dtype) or duckdb_dtype_is_numeric(dtype):
                        where_clause = Sql.template("WHERE {column}", column=wtcol)
                    else:
                        where_clause = Sql.template("WHERE {column} IN ('W', 'p.=', 'g.=')", column=wtcol)

        keep_fields: Fragment
        if self.drop_input:
            keep_fields = join(Id.dotted("A", f) for f in source.columns if f not in count_names)
        else:
            keep_fields = Raw("A.*")

        # with replicates each is compared with the population of its own replicate
        join_op: Fragment
        repl_select: Fragment
        group_clause: Fragment
        join_using: Fragment
        if self.replicate.is_none():
            join_op, repl_select, group_clause, join_using = Raw("CROSS JOIN"), EMPTY, EMPTY, EMPTY
        else:
            repl = Id(self.replicate.value)
            join_op = Raw("JOIN")
            repl_select = Sql.template("{repl}, ", repl=repl)
            group_clause = Sql.template("GROUP BY {repl}", repl=repl)
            join_using = Sql.template("USING ({repl})", repl=repl)

        query = Sql.template(
            """
            SELECT * EXCLUDE _SCORE, _SCORE[1] AS {output}, _SCORE[2] AS {stddev}
            FROM (
                SELECT {keep_fields}, {function}([{x_cols}], [{a_counts}], [{b_counts}]) AS _SCORE
                FROM {source} AS A {join_op} (
                    SELECT {repl_select}{sums} FROM {source} {where_clause} {group_clause}
                ) AS B {join_using}
            )
            """,
            output=Id(self.output.value),
            stddev=Id(self.stddev.value),
            keep_fields=keep_fields,
            function=Fn(self.function_name),
            x_cols=join(x_cols),
            a_counts=join(Id.dotted("A", name) for name in count_names),
            b_counts=join(Id.dotted("B", name) for name in count_names),
            source=Id(source.alias),
            join_op=join_op,
            repl_select=repl_select,
            sums=join(Sql.template("SUM({count}) AS {count}", count=Id(name)) for name in count_names),
            where_clause=where_clause,
            group_clause=group_clause,
            join_using=join_using,
        )

        logger.debug("ScoringPlugin.execute query %s", query)
        return query.relation(ddbc)
