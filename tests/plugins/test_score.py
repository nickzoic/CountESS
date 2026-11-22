import math

import duckdb
import numpy as np
import pytest

from countess.plugins.score import ScoringPlugin, _wls_slope, _wls_statsmodels, score_function


def wls_slope(times, counts, populations):
    """Enrich2's score for three or more time points: the slope of the weighted least squares line through
    log((count + 0.5) / (population + 0.5)), and the standard error of that slope, worked out directly from the
    normal equations."""
    x = np.array([[t, 1.0] for t in times])
    y = np.array([math.log((c + 0.5) / (p + 0.5)) for c, p in zip(counts, populations)])
    w = np.array([1 / (1 / (c + 0.5) + 1 / (p + 0.5)) for c, p in zip(counts, populations)])
    xtwx_inverse = np.linalg.inv(x.T @ np.diag(w) @ x)
    beta = xtwx_inverse @ x.T @ np.diag(w) @ y
    residuals = y - x @ beta
    scale = (w * residuals**2).sum() / (len(times) - 2)
    return float(beta[0]), float(math.sqrt(scale * xtwx_inverse[0, 0]))


def two_point(counts, populations):
    score = math.log((counts[1] + 0.5) / (populations[1] + 0.5)) - math.log((counts[0] + 0.5) / (populations[0] + 0.5))
    sigma = math.sqrt(sum(1 / (v + 0.5) for v in (*counts, *populations)))
    return score, sigma


# ---- the score function


def test_two_time_points():
    assert score_function([0.0, 1.0], [10, 20], [100, 100]) == pytest.approx(two_point([10, 20], [100, 100]))
    assert score_function([0.0, 1.0], [10, 20], [100, 100])[0] == pytest.approx(math.log(20.5 / 100.5 * 100.5 / 10.5))


def test_two_time_points_ignores_what_the_times_are():
    assert score_function([0.0, 1.0], [3, 9], [50, 60]) == pytest.approx(score_function([5.0, 17.0], [3, 9], [50, 60]))


@pytest.mark.parametrize(
    "times,counts,populations",
    [
        ([0.0, 0.5, 1.0], [10, 20, 40], [100, 100, 100]),
        ([0.0, 1.0, 2.0, 3.0], [50, 40, 22, 9], [200, 180, 190, 210]),
        ([0.0, 0.25, 0.5, 1.0], [0, 3, 0, 8], [50, 60, 55, 70]),  # (zero counts are fine)
        ([0.0, 1.0, 3.0], [30, 30, 30], [100, 100, 100]),
    ],
)
def test_three_or_more_time_points(times, counts, populations):
    assert score_function(times, counts, populations) == pytest.approx(wls_slope(times, counts, populations), rel=1e-9)


def _wls_inputs(times, counts, populations):
    ys = [math.log((c + 0.5) / (p + 0.5)) for c, p in zip(counts, populations)]
    ws = [1 / (1 / (c + 0.5) + 1 / (p + 0.5)) for c, p in zip(counts, populations)]
    return times, ys, ws


def test_closed_form_agrees_with_statsmodels_on_random_data():
    """`_wls_slope` is a replacement for a statsmodels WLS fit, so check them against each other."""
    rng = np.random.default_rng(1)
    for _ in range(300):
        n = int(rng.integers(3, 13))
        times = sorted(rng.uniform(0, 1, n))
        populations = rng.integers(1, 2_000_000, n)
        counts = np.minimum(populations, rng.poisson(rng.uniform(0, 50), n))
        args = _wls_inputs(times, [int(c) for c in counts], [int(p) for p in populations])
        assert _wls_slope(*args) == pytest.approx(_wls_statsmodels(*args), rel=1e-8, abs=1e-12)


def test_closed_form_is_accurate_when_the_times_are_large_compared_to_their_spread():
    times = [1e6 + i for i in range(5)]
    args = _wls_inputs(times, [3, 8, 20, 41, 90], [1000] * 5)
    assert _wls_slope(*args) == pytest.approx(_wls_statsmodels(*args), rel=1e-8)


def test_closed_form_with_extreme_weights():
    args = _wls_inputs([0.0, 0.25, 0.5, 0.75, 1.0], [0, 0, 0, 1, 0], [10**9] * 5)
    assert _wls_slope(*args) == pytest.approx(_wls_statsmodels(*args), rel=1e-8)


def test_identical_times_have_no_slope_of_their_own_so_it_is_left_to_statsmodels():
    args = _wls_inputs([1.0, 1.0, 1.0], [1, 5, 9], [100, 100, 100])
    assert _wls_slope(*args) == pytest.approx(_wls_statsmodels(*args))
    assert score_function([1.0, 1.0, 1.0], [1, 5, 9], [100, 100, 100]) == pytest.approx(_wls_statsmodels(*args))


def test_a_variant_which_gets_more_common_scores_positive_and_one_which_dies_out_negative():
    assert score_function([0.0, 1.0, 2.0], [10, 20, 40], [1000, 1000, 1000])[0] > 0
    assert score_function([0.0, 1.0, 2.0], [40, 20, 10], [1000, 1000, 1000])[0] < 0
    assert score_function([0.0, 1.0, 2.0], [20, 20, 20], [1000, 1000, 1000])[0] == pytest.approx(0.0, abs=1e-12)


def test_more_data_means_a_smaller_sigma():
    _, small = score_function([0.0, 1.0, 2.0], [10, 20, 40], [1000, 1000, 1000])
    _, large = score_function([0.0, 1.0, 2.0], [1000, 2000, 4000], [100000, 100000, 100000])
    assert large < small


@pytest.mark.parametrize(
    "times,counts,populations",
    [([0.0, None], [1, 2], [10, 10]), ([0.0, 1.0], [1, None], [10, 10]), ([0.0, 1.0], [1, 2], [10, None])],
)
def test_missing_values_give_nothing(times, counts, populations):
    assert score_function(times, counts, populations) is None


@pytest.mark.parametrize(
    "times,counts,populations",
    [
        ([0.0], [1], [10]),  # (needs more than one time point)
        ([0.0, 1.0], [1, 2, 3], [10, 10, 10]),
        ([0.0, 1.0], [11, 2], [10, 10]),  # (more of the variant than there is of everything)
        ([0.0, 1.0], [-1, 2], [10, 10]),
    ],
)
def test_nonsense_is_refused(times, counts, populations):
    with pytest.raises(AssertionError):
        score_function(times, counts, populations)


# ---- scoring a table

COUNTS = {"v1": (10, 20, 40), "v2": (30, 25, 10), "v3": (60, 55, 50)}  # (so each column adds up to 100)
POPULATION = (100, 100, 100)


def counts_table(make_source, extra="", rows=None):
    rows = rows or COUNTS
    values = ", ".join(f"('{name}', {c[0]}, {c[1]}, {c[2]}{extra})" for name, c in rows.items())
    return make_source(
        f"select * from (values {values}) t(name, count_0, count_1, count_2{', ' if extra else ''}{extra and 'x'})"
    )


def score(ddbc, source, prefix="count_*", **settings):
    plugin = ScoringPlugin()
    plugin.prepare(ddbc, source)
    plugin.columns.set_value(prefix)
    for key, value in settings.items():
        getattr(plugin, key).value = value
    return plugin, plugin.execute(ddbc, source)


def test_scores_for_every_variant(ddbc, make_source, assert_rows):
    source = counts_table(make_source)
    _, output = score(ddbc, source)
    assert output.columns == ["name", "count_0", "count_1", "count_2", "score", "sigma"]
    times = [0.0, 0.5, 1.0]  # (the time points are the numbers in the column names, scaled so the last is 1)
    assert_rows(
        output,
        [(name, *counts, *wls_slope(times, counts, POPULATION)) for name, counts in COUNTS.items()],
    )


def test_the_time_points_are_scaled_by_the_last(ddbc, make_source, assert_rows):
    source = make_source(
        "select * from (values ('v1', 10, 20, 40), ('v2', 30, 25, 10), ('v3', 60, 55, 50)) t(name, count_0, count_5, count_20)"
    )
    _, output = score(ddbc, source)
    assert_rows(
        output,
        [(name, *counts, *wls_slope([0.0, 0.25, 1.0], counts, POPULATION)) for name, counts in COUNTS.items()],
    )


def test_two_time_points_in_a_table(ddbc, make_source, assert_rows):
    source = make_source("select * from (values ('v1', 10, 20), ('v2', 90, 80)) t(name, count_0, count_1)")
    _, output = score(ddbc, source)
    assert_rows(
        output,
        [("v1", 10, 20, *two_point([10, 20], [100, 100])), ("v2", 90, 80, *two_point([90, 80], [100, 100]))],
    )


def test_time_points_which_are_not_numbers_are_in_order_of_name(ddbc, make_source, assert_rows):
    source = make_source(
        "select * from (values ('v1', 10, 20, 40), ('v2', 30, 25, 10), ('v3', 60, 55, 50)) t(name, count_a, count_b, count_c)"
    )
    _, output = score(ddbc, source)
    assert_rows(
        output,
        [(name, *counts, *wls_slope([0.0, 0.5, 1.0], counts, POPULATION)) for name, counts in COUNTS.items()],
    )


def test_output_columns_can_be_named(ddbc, make_source):
    _, output = score(ddbc, counts_table(make_source), output="ünï score", stddev="ünï sd")
    assert output.columns[-2:] == ["ünï score", "ünï sd"]


def test_input_columns_can_be_dropped(ddbc, make_source, assert_rows):
    _, output = score(ddbc, counts_table(make_source), drop_input=True)
    assert output.columns == ["name", "score", "sigma"]
    assert len(output.fetchall()) == 3


def test_a_missing_count_gives_a_missing_score(ddbc, make_source, assert_rows):
    source = make_source(
        "select * from (values ('v1', 10, 20), ('v2', NULL, 80), ('v3', 90, 80)) t(name, count_0, count_1)"
    )
    _, output = score(ddbc, source)
    rows = {r[0]: r for r in output.fetchall()}
    assert rows["v2"][3:] == (None, None)
    populations = [100, 180]  # (a missing count isn't counted in the population)
    assert rows["v1"][3:] == pytest.approx(two_point([10, 20], populations))


# ---- replicates


def test_each_replicate_is_scored_against_its_own_population(ddbc, make_source, assert_rows):
    source = make_source(
        """select * from (values
            ('v1', 'r1', 10, 20), ('v2', 'r1', 90, 80), ('v1', 'r2', 5, 5), ('v2', 'r2', 15, 45)) t(name, rep, count_0, count_1)"""
    )
    _, output = score(ddbc, source, replicate="rep")
    assert_rows(
        output,
        [
            ("v1", "r1", 10, 20, *two_point([10, 20], [100, 100])),
            ("v2", "r1", 90, 80, *two_point([90, 80], [100, 100])),
            ("v1", "r2", 5, 5, *two_point([5, 5], [20, 50])),
            ("v2", "r2", 15, 45, *two_point([15, 45], [20, 50])),
        ],
    )


@pytest.mark.xfail(
    strict=True,
    reason="the replicates are joined back to the rows with USING, and NULL never equals NULL, so every row "
    "whose replicate is NULL is silently dropped from the output",
)
def test_rows_with_no_replicate_are_kept(ddbc, make_source):
    source = make_source(
        "select * from (values ('v1', 'r1', 10, 20), ('v2', 'r1', 90, 80), ('v1', NULL, 5, 5), ('v2', NULL, 15, 45)) t(name, rep, count_0, count_1)"
    )
    _, output = score(ddbc, source, replicate="rep")
    assert len(output.fetchall()) == 4


# ---- wildtype


WILDTYPE = [
    ("v1", 10, 20, True),
    ("v2", 30, 25, False),
    ("wt", 100, 90, True),
]


def wildtype_table(make_source, flag):
    rows = ", ".join(f"('{n}', {a}, {b}, {flag(w)})" for n, a, b, w in WILDTYPE)
    return make_source(f"select * from (values {rows}) t(name, count_0, count_1, wt)")


@pytest.mark.parametrize(
    "flag",
    [
        lambda w: "true" if w else "false",  # a boolean column
        lambda w: "1" if w else "0",  # a number
        lambda w: "'W'" if w else "'V'",  # text
        lambda w: "'p.='" if w else "'p.1A>T'",  # (protein change, no change)
        lambda w: "'g.='" if w else "'g.1A>T'",  # (genome change, no change)
    ],
    ids=["boolean", "numeric", "W", "p.=", "g.="],
)
def test_only_wildtype_rows_make_up_the_population(ddbc, make_source, assert_rows, flag):
    source = wildtype_table(make_source, flag)
    _, output = score(ddbc, source, wildtype="wt")
    population = [110, 110]  # the first and third rows, not all of them
    assert_rows(
        [r[:2] + r[-2:] for r in output.fetchall()],
        [
            ("v1", 10, *two_point([10, 20], population)),
            ("v2", 30, *two_point([30, 25], population)),
            ("wt", 100, *two_point([100, 90], population)),
        ],
    )


def test_without_a_wildtype_every_row_is_in_the_population(ddbc, make_source, assert_rows):
    source = wildtype_table(make_source, lambda w: "true" if w else "false")
    _, output = score(ddbc, source)
    population = [140, 135]
    assert_rows(
        [r[:2] + r[-2:] for r in output.fetchall()],
        [
            ("v1", 10, *two_point([10, 20], population)),
            ("v2", 30, *two_point([30, 25], population)),
            ("wt", 100, *two_point([100, 90], population)),
        ],
    )


def test_a_variant_with_more_counts_than_the_wildtype_population_is_an_error(ddbc, make_source):
    """This documents what happens now: the score function insists that a count is no more than its
    population, which is the whole table when there's no wildtype but only the wildtype rows when there is,
    so a variant which is more common than the wildtype brings the whole query down.  That may not be intended."""
    source = make_source(
        "select * from (values ('v1', 100, 200, false), ('wt', 4, 4, true)) t(name, count_0, count_1, wt)"
    )
    plugin, output = score(ddbc, source, wildtype="wt")
    with pytest.raises(duckdb.InvalidInputException, match="AssertionError"):
        output.fetchall()
    assert plugin.wildtype.value == "wt"


def test_the_function_is_made_once_per_plugin(ddbc, make_source):
    source = counts_table(make_source)
    plugin, _ = score(ddbc, source)
    name = plugin.function_name
    plugin.prepare(ddbc, source)
    assert plugin.function_name == name


def test_odd_column_names(ddbc, make_source, assert_rows):
    # (the columns are called  na me,  count "0  and  count "1 )
    source = make_source('select * from (values (\'x\', 10, 20), (\'y\', 90, 80)) t("na me", "count ""0", "count ""1")')
    plugin = ScoringPlugin()
    plugin.prepare(ddbc, source)
    plugin.columns.set_value('count "*')
    output = plugin.execute(ddbc, source)
    assert_rows(
        [(r[0], r[3], r[4]) for r in output.fetchall()],
        [("x", *two_point([10, 20], [100, 100])), ("y", *two_point([90, 80], [100, 100]))],
    )
