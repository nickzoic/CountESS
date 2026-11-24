import math

import duckdb
import numpy as np
import pytest
from scipy.optimize import minimize_scalar

from countess.plugins.random_effects import (
    RandomEffectsPlugin,
    RandomEffectsPlugin2,
    rml_estimate,
    rml_estimate_wrapper,
)


def reml(scores, sigmas):
    """The restricted maximum likelihood random effects estimate, found by directly maximizing the
    likelihood (rather than by the iteration which is being tested): returns (estimate, sigma, heterogeneity)."""
    y, v = np.array(scores), np.array(sigmas) ** 2

    def negative_log_likelihood(tau2):
        w = 1 / (v + tau2)
        mu = (w * y).sum() / w.sum()
        return 0.5 * (np.log(v + tau2).sum() + np.log(w.sum()) + (w * (y - mu) ** 2).sum())

    best = minimize_scalar(negative_log_likelihood, bounds=(0.0, 100.0), method="bounded", options={"xatol": 1e-13})
    tau2 = best.x if negative_log_likelihood(best.x) < negative_log_likelihood(0.0) else 0.0
    w = 1 / (v + tau2)
    return (w * y).sum() / w.sum(), 1 / math.sqrt(w.sum()), tau2


# ---- the estimate


def test_one_score_is_its_own_estimate():
    assert rml_estimate([0.7], [0.1]) == (0.7, 0.1)


def test_scores_which_agree_give_that_score_with_a_smaller_sigma():
    estimate, sigma = rml_estimate([2.0, 2.0, 2.0, 2.0], [0.4, 0.4, 0.4, 0.4])
    assert estimate == pytest.approx(2.0)
    assert sigma == pytest.approx(0.4 / math.sqrt(4))  # four equal measurements: sigma / sqrt(n)


@pytest.mark.parametrize(
    "scores,sigmas",
    [
        ([0.1, 0.5, 0.9, 0.3], [0.1, 0.1, 0.1, 0.1]),
        ([1.0, 2.0, 4.0, 3.0, 2.5], [0.2, 0.5, 0.3, 0.4, 0.6]),
        ([-1.0, 1.0], [0.1, 0.1]),
        ([0.0, 10.0, 5.0], [0.5, 2.0, 1.0]),
    ],
)
def test_matches_the_maximum_likelihood_estimate(scores, sigmas):
    estimate, sigma = rml_estimate(scores, sigmas)
    expected_estimate, expected_sigma, heterogeneity = reml(scores, sigmas)
    assert heterogeneity > 0.05  # (these are all sets of scores which really do disagree)
    assert estimate == pytest.approx(expected_estimate, rel=1e-4)
    assert sigma == pytest.approx(expected_sigma, rel=1e-4)


def test_matches_the_maximum_likelihood_estimate_for_random_scores():
    rng = np.random.default_rng(2024)
    compared = 0
    for _ in range(200):
        n = int(rng.integers(2, 9))
        sigmas = list(rng.uniform(0.05, 0.5, n))
        scores = list(rng.normal(0, rng.uniform(0.05, 2.0), n))
        expected_estimate, expected_sigma, heterogeneity = reml(scores, sigmas)
        if heterogeneity < 0.05:
            continue  # (when there's hardly any heterogeneity the iteration only gets close, see below)
        estimate, sigma = rml_estimate(scores, sigmas)
        assert estimate == pytest.approx(expected_estimate, rel=1e-3, abs=1e-6), (scores, sigmas)
        assert sigma == pytest.approx(expected_sigma, rel=1e-3), (scores, sigmas)
        compared += 1
    assert compared > 100


def test_scores_which_barely_differ_are_the_inverse_variance_weighted_mean():
    scores, sigmas = [1.0, 1.001, 0.999, 1.0], [0.5, 0.25, 0.5, 0.25]
    estimate, sigma = rml_estimate(scores, sigmas)
    weights = [1 / s**2 for s in sigmas]
    assert estimate == pytest.approx(sum(w * y for w, y in zip(weights, scores)) / sum(weights), rel=1e-4)
    assert sigma == pytest.approx(1 / math.sqrt(sum(weights)), rel=1e-2)


def test_more_precise_scores_count_for_more():
    scores, sigmas = [0.0, 10.0], [0.01, 5.0]
    estimate, _ = rml_estimate(scores, sigmas)
    # (it's pulled towards the precise score, though not all the way: when the scores disagree this much the
    # differences between them swamp how precise each is, so the weights are much more even than 1/sigma^2)
    assert estimate < sum(scores) / 2
    assert estimate == pytest.approx(reml(scores, sigmas)[0], rel=1e-4)
    equal_sigmas, _ = rml_estimate(scores, [1.0, 1.0])
    assert equal_sigmas == pytest.approx(5.0)


def test_order_makes_no_difference():
    scores, sigmas = [0.3, 1.9, 0.8, 1.1], [0.2, 0.3, 0.25, 0.4]
    pairs = list(zip(scores, sigmas))
    expected = rml_estimate(scores, sigmas)
    for order in ([3, 2, 1, 0], [1, 3, 0, 2]):
        shuffled = [pairs[i] for i in order]
        got = rml_estimate([p[0] for p in shuffled], [p[1] for p in shuffled])
        assert got == pytest.approx(expected)


def test_shifting_the_scores_shifts_the_estimate_but_not_its_sigma():
    scores, sigmas = [0.3, 1.9, 0.8, 1.1], [0.2, 0.3, 0.25, 0.4]
    estimate, sigma = rml_estimate(scores, sigmas)
    shifted_estimate, shifted_sigma = rml_estimate([s + 100 for s in scores], sigmas)
    assert shifted_estimate == pytest.approx(estimate + 100)
    assert shifted_sigma == pytest.approx(sigma)


def test_disagreement_makes_the_estimate_less_certain():
    _, certain = rml_estimate([1.0, 1.0, 1.0], [0.1, 0.1, 0.1])
    _, uncertain = rml_estimate([0.0, 1.0, 2.0], [0.1, 0.1, 0.1])
    assert uncertain > certain


def test_bad_input_is_refused():
    with pytest.raises(AssertionError, match="Mismatched"):
        rml_estimate([1.0, 2.0], [0.1])
    with pytest.raises(AssertionError, match="sigmas must be > 0"):
        rml_estimate([1.0, 2.0], [0.1, 0.0])
    with pytest.raises(AssertionError, match="sigmas must be > 0"):
        rml_estimate([1.0, 2.0], [0.1, -0.5])


def test_the_wrapper_gives_nulls_for_bad_input_instead_of_raising():
    assert rml_estimate_wrapper([1.0, 2.0], [0.1, 0.0]) == (None, None)
    assert rml_estimate_wrapper([1.0, 2.0], [0.1]) == (None, None)
    assert rml_estimate_wrapper([1.0, 3.0], [0.1, 0.1]) == pytest.approx(rml_estimate([1.0, 3.0], [0.1, 0.1]))


@pytest.mark.parametrize(
    "scores,sigmas",
    [
        ([1.0, None], [0.1, 0.2]),
        ([None, None], [0.1, 0.2]),
        ([1.0, 2.0], [0.1, None]),
        ([None], [None]),
        ([], []),
        ([1.0, 2.0], []),
        ([], [0.1, 0.2]),
        (None, None),  # (what LIST() of no rows at all is)
        ([1.0, 2.0], [0.1, -0.2]),
        ([1.0, 2.0], [0.1, 1e-200]),  # (so small that its square is 0, which divides by 0)
    ],
    ids=[
        "null score",
        "null scores",
        "null sigma",
        "only nulls",
        "no scores",
        "no sigmas",
        "no scores for sigmas",
        "no lists",
        "negative sigma",
        "tiny sigma",
    ],
)
def test_what_cant_be_combined_is_nulls_and_not_an_error(scores, sigmas):
    """(A NULL score can't be combined with the others, so there is no score; it's not for it to be left out, or
    to stop whatever else is being done as well.)"""
    assert rml_estimate_wrapper(scores, sigmas) == (None, None)


def test_the_wrapper_takes_decimals_as_the_floats_they_are_worth():
    from decimal import Decimal  # pylint: disable=import-outside-toplevel

    decimals = rml_estimate_wrapper([Decimal("0.5"), Decimal("1.5")], [Decimal("0.1"), Decimal("0.2")])
    assert decimals == pytest.approx(rml_estimate([0.5, 1.5], [0.1, 0.2]))
    assert all(isinstance(v, float) for v in decimals)
    assert rml_estimate_wrapper([Decimal("1"), 2.0, 3], [Decimal("0.1"), 0.2, 1]) == pytest.approx(
        rml_estimate([1.0, 2.0, 3.0], [0.1, 0.2, 1.0])
    )


# ---- combining the scores in each row


def row_plugin(ddbc, source, drop=False):
    plugin = RandomEffectsPlugin()
    plugin.prepare(ddbc, source)
    plugin.score_cols.set_value("score_*")
    plugin.sigma_cols.set_value("sigma_*")
    plugin.drop_cols.value = drop
    return plugin


# (a literal like 0.1 is a DECIMAL as far as DuckDB is concerned, which isn't what's being tested here)
DOUBLES = """select id, score_1::double as score_1, score_2::double as score_2, sigma_1::double as sigma_1,
    sigma_2::double as sigma_2 from (values {}) t(id, score_1, score_2, sigma_1, sigma_2)"""

ROW = {"id": "v", "score_1": 0.5, "score_2": 1.5, "score_3": 1.0, "sigma_1": 0.1, "sigma_2": 0.2, "sigma_3": 0.15}


@pytest.fixture(name="row_source")
def fixture_row_source(make_source):
    return make_source(
        "select 'v' as id, 0.5::double as score_1, 1.5::double as score_2, 1.0::double as score_3, "
        "0.1::double as sigma_1, 0.2::double as sigma_2, 0.15::double as sigma_3"
    )


def test_a_row_is_given_a_combined_score(ddbc, row_source):
    plugin = row_plugin(ddbc, row_source)
    output = plugin.transform(dict(ROW))
    expected = rml_estimate([0.5, 1.5, 1.0], [0.1, 0.2, 0.15])
    assert (output["score"], output["sigma"]) == pytest.approx(expected)
    assert {k: v for k, v in output.items() if k not in ("score", "sigma")} == ROW  # (everything else is kept)


def test_scores_are_paired_with_sigmas_by_name(ddbc, row_source):
    plugin = row_plugin(ddbc, row_source)
    # ... so what order the columns are in doesn't matter
    jumbled = dict(reversed(list(ROW.items())))
    assert plugin.transform(jumbled)["score"] == pytest.approx(plugin.transform(dict(ROW))["score"])


def test_inputs_can_be_dropped(ddbc, row_source):
    plugin = row_plugin(ddbc, row_source, drop=True)
    output = plugin.transform(dict(ROW))
    assert sorted(output) == ["id", "score", "sigma"]
    assert plugin.remove_fields() == plugin.score_cols.get_matching_columns() + plugin.sigma_cols.get_matching_columns()
    assert row_plugin(ddbc, row_source, drop=False).remove_fields() == []


@pytest.mark.parametrize(
    "change",
    [
        {"score_2": None},
        {"sigma_2": None},
        {"sigma_2": 0},
        {"sigma_2": -0.2},
        {"score_1": None, "score_2": None, "score_3": None},
    ],
    ids=["missing score", "missing sigma", "zero sigma", "negative sigma", "no scores"],
)
def test_a_row_which_cant_be_combined_is_kept_with_nulls_for_its_score(ddbc, row_source, change):
    plugin = row_plugin(ddbc, row_source)
    row = {**ROW, **change}
    output = plugin.transform(dict(row))
    assert output == {**row, "score": None, "sigma": None}  # (so its id, and all of the rest of it, are still there)


def test_a_row_which_cant_be_combined_has_its_inputs_dropped_too_if_it_is_to_be(ddbc, row_source):
    plugin = row_plugin(ddbc, row_source, drop=True)
    assert plugin.transform({**ROW, "score_2": None}) == {"id": "v", "score": None, "sigma": None}


def test_without_any_scores_there_is_nothing_to_combine_but_the_row_is_kept(ddbc, row_source):
    plugin = row_plugin(ddbc, row_source)
    assert plugin.transform({"id": "v"}) == {"id": "v", "score": None, "sigma": None}


def test_row_plugin_in_a_query(ddbc, make_source):
    source = make_source(DOUBLES.format("('a', 0.5, 1.5, 0.1, 0.2), ('b', 2.0, 2.0, 0.3, 0.3)"))
    plugin = row_plugin(ddbc, source)
    output = plugin.execute(ddbc, source)
    assert output.columns == ["id", "score_1", "score_2", "sigma_1", "sigma_2", "score", "sigma"]
    rows = {r[0]: r for r in output.fetchall()}
    assert (rows["a"][5], rows["a"][6]) == pytest.approx(rml_estimate([0.5, 1.5], [0.1, 0.2]))
    assert (rows["b"][5], rows["b"][6]) == pytest.approx((2.0, 0.3 / math.sqrt(2)))
    assert [str(t) for t in output.dtypes][-2:] == ["DOUBLE", "DOUBLE"]


def test_row_plugin_drops_inputs_in_a_query(ddbc, make_source):
    source = make_source(DOUBLES.format("('a', 0.5, 1.5, 0.1, 0.2)"))
    output = row_plugin(ddbc, source, drop=True).execute(ddbc, source)
    assert output.columns == ["id", "score", "sigma"]


def test_a_row_which_cant_be_combined_is_still_a_row_in_a_query(ddbc, make_source):
    """A NULL score gives a NULL score: the row still has its id and everything else, and the other rows are as
    they were.  (It used to be a row of nothing but NULLs, the id as well, as DuckdbTransformPlugin.transform says
    that None is.)"""
    source = make_source(
        DOUBLES.format("('a', 0.5, 1.5, 0.1, 0.2), ('b', 2.0, NULL, 0.1, 0.3), ('c', 2.0, 2.0, 0.0, 0.3)")
    )
    rows = {r[0]: r for r in row_plugin(ddbc, source).execute(ddbc, source).fetchall()}
    assert sorted(rows) == ["a", "b", "c"]  # (all three, none of them lost)
    assert (rows["a"][5], rows["a"][6]) == pytest.approx(rml_estimate([0.5, 1.5], [0.1, 0.2]))
    assert rows["b"] == ("b", 2.0, None, 0.1, 0.3, None, None)
    assert rows["c"] == ("c", 2.0, 2.0, 0.0, 0.3, None, None)


def test_a_missing_score_doesnt_stop_the_rows_after_it(ddbc, make_source):
    values = ", ".join(f"('r{n}', {n}.5, {'NULL' if n % 3 == 0 else f'{n + 1}.5'}, 0.1, 0.2)" for n in range(30))
    source = make_source(DOUBLES.format(values))
    rows = row_plugin(ddbc, source).execute(ddbc, source).fetchall()
    assert len(rows) == 30
    assert sum(1 for r in rows if r[5] is None) == 10 and sum(1 for r in rows if r[5] is not None) == 20


def test_decimal_columns_with_missing_scores_in_a_query(ddbc, make_source):
    source = make_source(
        "select * from (values ('a', 0.5, 1.5, 0.1, 0.2), ('b', 0.5, NULL, 0.1, 0.2)) t(id, score_1, score_2, sigma_1, sigma_2)"
    )
    assert str(source.dtypes[1]).startswith("DECIMAL")
    rows = {r[0]: r for r in row_plugin(ddbc, source).execute(ddbc, source).fetchall()}
    assert rows["a"][5:] == pytest.approx(rml_estimate([0.5, 1.5], [0.1, 0.2]))
    assert rows["b"][0] == "b" and rows["b"][5:] == (None, None)


# ---- combining the scores in groups

GROUPED = """select * from (values
    ('a', 0.5, 0.1), ('a', 1.5, 0.2), ('a', 1.0, 0.15), ('b', 3.0, 0.1), ('b', 4.0, 0.3), ('c', 7.0, 0.5))
    t(grp, score, sigma)"""


def group_plugin(ddbc, source, group=True, score="score", sigma="sigma"):
    plugin = RandomEffectsPlugin2()
    plugin.prepare(ddbc, source)
    if score:
        plugin.score_col.value = score
    if sigma:
        plugin.sigma_col.value = sigma
    if group:
        plugin.set_parameter("group_cols.0", "grp")
    return plugin


def test_scores_combined_by_group(ddbc, make_source, assert_rows):
    source = make_source(GROUPED)
    output = group_plugin(ddbc, source).execute(ddbc, source)
    assert output.columns == ["grp", "score", "sigma", "count"]
    assert_rows(
        output,
        [
            ("a", *rml_estimate([0.5, 1.5, 1.0], [0.1, 0.2, 0.15]), 3),
            ("b", *rml_estimate([3.0, 4.0], [0.1, 0.3]), 2),
            ("c", 7.0, 0.5, 1),  # (a group of one keeps its score and sigma)
        ],
    )


def test_all_the_scores_combined(ddbc, make_source, assert_rows):
    source = make_source(GROUPED)
    output = group_plugin(ddbc, source, group=False).execute(ddbc, source)
    assert output.columns == ["score", "sigma", "count"]
    scores = [0.5, 1.5, 1.0, 3.0, 4.0, 7.0]
    sigmas = [0.1, 0.2, 0.15, 0.1, 0.3, 0.5]
    assert_rows(output, [(*rml_estimate(scores, sigmas), 6)])


def test_a_group_with_a_sigma_of_zero_gives_nulls_but_still_counts(ddbc, make_source, assert_rows):
    source = make_source(
        "select * from (values ('a', 1.0, 0.1), ('a', 2.0, 0.2), ('b', 1.0, 0.0), ('b', 2.0, 0.1)) t(grp, score, sigma)"
    )
    output = group_plugin(ddbc, source).execute(ddbc, source)
    assert_rows(output, [("a", *rml_estimate([1.0, 2.0], [0.1, 0.2]), 2), ("b", None, None, 2)])


def test_nothing_chosen_passes_the_table_through(ddbc, make_source):
    source = make_source(GROUPED)
    plugin = group_plugin(ddbc, source, score=None, sigma=None)
    plugin.score_col.value = ""  # (once prepared the first column is chosen, so this has to be done on purpose)
    plugin.sigma_col.value = ""
    output = plugin.execute(ddbc, source)
    assert output.columns == ["grp", "score", "sigma"]
    assert len(output.fetchall()) == 6


def test_the_function_is_made_once_per_plugin(ddbc, make_source):
    source = make_source(GROUPED)
    plugin = group_plugin(ddbc, source)
    name = plugin.function_name
    plugin.prepare(ddbc, source)
    assert plugin.function_name == name
    assert group_plugin(ddbc, source).function_name != name  # (another plugin has a function of its own)


def test_odd_column_names(ddbc, make_source, assert_rows):
    source = make_source(
        """select * from (values ('x', 1.0, 0.1), ('x', 2.0, 0.2)) t("gr oup", "sc ""ore", "sig ma")"""
    )
    plugin = RandomEffectsPlugin2()
    plugin.prepare(ddbc, source)
    plugin.score_col.value = 'sc "ore'
    plugin.sigma_col.value = "sig ma"
    plugin.set_parameter("group_cols.0", "gr oup")
    assert_rows(plugin.execute(ddbc, source), [("x", *rml_estimate([1.0, 2.0], [0.1, 0.2]), 2)])


def test_a_group_with_a_missing_score_doesnt_stop_the_others(ddbc, make_source, assert_rows):
    source = make_source(
        "select * from (values ('a', 1.0, 0.1), ('a', 2.0, 0.2), ('b', 3.0, 0.1), ('b', NULL, 0.1)) t(grp, score, sigma)"
    )
    output = group_plugin(ddbc, source).execute(ddbc, source)
    rows = {r[0]: r for r in output.fetchall()}
    assert rows["a"][1:3] == pytest.approx(rml_estimate([1.0, 2.0], [0.1, 0.2]))
    # (the group with the NULL is there, with NULLs for its score and sigma, and it still counts its rows)
    assert rows["b"] == ("b", None, None, 2)


def test_a_group_with_a_missing_sigma_is_the_same(ddbc, make_source, assert_rows):
    source = make_source(
        "select * from (values ('a', 1.0, 0.1), ('a', 2.0, NULL), ('b', 3.0, 0.1)) t(grp, score, sigma)"
    )
    assert_rows(group_plugin(ddbc, source).execute(ddbc, source), [("a", None, None, 2), ("b", 3.0, 0.1, 1)])


def test_all_the_scores_with_a_missing_one_are_a_row_of_nulls_which_counts(ddbc, make_source, assert_rows):
    source = make_source(
        "select * from (values ('a', 1.0, 0.1), ('b', NULL, 0.2), ('c', 3.0, 0.3)) t(grp, score, sigma)"
    )
    assert_rows(group_plugin(ddbc, source, group=False).execute(ddbc, source), [(None, None, 3)])


def test_decimal_columns_are_combined_in_groups_too(ddbc, make_source, assert_rows):
    source = make_source(
        "select * from (values ('a', 1.0, 0.1), ('a', 2.0, 0.2), ('b', 3.0, NULL)) t(grp, score, sigma)"
    )
    assert str(source.dtypes[1]).startswith("DECIMAL")
    assert_rows(
        group_plugin(ddbc, source).execute(ddbc, source),
        [("a", *rml_estimate([1.0, 2.0], [0.1, 0.2]), 2), ("b", None, None, 1)],
    )


def test_decimal_columns_can_be_combined_too(ddbc, make_source):
    source = make_source("select 'a' as id, 0.5 as score_1, 1.5 as score_2, 0.1 as sigma_1, 0.2 as sigma_2")
    assert str(source.dtypes[1]).startswith("DECIMAL")
    output = row_plugin(ddbc, source).execute(ddbc, source)
    assert output.fetchall()[0][5] == pytest.approx(rml_estimate([0.5, 1.5], [0.1, 0.2])[0])
