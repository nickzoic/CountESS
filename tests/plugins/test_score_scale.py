import statistics

import pytest

from countess.plugins.score_scale import ScaleClassParam, ScoreScalingPlugin

# variant class, score:  class "syn" sits at 3 (its median) and class "non" at -8
DATA = """select * from (values
    ('syn', 1.0), ('syn', 3.0), ('syn', 5.0), ('non', -9.0), ('non', -7.0), ('mis', 2.0)) t(v, score)"""


def scale(ddbc, source, zero, one, scaled=None, group=None):
    """`zero` and `one` are (aggregation, column, operation, value) for the two classifiers"""
    plugin = ScoreScalingPlugin()
    plugin.prepare(ddbc, source)
    plugin.score_col.value = "score"
    if scaled:
        plugin.scaled_col.value = scaled
    for classifier, (agg, col, op, value) in zip(plugin.classifiers, (zero, one)):
        classifier.agg.value = agg
        classifier.col.value = col
        classifier.op.value = op
        classifier.st.value = value
    if group:
        plugin.set_parameter("group_col.0", group)
    return plugin, plugin.execute(ddbc, source)


def test_scaled_scores(ddbc, make_source, assert_rows):
    _, output = scale(ddbc, make_source(DATA), ("Median", "v", "Equals", "syn"), ("Median", "v", "Equals", "non"))
    assert output.columns == ["v", "score", "scaled_score"]
    zero, one = 3.0, -8.0  # the medians of syn and non
    assert_rows(
        output,
        [
            (v, s, (s - zero) / (one - zero))
            for v, s in [("syn", 1.0), ("syn", 3.0), ("syn", 5.0), ("non", -9.0), ("non", -7.0), ("mis", 2.0)]
        ],
    )


def test_by_construction_the_classes_scale_to_zero_and_one(ddbc, make_source):
    _, output = scale(ddbc, make_source(DATA), ("Median", "v", "Equals", "syn"), ("Median", "v", "Equals", "non"))
    scaled = {}
    for v, _, s in output.fetchall():
        scaled.setdefault(v, []).append(s)
    assert statistics.median(scaled["syn"]) == pytest.approx(0.0)
    assert statistics.median(scaled["non"]) == pytest.approx(1.0)


def test_median_or_average(ddbc, make_source, assert_rows):
    source = make_source(
        "select * from (values ('a', 1.0), ('a', 2.0), ('a', 9.0), ('b', 10.0), ('b', 20.0), ('b', 60.0), ('c', 5.0)) t(v, score)"
    )
    # the median of a is 2 and its average is 4, of b it's 20 and 30
    _, median = scale(ddbc, source, ("Median", "v", "Equals", "a"), ("Median", "v", "Equals", "b"))
    assert_rows(
        median,
        [
            (v, s, (s - 2) / (20 - 2))
            for v, s in [("a", 1.0), ("a", 2.0), ("a", 9.0), ("b", 10.0), ("b", 20.0), ("b", 60.0), ("c", 5.0)]
        ],
    )
    _, average = scale(ddbc, source, ("Avg", "v", "Equals", "a"), ("Avg", "v", "Equals", "b"))
    assert_rows(
        average,
        [
            (v, s, (s - 4) / (30 - 4))
            for v, s in [("a", 1.0), ("a", 2.0), ("a", 9.0), ("b", 10.0), ("b", 20.0), ("b", 60.0), ("c", 5.0)]
        ],
    )


def test_classes_with_no_members_default_to_zero_and_one(ddbc, make_source, assert_rows):
    source = make_source("select * from (values ('x', 0.25), ('y', 0.75)) t(v, score)")
    _, output = scale(ddbc, source, ("Median", "v", "Equals", "nothing"), ("Median", "v", "Equals", "nothing either"))
    assert_rows(output, [("x", 0.25, 0.25), ("y", 0.75, 0.75)])  # (so the scores are unchanged)
    _, half = scale(ddbc, source, ("Median", "v", "Equals", "x"), ("Median", "v", "Equals", "nothing"))
    assert_rows(half, [("x", 0.25, 0.0), ("y", 0.75, 0.5 / 0.75)])  # score_0 = 0.25, score_1 = 1.0


def test_both_classes_in_the_same_place_give_nan_and_infinity_not_an_error(ddbc, make_source, assert_rows):
    source = make_source("select * from (values ('a', 1.0), ('b', 2.0)) t(v, score)")
    _, output = scale(ddbc, source, ("Median", "v", "Equals", "a"), ("Median", "v", "Equals", "a"))
    # (there's no scale to put them on: 0/0 and 1/0)
    assert_rows(output, [("a", 1.0, float("nan")), ("b", 2.0, float("inf"))])


def test_scaled_within_groups(ddbc, make_source, assert_rows):
    source = make_source("""select * from (values
            ('g1', 'syn', 1.0), ('g1', 'syn', 3.0), ('g1', 'non', 5.0), ('g1', 'mis', 4.0),
            ('g2', 'syn', 10.0), ('g2', 'non', 30.0), ('g2', 'mis', 20.0)) t(g, v, score)""")
    _, output = scale(ddbc, source, ("Avg", "v", "Equals", "syn"), ("Avg", "v", "Equals", "non"), group="g")
    # g1 scales 2 -> 0 and 5 -> 1, g2 scales 10 -> 0 and 30 -> 1
    assert_rows(
        output,
        [
            ("g1", "syn", 1.0, -1 / 3),
            ("g1", "syn", 3.0, 1 / 3),
            ("g1", "non", 5.0, 1.0),
            ("g1", "mis", 4.0, 2 / 3),
            ("g2", "syn", 10.0, 0.0),
            ("g2", "non", 30.0, 1.0),
            ("g2", "mis", 20.0, 0.5),
        ],
    )


@pytest.mark.xfail(
    strict=True,
    reason="the groups are joined back to the rows with USING, and NULL never equals NULL, so every row whose "
    "group value is NULL is silently dropped from the output",
)
def test_rows_with_no_group_value_are_kept(ddbc, make_source):
    source = make_source(
        "select * from (values ('g1', 'syn', 1.0), ('g1', 'non', 5.0), (NULL, 'syn', 1.0), (NULL, 'non', 5.0)) t(g, v, score)"
    )
    _, output = scale(ddbc, source, ("Avg", "v", "Equals", "syn"), ("Avg", "v", "Equals", "non"), group="g")
    assert len(output.fetchall()) == 4


def test_the_scaled_column_has_a_name_of_your_choosing(ddbc, make_source):
    _, output = scale(
        ddbc, make_source(DATA), ("Median", "v", "Equals", "syn"), ("Median", "v", "Equals", "non"), scaled="ünï col"
    )
    assert output.columns == ["v", "score", "ünï col"]


def test_scaling_into_an_existing_column_replaces_it(ddbc, make_source, assert_rows):
    source = make_source("select * from (values ('a', 2.0), ('b', 4.0)) t(v, score)")
    _, output = scale(ddbc, source, ("Avg", "v", "Equals", "a"), ("Avg", "v", "Equals", "b"), scaled="score")
    assert output.columns == ["v", "score"]
    assert_rows(output, [("a", 0.0), ("b", 1.0)])


# ---- what the classifiers pick out

WORDS = ["apple", "apricot", "banana", "grape", "pineapple"]


def members(ddbc, make_source, op, value):
    source = make_source("select * from (values " + ", ".join(f"('{w}')" for w in WORDS) + ") t(word)")
    classifier = ScaleClassParam("Class")
    classifier.col.value = "word"
    classifier.op.value = op
    classifier.st.value = value
    return sorted(w for (w,) in ddbc.sql(f"select word from {source.alias} where {classifier.filter()}").fetchall())


@pytest.mark.parametrize(
    "op,value,expected",
    [
        ("Equals", "grape", ["grape"]),
        ("Starts With", "ap", ["apple", "apricot"]),
        ("Ends With", "ple", ["apple", "pineapple"]),
        ("Contains", "ana", ["banana"]),
        ("Matches", "a.*e", ["apple"]),  # (the whole word must match)
        ("Matches", ".*ap.*", ["apple", "apricot", "grape", "pineapple"]),
        ("Greater Than", "b", ["banana", "grape", "pineapple"]),
        ("Less Than", "b", ["apple", "apricot"]),
    ],
)
def test_operations(ddbc, make_source, op, value, expected):
    assert members(ddbc, make_source, op, value) == sorted(expected)


def test_a_value_with_quotes_is_not_a_problem(ddbc, make_source):
    source = make_source("""select * from (values ('it''s'), ('its')) t(word)""")
    classifier = ScaleClassParam("Class")
    classifier.col.value = "word"
    classifier.op.value = "Equals"
    classifier.st.value = "it's"
    assert ddbc.sql(f"select word from {source.alias} where {classifier.filter()}").fetchall() == [("it's",)]


def test_numeric_comparisons(ddbc, make_source, assert_rows):
    source = make_source("select * from (values (1.0, 'lo'), (5.0, 'mid'), (9.0, 'hi')) t(score, name)")
    for op, value, expected in (("Greater Than", "4", {"mid", "hi"}), ("Less Than", "6", {"lo", "mid"})):
        classifier = ScaleClassParam("Class")
        classifier.col.value = "score"
        classifier.op.value = op
        classifier.st.value = value
        found = {n for (n,) in ddbc.sql(f"select name from {source.alias} where {classifier.filter()}").fetchall()}
        assert found == expected


def test_an_operation_it_doesnt_know(ddbc):
    classifier = ScaleClassParam("Class")
    classifier.op.value = "Resembles"
    with pytest.raises(NotImplementedError):
        classifier.filter()


def test_the_classifiers_are_labelled(ddbc, make_source):
    plugin = ScoreScalingPlugin()
    assert [c.label for c in plugin.classifiers] == ["Scale to 0.0", "Scale to 1.0"]
    assert len(plugin.classifiers) == 2
