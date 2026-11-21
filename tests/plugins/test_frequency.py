import math

import pytest

from countess.plugins.frequency import FrequencyPlugin


def frequencies(ddbc, source, **params):
    plugin = FrequencyPlugin()
    plugin.prepare(ddbc, source)
    for key, value in params.items():
        plugin.set_parameter(key, value)
    return plugin, plugin.execute(ddbc, source)


DATA = "select * from (values ('a', 1, 10), ('a', 3, 30), ('b', 0, 0), ('b', 0, 0)) t(g, count_1, count_2)"


def test_the_columns_called_count_are_chosen_without_being_asked(ddbc, make_source):
    plugin, _ = frequencies(ddbc, make_source(DATA))
    assert plugin.columns.get_values() == {"count_1", "count_2"}
    assert plugin.group_cols.choices == ["g"]  # (a count column can't also be a group)


def test_frequencies_of_the_whole_table(ddbc, make_source, assert_rows):
    source = make_source(DATA)
    _, output = frequencies(ddbc, source)
    assert output.columns == ["g", "count_1", "count_2", "freq_1", "freq_2"]  # (count_N -> freq_N)
    assert_rows(
        output,
        [
            ("a", 1, 10, 1 / 4, 10 / 40),
            ("a", 3, 30, 3 / 4, 30 / 40),
            ("b", 0, 0, 0.0, 0.0),
            ("b", 0, 0, 0.0, 0.0),
        ],
    )


def test_frequencies_within_groups(ddbc, make_source, assert_rows):
    source = make_source(
        "select * from (values ('a', 1, 10), ('a', 3, 30), ('b', 2, 5), ('b', 2, 15)) t(g, count_1, count_2)"
    )
    _, output = frequencies(ddbc, source, **{"group_cols.0": "g"})
    assert_rows(
        output,
        [
            ("a", 1, 10, 1 / 4, 10 / 40),
            ("a", 3, 30, 3 / 4, 30 / 40),
            ("b", 2, 5, 1 / 2, 5 / 20),
            ("b", 2, 15, 1 / 2, 15 / 20),
        ],
    )


def test_a_group_with_no_counts_has_zero_frequencies_not_errors(ddbc, make_source, assert_rows):
    source = make_source("select * from (values ('a', 4), ('a', 4), ('b', 0)) t(g, count)")
    _, output = frequencies(ddbc, source, **{"group_cols.0": "g"})
    assert output.columns == ["g", "count", "freq_count"]  # (count -> freq_count)
    assert_rows(output, [("a", 4, 0.5), ("a", 4, 0.5), ("b", 0, 0.0)])


def test_sigmas(ddbc, make_source, assert_rows):
    source = make_source("select * from (values (1, 3), (3, 5)) t(count_a, count_b)")
    _, output = frequencies(ddbc, source, sigma=True)
    assert output.columns == ["count_a", "count_b", "freq_a", "freq_b", "sigma_a", "sigma_b"]
    assert_rows(
        output,
        [
            (1, 3, 1 / 4, 3 / 8, math.sqrt(1.5) / 4, math.sqrt(3.5) / 8),
            (3, 5, 3 / 4, 5 / 8, math.sqrt(3.5) / 4, math.sqrt(5.5) / 8),
        ],
    )


@pytest.mark.xfail(
    strict=True,
    reason="the groups are joined back to the rows with USING, and NULL never equals NULL, so every row whose "
    "group value is NULL is silently dropped from the output",
)
def test_rows_with_no_group_value_are_kept(ddbc, make_source, assert_rows):
    source = make_source("select * from (values ('a', 1), ('a', 3), (NULL, 2), (NULL, 2)) t(g, count)")
    _, output = frequencies(ddbc, source, **{"group_cols.0": "g"})
    assert_rows(output, [("a", 1, 0.25), ("a", 3, 0.75), (None, 2, 0.5), (None, 2, 0.5)])


def test_counts_can_be_chosen(ddbc, make_source, assert_rows):
    source = make_source("select * from (values (1, 2), (3, 6)) t(reads, other)")
    _, output = frequencies(ddbc, source, **{"columns.0": "reads"})
    assert output.columns == ["reads", "other", "freq_reads"]
    assert_rows(output, [(1, 2, 0.25), (3, 6, 0.75)])


def test_missing_counts(ddbc, make_source, assert_rows):
    source = make_source("select * from (values (1), (NULL), (3)) t(count)")
    _, output = frequencies(ddbc, source)
    assert_rows(output, [(1, 0.25), (None, None), (3, 0.75)])  # (the NULL doesn't count towards the total)


def test_without_any_counts_there_is_nothing_to_do(ddbc, make_source):
    _, output = frequencies(ddbc, make_source("select 1 as a, 2 as b"))
    assert output is None


def test_odd_column_names(ddbc, make_source, assert_rows):
    # (columns named  gr oup  and  count "1 )
    source = make_source('select * from (values (\'x\', 1), (\'x\', 3)) t("gr oup", "count ""1")')
    _, output = frequencies(ddbc, source, **{"columns.0": 'count "1', "group_cols.0": "gr oup"})
    assert_rows(output, [("x", 1, 0.25), ("x", 3, 0.75)])


def test_chosen_columns_are_kept_when_prepared_again(ddbc, make_source):
    source = make_source("select * from (values (1, 2, 3)) t(count_1, reads, other)")
    plugin, _ = frequencies(ddbc, source)
    assert plugin.columns.get_values() == {"count_1"}  # (the default)
    plugin.set_parameter("columns.0", "reads")
    plugin.set_parameter("columns.1", "other")
    plugin.prepare(ddbc, source)
    assert plugin.columns.get_values() >= {"reads", "other"}  # (not put back as it was)
