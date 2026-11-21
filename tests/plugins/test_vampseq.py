import pytest

from countess.plugins.vampseq import VampSeqScorePlugin

COUNT_COLUMNS = ["count_1", "count_2", "count_3"]


def expected_scores(rows, weights, threshold=1.78e-5, group=None):
    """The VAMP-seq score worked out in plain Python: for each variant, the average of the bin weights
    weighted by the fraction of each bin's reads which are this variant (so that bins which have more reads
    in total don't count for more), and the variant's share of all the reads for the group it's in."""
    results = []
    for name, row_group, counts in rows:
        members = [c for n, g, c in rows if g == row_group] if group else [c for _, _, c in rows]
        totals = [sum(m[i] for m in members) for i in range(len(counts))]
        weighted = sum(c * w / t for c, w, t in zip(counts, weights, totals) if t > 0)
        unweighted = sum(c / t for c, t in zip(counts, totals) if t > 0)
        total_freq = sum(counts) / sum(totals)
        if total_freq > threshold:
            results.append((name, row_group, *counts, weighted / unweighted if unweighted else None, total_freq))
    return results


ROWS = [
    ("r1", "g", (10, 0, 0)),
    ("r2", "g", (0, 20, 0)),
    ("r3", "g", (0, 0, 30)),
    ("r4", "g", (10, 20, 30)),
]


def table(make_source, rows):
    values = ", ".join(f"('{n}', '{g}', {c[0]}, {c[1]}, {c[2]})" for n, g, c in rows)
    return make_source(f"select * from (values {values}) t(name, grp, {', '.join(COUNT_COLUMNS)})")


def score(ddbc, source, weights=None, threshold=None, group=None):
    plugin = VampSeqScorePlugin()
    plugin.prepare(ddbc, source)
    if weights:
        for param in plugin.columns.params:
            param.weight.value = weights.get(param.label)
    if threshold is not None:
        plugin.threshold.value = threshold
    if group:
        plugin.set_parameter("group_by.0", group)
    return plugin, plugin.execute(ddbc, source)


def test_the_count_columns_get_evenly_spaced_weights_without_being_asked(ddbc, make_source):
    plugin, _ = score(ddbc, table(make_source, ROWS))
    assert {p.label: p.weight.value for p in plugin.columns.params if p.weight.value is not None} == {
        "count_1": pytest.approx(1 / 3),
        "count_2": pytest.approx(2 / 3),
        "count_3": pytest.approx(1.0),
    }


def test_scores(ddbc, make_source, assert_rows):
    _, output = score(ddbc, table(make_source, ROWS))
    assert output.columns == ["name", "grp", *COUNT_COLUMNS, "score", "total_freq"]
    rows = output.fetchall()
    assert_rows(rows, expected_scores(ROWS, [1 / 3, 2 / 3, 1.0]))
    # ... which for reads only in the first, second or third bin are the weights of those bins, and a variant
    # with reads in all three bins is in between
    scores = {row[0]: row[5] for row in rows}
    assert scores == {
        "r1": pytest.approx(1 / 3),
        "r2": pytest.approx(2 / 3),
        "r3": pytest.approx(1.0),
        "r4": pytest.approx(2 / 3),
    }


def test_total_frequency_is_the_variants_share_of_all_the_reads(ddbc, make_source):
    _, output = score(ddbc, table(make_source, ROWS))
    freq = {row[0]: row[6] for row in output.fetchall()}
    assert freq == {
        "r1": pytest.approx(10 / 120),
        "r2": pytest.approx(20 / 120),
        "r3": pytest.approx(30 / 120),
        "r4": pytest.approx(60 / 120),
    }


def test_the_threshold_drops_rare_variants(ddbc, make_source, assert_rows):
    _, output = score(ddbc, table(make_source, ROWS), threshold=0.2)  # r1 and r2 are 0.083 and 0.167
    assert sorted(row[0] for row in output.fetchall()) == ["r3", "r4"]
    _, none = score(ddbc, table(make_source, ROWS), threshold=0.6)
    assert none.fetchall() == []


def test_the_threshold_is_exclusive(ddbc, make_source):
    _, output = score(ddbc, table(make_source, [("r1", "g", (1, 0, 0)), ("r2", "g", (3, 0, 0))]), threshold=0.25)
    assert [row[0] for row in output.fetchall()] == ["r2"]  # r1 is exactly 0.25, which is not > 0.25


def test_weights_can_be_chosen(ddbc, make_source, assert_rows):
    weights = {"count_1": 0.0, "count_2": 5.0, "count_3": 10.0}
    _, output = score(ddbc, table(make_source, ROWS), weights=weights)
    assert_rows(output, expected_scores(ROWS, [0.0, 5.0, 10.0]))


def test_only_columns_with_weights_are_used(ddbc, make_source, assert_rows):
    _, output = score(ddbc, table(make_source, ROWS), weights={"count_1": 1.0, "count_3": 3.0})
    expected = []
    for name, grp, counts in ROWS:
        c1, c3 = counts[0], counts[2]
        t1, t3 = 20, 60  # (the totals of the bins which count)
        total_freq = (c1 + c3) / (t1 + t3)
        if total_freq == 0:
            continue  # r2 has reads in the bin which doesn't count, so for our purposes it has none at all
        weighted = c1 * 1.0 / t1 + c3 * 3.0 / t3
        unweighted = c1 / t1 + c3 / t3
        expected.append((name, grp, *counts, weighted / unweighted, total_freq))
    assert [e[0] for e in expected] == ["r1", "r3", "r4"]
    assert_rows(output, expected)


def test_scores_within_groups(ddbc, make_source, assert_rows):
    rows = [
        ("a1", "x", (10, 0, 0)),
        ("a2", "x", (10, 30, 0)),
        ("b1", "y", (5, 5, 5)),
        ("b2", "y", (0, 5, 15)),
    ]
    _, output = score(ddbc, table(make_source, rows), group="grp")
    assert_rows(output, expected_scores(rows, [1 / 3, 2 / 3, 1.0], group=True))


def test_a_bin_with_no_reads_in_it_is_ignored(ddbc, make_source, assert_rows):
    rows = [("r1", "g", (10, 0, 0)), ("r2", "g", (30, 0, 0))]  # nothing in the second and third bins
    _, output = score(ddbc, table(make_source, rows))
    assert_rows(output, expected_scores(rows, [1 / 3, 2 / 3, 1.0]))
    assert all(row[5] == pytest.approx(1 / 3) for row in output.fetchall())


def test_nothing_to_weigh_gives_nothing(ddbc, make_source):
    source = make_source("select 'x' as name, 1 as reads")  # (no column called count*)
    _, output = score(ddbc, source)
    assert output is None


def test_group_choices_leave_out_the_weighted_columns(ddbc, make_source):
    plugin, _ = score(ddbc, table(make_source, ROWS))
    assert plugin.group_by.choices == ["name", "grp"]


@pytest.mark.xfail(
    strict=True,
    reason="the groups are joined back to the rows with an equality test, and NULL never equals NULL, so every row "
    "whose group value is NULL is silently dropped from the output",
)
def test_rows_with_no_group_value_are_kept(ddbc, make_source):
    source = make_source(
        "select * from (values ('a', 'x', 10, 1, 1), ('b', NULL, 10, 1, 1), ('c', NULL, 5, 1, 1)) t(name, grp, count_1, count_2, count_3)"
    )
    _, output = score(ddbc, source, group="grp")
    assert len(output.fetchall()) == 3


def test_weights_which_have_been_set_are_kept_when_prepared_again(ddbc, make_source):
    source = table(make_source, ROWS)
    plugin, _ = score(ddbc, source, weights={"count_1": 5.0, "count_2": 6.0, "count_3": 7.0})
    plugin.prepare(ddbc, source)
    assert {p.label: p.weight.value for p in plugin.columns.params if p.weight.value is not None} == {
        "count_1": 5.0,
        "count_2": 6.0,
        "count_3": 7.0,
    }


def test_preparing_with_no_table_is_harmless(ddbc):
    plugin = VampSeqScorePlugin()
    plugin.prepare(ddbc, None)
    assert plugin.columns.params == []
