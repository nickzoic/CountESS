from countess.plugins.ranking import RankingPlugin

DATA = "select * from (values (10, 'a'), (20, 'a'), (30, 'b'), (40, 'b'), (NULL, 'b')) t(x, g)"


def rank(ddbc, source, **params):
    plugin = RankingPlugin()
    plugin.prepare(ddbc, source)
    for key, value in params.items():
        plugin.set_parameter(key, value)
    return plugin.execute(ddbc, source)


def test_rank_by_one_column(ddbc, make_source, assert_rows):
    output = rank(ddbc, make_source(DATA), **{"order_by.0": "x"})
    assert output.columns == ["x", "g", "rank"]  # the new column goes on the end
    # percent_rank is (position - 1) / (rows - 1); a NULL sorts last, so it ranks highest
    assert_rows(output, [(10, "a", 0.0), (20, "a", 0.25), (30, "b", 0.5), (40, "b", 0.75), (None, "b", 1.0)])


def test_rank_within_partitions(ddbc, make_source, assert_rows):
    output = rank(ddbc, make_source(DATA), **{"order_by.0": "x", "partition.0": "g"})
    assert_rows(output, [(10, "a", 0.0), (20, "a", 1.0), (30, "b", 0.0), (40, "b", 0.5), (None, "b", 1.0)])


def test_rank_by_several_columns_breaks_ties(ddbc, make_source, assert_rows):
    source = make_source("select * from (values (1, 5), (1, 3), (2, 0)) t(a, b)")
    output = rank(ddbc, source, **{"order_by.0": "a", "order_by.1": "b"})
    assert_rows(output, [(1, 3, 0.0), (1, 5, 0.5), (2, 0, 1.0)])


def test_tied_values_share_a_rank(ddbc, make_source, assert_rows):
    output = rank(ddbc, make_source("select * from (values (1), (1), (2), (3)) t(x)"), **{"order_by.0": "x"})
    assert_rows(output, [(1, 0.0), (1, 0.0), (2, 2 / 3), (3, 1.0)])


def test_without_anything_to_order_by_everything_ties(ddbc, make_source, assert_rows):
    output = rank(ddbc, make_source("select * from (values (5), (6), (7)) t(x)"))
    assert_rows(output, [(5, 0.0), (6, 0.0), (7, 0.0)])


def test_only_numeric_columns_can_be_ordered_by(ddbc, make_source):
    plugin = RankingPlugin()
    plugin.prepare(ddbc, make_source("select 1 as n, 'text' as s, 2.5 as f"))
    assert plugin.order_by.param.choices == ["n", "f"]
    assert "s" in plugin.partition.param.choices  # ... but anything can be partitioned by


def test_config_is_kept(ddbc, make_source):
    plugin = RankingPlugin()
    plugin.prepare(ddbc, make_source(DATA))
    plugin.set_parameter("order_by.0", "x")
    plugin.set_parameter("partition.0", "g")
    parameters = dict(plugin.get_parameters("", "."))
    assert parameters == {"order_by.0": "x", "partition.0": "g"}
