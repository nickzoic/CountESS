"""Testing a DuckDBPyRelation for truthiness (`if source:`) calls `len()` on it, which runs `count(*)` over the whole
relation: for a lazy chain of nodes that means running everything upstream again, once for every such test.
Plugins must check `is not None` instead."""

from countess.core.plugins import DuckdbSqlPlugin
from countess.plugins.join import JoinPlugin


class PassThrough(DuckdbSqlPlugin):
    name = "PassThrough"
    version = "0"

    def sql(self, table_name, columns):
        return f"SELECT * FROM {table_name}"


def make_expensive_relation(ddbc, name):
    """A relation which DuckDB can't answer `count(*)` for without calling the (counting) function on every row,
    because of the DISTINCT over its output."""
    calls = []

    def expensive(i: int) -> int:
        calls.append(i)
        return i

    ddbc.create_function(f"expensive_{name}", expensive)
    ddbc.sql(f"create table base_{name} as select i from range(50) t(i)")
    # (as the pipeline does: a lazy view, which later nodes find by name)
    ddbc.sql(f"select distinct i, expensive_{name}(i) as v from base_{name}").to_view(f"lazy_{name}")
    return ddbc.table(f"lazy_{name}"), calls


def test_the_hazard_is_real(ddbc):
    relation, calls = make_expensive_relation(ddbc, "hazard")
    assert bool(relation)
    assert len(calls) == 50


def test_preparing_and_executing_a_plugin_doesnt_run_its_source(ddbc):
    relation, calls = make_expensive_relation(ddbc, "plugin")
    plugin = PassThrough()
    plugin.prepare_multi(ddbc, {"input": relation})
    result = plugin.execute_multi(ddbc, {"input": relation})
    assert result is not None
    assert not calls  # (nothing has asked for any rows yet)
    assert len(result.fetchall()) == 50
    assert len(calls) == 50


def test_an_empty_source_is_still_a_source(ddbc):
    ddbc.sql("create table nothing as select 1 as i where false")
    plugin = PassThrough()
    plugin.prepare_multi(ddbc, {"input": ddbc.table("nothing")})
    result = plugin.execute_multi(ddbc, {"input": ddbc.table("nothing")})
    assert result is not None
    assert result.columns == ["i"]
    assert result.fetchall() == []


def test_joining_runs_each_source_once_and_not_more_for_the_sake_of_log_messages(ddbc):
    """(A join does run once to find out if the join can be made at all, but that's all.)"""
    left, left_calls = make_expensive_relation(ddbc, "left")
    right, right_calls = make_expensive_relation(ddbc, "right")
    plugin = JoinPlugin()
    result = plugin.execute_multi(ddbc, {"left": left, "right": right})
    assert result is not None
    assert len(left_calls) == 50
    assert len(right_calls) == 50
