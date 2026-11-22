"""Fixtures shared by the plugin tests."""

import itertools
import math

import duckdb
import pytest


@pytest.fixture(name="ddbc")
def fixture_ddbc():
    connection = duckdb.connect()
    yield connection
    connection.close()


@pytest.fixture(name="make_source")
def fixture_make_source(ddbc):
    """make_source("select ...") makes a table from the query and returns it as the relation
    which a plugin is given as its input (plugins build their SQL from the relation's name)."""
    names = itertools.count()

    def make_source(select: str):
        name = f"src_{next(names)}"
        ddbc.sql(f"create table {name} as {select}")
        return ddbc.table(name)

    return make_source


@pytest.fixture(name="set_columns")
def fixture_set_columns():
    """set_columns(plugin.columns, {"colname": value}) for the PerColumnArrayParams, which
    get a row for each column of the input once the plugin has been prepared."""

    def set_columns(array_param, values):
        found = set()
        for param in array_param.params:
            if param.label in values:
                param.value = values[param.label]
                found.add(param.label)
        assert found == set(values), f"no such columns: {set(values) - found}"

    return set_columns


def same(actual, expected, tolerance):
    if isinstance(expected, float) and isinstance(actual, (int, float)):
        if math.isnan(expected):
            return math.isnan(actual)
        return math.isclose(actual, expected, rel_tol=tolerance, abs_tol=tolerance)
    return actual == expected


def sort_key(row):
    # (floats are rounded so that rows which are the same to within a tolerance sort the same way)
    return [repr(round(v, 6)) if isinstance(v, float) and not math.isnan(v) else repr(v) for v in row]


@pytest.fixture(name="assert_rows")
def fixture_assert_rows():
    """assert_rows(relation_or_rows, expected, ordered=False, tolerance=1e-9) compares rows,
    allowing for floating point, NaN==NaN, and rows in any order unless `ordered`."""

    def assert_rows(actual, expected, ordered=False, tolerance=1e-9):
        rows = actual.fetchall() if hasattr(actual, "fetchall") else list(actual)
        rows = [tuple(r) for r in rows]
        expected = [tuple(r) for r in expected]
        if not ordered:
            rows.sort(key=sort_key)
            expected.sort(key=sort_key)
        assert len(rows) == len(expected), f"{len(rows)} rows, expected {len(expected)}:\n{rows}\n{expected}"
        for row, wanted in zip(rows, expected):
            assert len(row) == len(wanted) and all(
                same(a, e, tolerance) for a, e in zip(row, wanted)
            ), f"row {row} != {wanted}\nall rows: {rows}"

    return assert_rows
