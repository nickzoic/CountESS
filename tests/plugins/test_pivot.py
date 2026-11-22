import math
import random

import pandas as pd
import pytest

from countess.plugins.pivot import PivotPlugin

LONG = [
    # gene, rep, count, score
    ("g1", "r1", 10, 1.5),
    ("g1", "r2", 20, 2.5),
    ("g2", "r1", 5, 0.5),
    ("g3", "r2", 7, 3.5),
]


def long_table(make_source, rows=LONG):
    values = ", ".join(f"('{g}', '{r}', {c}, {s})" for g, r, c, s in rows)
    return make_source(
        f"select gene, rep, count, score::double as score from (values {values}) t(gene, rep, count, score)"
    )


def pivot(ddbc, source, roles, set_columns, **params):
    plugin = PivotPlugin()
    plugin.prepare(ddbc, source)
    set_columns(plugin.columns, roles)
    for key, value in params.items():
        getattr(plugin, key).value = value
    return plugin.execute(ddbc, source)


def as_dicts(output):
    return [dict(zip(output.columns, row)) for row in output.fetchall()]


def assert_pivot(output, expected, key="gene"):
    """`expected` is a list of dicts, which have to match the output's rows (in any order) in every column"""
    actual = sorted(as_dicts(output), key=lambda r: str(r[key]))
    expected = sorted(expected, key=lambda r: str(r[key]))
    assert len(actual) == len(expected), (actual, expected)
    for got, wanted in zip(actual, expected):
        assert set(got) == set(wanted), (got, wanted)
        for column, value in wanted.items():
            if value is None or (isinstance(value, float) and math.isnan(value)):
                assert got[column] is None, (column, got, wanted)
            elif isinstance(value, (int, float)):
                assert got[column] == pytest.approx(value), (column, got, wanted)
            else:
                assert got[column] == value, (column, got, wanted)


def with_pandas(rows, index, columns, values, aggfunc, long_names=True, fill=None):
    """The same pivot done by pandas, as a list of dicts, with the column names CountESS gives them"""
    frame = pd.DataFrame(rows, columns=["gene", "rep", "count", "score"])
    table = frame.pivot_table(index=index, columns=columns, values=values, aggfunc=aggfunc)
    result = []
    for key, series in table.iterrows():
        row = {index: key}
        for (value_name, *pivot_values), cell in series.items():
            name = (
                f"{value_name}__" + "__".join(f"{c}_{v}" for c, v in zip(columns, pivot_values))
                if long_names
                else f"{value_name}_" + "_".join(pivot_values)
            )
            row[name] = fill if pd.isna(cell) else cell
        result.append(row)
    return result


def test_pivot(ddbc, make_source, set_columns):
    output = pivot(ddbc, long_table(make_source), {"gene": "Index", "rep": "Pivot", "count": "Expand"}, set_columns)
    assert output.columns == ["gene", "count__rep_r1", "count__rep_r2"]
    assert_pivot(output, with_pandas(LONG, "gene", ["rep"], ["count"], "sum"))
    assert_pivot(
        output,
        [
            {"gene": "g1", "count__rep_r1": 10, "count__rep_r2": 20},
            {"gene": "g2", "count__rep_r1": 5, "count__rep_r2": None},  # (no such combination: a NULL)
            {"gene": "g3", "count__rep_r1": None, "count__rep_r2": 7},
        ],
    )


def test_short_names(ddbc, make_source, set_columns):
    output = pivot(
        ddbc,
        long_table(make_source),
        {"gene": "Index", "rep": "Pivot", "count": "Expand"},
        set_columns,
        short_names=True,
    )
    assert output.columns == ["gene", "count_r1", "count_r2"]
    assert_pivot(output, with_pandas(LONG, "gene", ["rep"], ["count"], "sum", long_names=False))


def test_missing_combinations_can_be_zero(ddbc, make_source, set_columns):
    output = pivot(
        ddbc, long_table(make_source), {"gene": "Index", "rep": "Pivot", "count": "Expand"}, set_columns, default_0=True
    )
    assert_pivot(output, with_pandas(LONG, "gene", ["rep"], ["count"], "sum", fill=0))


DUPLICATED = LONG + [("g1", "r1", 30, 4.5), ("g1", "r1", 50, 5.5)]  # g1 has three rows for r1: 10, 30 and 50


@pytest.mark.parametrize("aggfunc", ["sum", "mean", "median", "min", "max"])
def test_aggregation(ddbc, make_source, set_columns, aggfunc):
    output = pivot(
        ddbc,
        long_table(make_source, DUPLICATED),
        {"gene": "Index", "rep": "Pivot", "count": "Expand", "score": "Expand"},
        set_columns,
        aggfunc=aggfunc,
    )
    assert_pivot(output, with_pandas(DUPLICATED, "gene", ["rep"], ["count", "score"], aggfunc))


def test_several_columns_to_pivot_by(ddbc, make_source, assert_rows, set_columns):
    source = make_source(
        """select * from (values
            ('g1', 'r1', 'a', 1), ('g1', 'r1', 'b', 2), ('g1', 'r2', 'a', 3), ('g2', 'r1', 'a', 4)) t(gene, rep, cond, count)"""
    )
    output = pivot(ddbc, source, {"gene": "Index", "rep": "Pivot", "cond": "Pivot", "count": "Expand"}, set_columns)
    assert sorted(output.columns) == sorted(
        ["gene", "count__rep_r1__cond_a", "count__rep_r1__cond_b", "count__rep_r2__cond_a"]
    )
    assert_pivot(
        output,
        [
            {"gene": "g1", "count__rep_r1__cond_a": 1, "count__rep_r1__cond_b": 2, "count__rep_r2__cond_a": 3},
            {"gene": "g2", "count__rep_r1__cond_a": 4, "count__rep_r1__cond_b": None, "count__rep_r2__cond_a": None},
        ],
    )
    short = pivot(
        ddbc,
        source,
        {"gene": "Index", "rep": "Pivot", "cond": "Pivot", "count": "Expand"},
        set_columns,
        short_names=True,
    )
    assert sorted(short.columns) == sorted(["gene", "count_r1_a", "count_r1_b", "count_r2_a"])


def test_several_index_columns(ddbc, make_source, set_columns):
    source = make_source(
        "select * from (values ('g1', 'x', 'r1', 1), ('g1', 'x', 'r2', 2), ('g1', 'y', 'r1', 3)) t(gene, kind, rep, count)"
    )
    output = pivot(ddbc, source, {"gene": "Index", "kind": "Index", "rep": "Pivot", "count": "Expand"}, set_columns)
    assert output.columns == ["gene", "kind", "count__rep_r1", "count__rep_r2"]
    assert_pivot(
        output,
        [
            {"gene": "g1", "kind": "x", "count__rep_r1": 1, "count__rep_r2": 2},
            {"gene": "g1", "kind": "y", "count__rep_r1": 3, "count__rep_r2": None},
        ],
        key="kind",
    )


def test_without_an_index_column_everything_is_one_row(ddbc, make_source, set_columns):
    source = make_source("select rep, count from (values ('r1', 10), ('r2', 20), ('r1', 5), ('r2', 7)) t(rep, count)")
    output = pivot(ddbc, source, {"rep": "Pivot", "count": "Expand"}, set_columns)
    assert as_dicts(output) == [{"count__rep_r1": 15, "count__rep_r2": 27}]


def test_dropped_columns_make_no_difference_without_an_index_column(ddbc, make_source, set_columns):
    output = pivot(ddbc, long_table(make_source), {"rep": "Pivot", "count": "Expand"}, set_columns)
    assert as_dicts(output) == [{"count__rep_r1": 15, "count__rep_r2": 27}]


def test_text_columns_are_joined_up(ddbc, make_source, set_columns):
    source = make_source(
        "select * from (values ('g1', 'r1', 'alpha'), ('g1', 'r2', 'beta'), ('g2', 'r1', 'gamma'), ('g2', 'r1', 'delta')) t(gene, rep, label)"
    )
    output = pivot(ddbc, source, {"gene": "Index", "rep": "Pivot", "label": "Expand"}, set_columns)
    rows = {r["gene"]: r for r in as_dicts(output)}
    assert rows["g1"]["label__rep_r1"] == "alpha" and rows["g1"]["label__rep_r2"] == "beta"
    assert set(rows["g2"]["label__rep_r1"].split(",")) == {"gamma", "delta"}  # (more than one is a list)
    assert rows["g2"]["label__rep_r2"] is None


def test_numbers_and_text_expanded_together(ddbc, make_source, set_columns):
    source = make_source("select * from (values ('g1', 'r1', 4, 'x'), ('g1', 'r2', 6, 'y')) t(gene, rep, count, label)")
    output = pivot(ddbc, source, {"gene": "Index", "rep": "Pivot", "count": "Expand", "label": "Expand"}, set_columns)
    assert as_dicts(output) == [
        {"gene": "g1", "count__rep_r1": 4, "count__rep_r2": 6, "label__rep_r1": "x", "label__rep_r2": "y"}
    ]


def test_columns_which_are_dropped_are_dropped(ddbc, make_source, set_columns):
    output = pivot(ddbc, long_table(make_source), {"gene": "Index", "rep": "Pivot", "count": "Expand"}, set_columns)
    assert not any("score" in c for c in output.columns)  # (score wasn't given a role)


def test_nothing_to_pivot_gives_the_table_back(ddbc, make_source, set_columns):
    source = long_table(make_source)
    assert pivot(ddbc, source, {"gene": "Index", "count": "Expand"}, set_columns) is source  # no Pivot
    assert pivot(ddbc, source, {"gene": "Index", "rep": "Pivot"}, set_columns) is source  # no Expand
    assert pivot(ddbc, source, {}, set_columns) is source


def test_no_table_no_pivot(ddbc):
    assert PivotPlugin().execute(ddbc, None) is None


def test_columns_which_have_gone_are_ignored(ddbc, make_source, set_columns):
    plugin = PivotPlugin()
    plugin.prepare(ddbc, long_table(make_source))
    set_columns(plugin.columns, {"gene": "Index", "rep": "Pivot", "count": "Expand", "score": "Expand"})
    narrower = make_source("select gene, rep, count from (values ('g1', 'r1', 3)) t(gene, rep, count)")
    assert as_dicts(plugin.execute(ddbc, narrower)) == [{"gene": "g1", "count__rep_r1": 3}]


def test_pivot_columns_whose_names_contain_the_separator_it_would_choose(ddbc, make_source, set_columns):
    """The pivot tells its own columns from the rest with a character not in the names of any of the others."""
    source = make_source(
        """select * from (values ('g1', 'r1', 1), ('g1', 'r2', 2)) t("a~b!c@d#e%f&g_h=i:j;k,l/m", rep, "~!@#%&_=:;,/")"""
    )
    output = pivot(
        ddbc, source, {"a~b!c@d#e%f&g_h=i:j;k,l/m": "Index", "rep": "Pivot", "~!@#%&_=:;,/": "Expand"}, set_columns
    )
    assert output.columns == ["a~b!c@d#e%f&g_h=i:j;k,l/m", "~!@#%&_=:;,/__rep_r1", "~!@#%&_=:;,/__rep_r2"]
    assert output.fetchall() == [("g1", 1, 2)]


def test_values_which_need_quoting(ddbc, make_source, set_columns):
    source = make_source("""select * from (values ('it''s', 'r "1"', 1), ('it''s', 'r 2', 2)) t(gene, rep, count)""")
    output = pivot(ddbc, source, {"gene": "Index", "rep": "Pivot", "count": "Expand"}, set_columns)
    assert sorted(output.columns) == sorted(["gene", 'count__rep_r "1"', "count__rep_r 2"])
    assert as_dicts(output) == [{"gene": "it's", 'count__rep_r "1"': 1, "count__rep_r 2": 2}]


# ---- column names and values which are awkward, or which look like each other

AWKWARD_NAMES = [
    "count (n)", "a+b", "x[1]", "a.b", "a|b", "a*", "a?", "^x", "x$", "back\\slash", "c{2}", "a b",
    'q"uote', "100%", "x-y", "#hash", "a&b", "a~b", "ünï", "tab\there", "new\nline", "(", ")", "[", "\\", ".*",
]  # fmt: skip


def sql_id(name):
    return '"' + name.replace('"', '""') + '"'


@pytest.mark.parametrize("short_names", [False, True], ids=["long names", "short names"])
@pytest.mark.parametrize("name", AWKWARD_NAMES)
def test_expand_columns_can_have_any_characters_in_their_names(ddbc, make_source, set_columns, name, short_names):
    """(The name of an expand column used to be put into a regular expression just as it was, so brackets,
    plus signs and the like meant that its columns couldn't be found again: "No matching columns found".)"""
    source = make_source(f"select * from (values ('g1', 'r1', 1), ('g1', 'r2', 2)) t(gene, rep, {sql_id(name)})")
    output = pivot(
        ddbc, source, {"gene": "Index", "rep": "Pivot", name: "Expand"}, set_columns, short_names=short_names
    )
    joiner, label = ("_", "") if short_names else ("__", "rep_")
    assert output.columns == ["gene", f"{name}{joiner}{label}r1", f"{name}{joiner}{label}r2"]
    assert output.fetchall() == [("g1", 1, 2)]


@pytest.mark.parametrize(
    "first,second", [("n", "n2"), ("count", "count_2"), ("score", "score_se"), ("n", "an"), ("a", "a a"), ("x", "x)")]
)
def test_expand_columns_whose_names_begin_or_end_the_same_are_kept_apart(ddbc, make_source, set_columns, first, second):
    """(Each expand column's output columns were found with an unanchored pattern, so those of "n" also matched
    those of "n2": the output had two columns called n__rep_r1, the second with the numbers of n2 in it.)"""
    source = make_source(
        f"select * from (values ('g1', 'r1', 1, 100), ('g1', 'r2', 2, 200)) t(gene, rep, {sql_id(first)}, {sql_id(second)})"
    )
    output = pivot(ddbc, source, {"gene": "Index", "rep": "Pivot", first: "Expand", second: "Expand"}, set_columns)
    assert len(output.columns) == len(set(output.columns))  # no two the same
    assert as_dicts(output) == [
        {
            "gene": "g1",
            f"{first}__rep_r1": 1,
            f"{first}__rep_r2": 2,
            f"{second}__rep_r1": 100,
            f"{second}__rep_r2": 200,
        }
    ]


@pytest.mark.parametrize(
    "value", ["a\nb", "line1\r\nline2", "a(b)", "a.b+c", "x_~y", "tab\there", "^$", "back\\slash", "[x]"]
)
def test_pivot_values_can_have_any_characters_in_them(ddbc, set_columns, value):
    """(A newline in one used to cut off the front of the name of its columns, as a . doesn't match one.)"""
    ddbc.execute("create or replace table awkward as select 'g1' as gene, ? as rep, 1 as n", [value])
    output = pivot(ddbc, ddbc.table("awkward"), {"gene": "Index", "rep": "Pivot", "n": "Expand"}, set_columns)
    assert output.columns == ["gene", f"n__rep_{value}"]
    assert output.fetchall() == [("g1", 1)]


def test_any_column_names_at_all(ddbc, set_columns):
    """The same, for a lot of names made up at random (and a second expand column which has the first as the start of its name)"""
    rng = random.Random(20260531)
    alphabet = list("()[]{}?*+-|^$\\.&~#_ \"'%/;:,=!@") + ["é", "日", "\t", "\n"] + list("abc012")
    failures = []
    for trial in range(120):
        first = "".join(rng.choice(alphabet) for _ in range(rng.randint(1, 10)))
        second = first + "".join(rng.choice(alphabet) for _ in range(rng.randint(1, 4)))
        if {first, second} & {"gene", "rep"} or first == second:
            continue
        ddbc.execute(
            f"create or replace table fuzz as select 'g' as gene, 'r' as rep, 1 as {sql_id(first)}, 2 as {sql_id(second)}"
        )
        try:
            output = pivot(
                ddbc,
                ddbc.table("fuzz"),
                {"gene": "Index", "rep": "Pivot", first: "Expand", second: "Expand"},
                set_columns,
            )
            got = dict(zip(output.columns, output.fetchone()))
            if got != {"gene": "g", f"{first}__rep_r": 1, f"{second}__rep_r": 2} or len(output.columns) != 3:
                failures.append((first, second, got))
        except Exception as exc:  # pylint: disable=broad-exception-caught
            failures.append((first, second, f"{type(exc).__name__}: {str(exc).splitlines()[0][:60]}"))
    assert not failures, f"{len(failures)} of the names went wrong, such as {failures[:3]}"


# ---- columns which are dropped have no say


def wide_table(make_source):
    return make_source(
        """select * from (values
            ('g1', 'r1', 10, 'a', 1.5, true), ('g1', 'r2', 20, 'b', 2.5, false), ('g2', 'r1', 5, NULL, NULL, true),
            ('g3', 'r2', 7, 'd', 3.5, NULL)) t(gene, rep, n, note, score, flag)"""
    )


def test_dropped_columns_of_any_kind_make_no_difference_without_an_index_column(ddbc, make_source, set_columns):
    output = pivot(ddbc, wide_table(make_source), {"rep": "Pivot", "n": "Expand"}, set_columns)
    assert as_dicts(output) == [{"n__rep_r1": 15, "n__rep_r2": 27}]  # (text, numbers, booleans and nulls, all ignored)


def test_dropped_columns_make_no_difference_with_an_index_column(ddbc, make_source, set_columns):
    wide = pivot(ddbc, wide_table(make_source), {"gene": "Index", "rep": "Pivot", "n": "Expand"}, set_columns)
    narrow_source = make_source(
        "select gene, rep, n from (values ('g1', 'r1', 10), ('g1', 'r2', 20), ('g2', 'r1', 5), ('g3', 'r2', 7)) t(gene, rep, n)"
    )
    narrow = pivot(ddbc, narrow_source, {"gene": "Index", "rep": "Pivot", "n": "Expand"}, set_columns)
    assert sorted(wide.fetchall(), key=repr) == sorted(narrow.fetchall(), key=repr)
    assert wide.columns == narrow.columns


@pytest.mark.parametrize("short_names", [False, True])
@pytest.mark.parametrize("default_0", [False, True])
def test_no_index_column_with_several_of_everything(ddbc, make_source, set_columns, short_names, default_0):
    source = make_source(
        """select * from (values
            ('a', 'x', 1, 10, 'junk'), ('a', 'y', 2, 20, 'junk'), ('b', 'x', 3, 30, 'more'), ('b', 'x', 4, 40, 'more')) t(p, q, u, v, dropped)"""
    )
    output = pivot(
        ddbc,
        source,
        {"p": "Pivot", "q": "Pivot", "u": "Expand", "v": "Expand"},
        set_columns,
        short_names=short_names,
        default_0=default_0,
    )
    rows = as_dicts(output)
    assert len(rows) == 1
    row = rows[0]
    prefix = {"_": ("a_x", "a_y", "b_x"), "__": ("p_a__q_x", "p_a__q_y", "p_b__q_x")}["_" if short_names else "__"]
    joiner = "_" if short_names else "__"
    assert row[f"u{joiner}{prefix[0]}"] == 1 and row[f"u{joiner}{prefix[1]}"] == 2 and row[f"u{joiner}{prefix[2]}"] == 7
    assert (
        row[f"v{joiner}{prefix[0]}"] == 10 and row[f"v{joiner}{prefix[1]}"] == 20 and row[f"v{joiner}{prefix[2]}"] == 70
    )
    assert len(row) == 6  # (and nothing else: there was a row for every combination of what's dropped before)
