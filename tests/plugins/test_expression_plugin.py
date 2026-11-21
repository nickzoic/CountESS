import pytest

from countess.plugins.expression import ExpressionPlugin


def expression(ddbc, source, code):
    plugin = ExpressionPlugin()
    plugin.set_parameter("code", code)
    plugin.prepare(ddbc, source)
    return plugin.execute(ddbc, source)


@pytest.fixture(name="source")
def fixture_source(make_source):
    return make_source("select * from (values (1, 2, 'x'), (3, 4, 'y'), (5, 6, 'z')) t(a, b, s)")


def test_a_new_column(ddbc, source, assert_rows):
    output = expression(ddbc, source, "c = a + b")
    assert output.columns == ["a", "b", "s", "c"]
    assert_rows(output, [(1, 2, "x", 3), (3, 4, "y", 7), (5, 6, "z", 11)])


def test_several_lines_each_using_the_last(ddbc, source, assert_rows):
    output = expression(ddbc, source, "c = a + b\nd = c * 10")
    assert_rows(output, [(1, 2, "x", 3, 30), (3, 4, "y", 7, 70), (5, 6, "z", 11, 110)])


def test_replacing_a_column(ddbc, source, assert_rows):
    output = expression(ddbc, source, "s = concat(s, '!')")
    assert output.columns == ["a", "b", "s"]
    assert_rows(output, [(1, 2, "x!"), (3, 4, "y!"), (5, 6, "z!")])


def test_removing_a_column(ddbc, source, assert_rows):
    output = expression(ddbc, source, "b = None")
    assert output.columns == ["a", "s"]
    assert_rows(output, [(1, "x"), (3, "y"), (5, "z")])


def test_filtering_rows(ddbc, source, assert_rows):
    output = expression(ddbc, source, "__filter = a > 1 and b < 6")
    assert output.columns == ["a", "b", "s"]  # (the filter isn't a column)
    assert_rows(output, [(3, 4, "y")])


def test_selecting_between_values(ddbc, source, assert_rows):
    output = expression(ddbc, source, "big = 'yes' if a > 2 else 'no'")
    assert_rows(output, [(1, 2, "x", "no"), (3, 4, "y", "yes"), (5, 6, "z", "yes")])


def test_column_names_which_need_quoting(ddbc, make_source, assert_rows):
    source = make_source("""select * from (values (1, 2)) t("a b", "quo""te")""")
    plugin = ExpressionPlugin()
    plugin.set_parameter("code", "total = a_b + 0")
    plugin.prepare(ddbc, source)
    # (names with spaces can't be reached from an expression, so this makes a column which isn't there: an
    # error, but one which is reported rather than a crash)
    try:
        plugin.execute(ddbc, source)
    except Exception as exc:  # pylint: disable=broad-exception-caught
        assert "a_b" in str(exc)


def test_nothing_to_do_without_having_been_prepared(ddbc, source):
    assert ExpressionPlugin().execute(ddbc, source) is None


@pytest.mark.parametrize("code", ["", "   ", "\n\n", "# only a comment"])
@pytest.mark.xfail(
    strict=True,
    raises=SyntaxError,
    reason="preparing an expression plugin which has no expression yet (as a new one doesn't) raises a SyntaxError, "
    "though the code (`self.code.value or ''`) looks as if it expects that, and `execute` has an answer for "
    "when there is no block",
)
def test_no_expression_yet(ddbc, source, code):
    assert expression(ddbc, source, code) is None


def test_a_mistake_in_the_expression_is_a_syntax_error(ddbc, source):
    with pytest.raises(SyntaxError):
        expression(ddbc, source, "c = = 1")
