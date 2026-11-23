"""A column of a CSV file which is given the type FLOAT is loaded as a DECIMAL which is exactly the numbers in the
file, of just the size which they need, rather than as DuckDB's single precision FLOAT."""

import bz2
import decimal
import gzip
import logging
import lzma
import random

import pytest

from countess.plugins.csv import CSV_PREVIEW_ROWS, DECIMAL_MAX_DIGITS, LoadCsvPlugin, csv_cast_sql, csv_decimal_sizes
from countess.utils.sql import Raw

TEXT = (
    "n,x,y\n"
    "3000000000,0.123456789012345,1.78e-05\n"
    "2147483648,2.01,7642656515e-5\n"
    "-9223372036854775808,-1.5,\n"
    "9223372036854775807,12345678.9,nan\n"
    "99999999999999999999,abc, 3.5 \n"
)
D = decimal.Decimal


def load(ddbc, path, columns=(("n", "INTEGER"), ("x", "FLOAT"), ("y", "FLOAT")), row_limit=None, **params):
    plugin = LoadCsvPlugin()
    plugin.set_parameter("files.0.filename", str(path))
    plugin.set_parameter("header", True)
    for key, value in params.items():
        plugin.set_parameter(key, value)
    for number, (name, type_) in enumerate(columns):
        plugin.set_parameter(f"columns.{number}.name", name)
        plugin.set_parameter(f"columns.{number}.type", type_)
    return plugin.execute(ddbc, None, row_limit)


def write(path, text):
    opener = {".gz": gzip.open, ".bz2": bz2.open, ".xz": lzma.open}.get(path.suffix, open)
    with opener(path, "wt", encoding="utf-8") as fh:
        fh.write(text)
    return path


def types(relation):
    return [str(t) for t in relation.dtypes]


def test_integers_are_64_bit_and_floats_are_exact_decimals(ddbc, tmp_path):
    output = load(ddbc, write(tmp_path / "a.csv", TEXT))
    assert types(output) == ["BIGINT", "DECIMAL(23,15)", "DECIMAL(12,7)"]  # (the size of what's in each column)
    assert output.fetchall() == [
        (3000000000, D("0.123456789012345"), D("0.0000178")),
        (2147483648, D("2.01"), D("76426.56515")),
        (-9223372036854775808, D("-1.5"), None),
        (9223372036854775807, D("12345678.9"), None),  # (nan isn't a number which a DECIMAL can be)
        (None, None, D("3.5")),  # (too big for 64 bits, and not a number at all, and with spaces round it)
    ]


@pytest.mark.parametrize("extension", [".gz", ".bz2", ".xz"])
def test_it_is_the_same_whether_or_not_the_file_is_compressed(ddbc, tmp_path, extension):
    """(Files are read in different ways, and the compressed ones, which can only be read once, as they are
    read are what's sized first.)"""
    plain = load(ddbc, write(tmp_path / "a.csv", TEXT))
    compressed = load(ddbc, write(tmp_path / f"a.csv{extension}", TEXT))
    assert types(compressed) == types(plain)
    assert compressed.fetchall() == plain.fetchall()


@pytest.mark.parametrize("extension", ["", ".bz2", ".xz"])
def test_a_preview_is_sized_by_the_rows_it_shows(ddbc, tmp_path, extension):
    output = load(ddbc, write(tmp_path / f"a.csv{extension}", TEXT), row_limit=1)
    assert types(output) == ["BIGINT", "DECIMAL(16,15)", "DECIMAL(8,7)"]
    assert output.fetchall() == [(3000000000, D("0.123456789012345"), D("0.0000178"))]


def test_the_number_of_rows_shown_in_the_csv_preview_is_not_confused_with_this_one():
    assert CSV_PREVIEW_ROWS == 100 and DECIMAL_MAX_DIGITS == 38


@pytest.mark.parametrize(
    "text", ["5e-10", "1e-5", "1E-5", "1.5e-3", "7642656515e-5", "0.1e-4", "12345e-2", "-2.5e+3", "1e5"]
)
def test_numbers_with_an_exponent_are_not_lost(ddbc, tmp_path, text):
    """(DuckDB can't cast some of these to a DECIMAL of the size which holds them: 1e-5 as a DECIMAL(5,5) is a NULL.)"""
    path = write(tmp_path / "e.csv", f"v\n{text}\n")
    output = load(ddbc, path, columns=(("v", "FLOAT"),))
    assert output.fetchall() == [(D(text),)]


def test_a_lot_of_awkward_numbers_are_each_exactly_what_is_in_the_file(ddbc, tmp_path):
    """(Up to 7 digits and an exponent of up to 6 either way, so the most any column needs is well inside the
    38 digits which a DECIMAL can have: with more than that, numbers have to be rounded, which is tested below.)"""
    rng = random.Random(20260531)
    lines = []
    for _ in range(400):
        whole = "".join(rng.choice("0123456789") for _ in range(rng.randint(0, 7)))
        fraction = "".join(rng.choice("0123456789") for _ in range(rng.randint(0, 7)))
        if not (whole or fraction):
            continue
        exponent = rng.choice(["", "", "", f"e{rng.randint(-6, 6)}", f"E+{rng.randint(0, 6)}"])
        lines.append(
            rng.choice(["", "-", "+"]) + whole + ("." if fraction or rng.random() < 0.3 else "") + fraction + exponent
        )
    path = write(tmp_path / "many.csv", "v\n" + "\n".join(lines) + "\n")
    output = load(ddbc, path, columns=(("v", "FLOAT"),))
    got = [row[0] for row in output.fetchall()]
    assert got == [D(line) for line in lines]  # (every one, with nothing rounded or lost, in order)


@pytest.mark.parametrize(
    "text,expected",
    [("+2", D(2)), (".5", D("0.5")), ("5.", D(5)), ("  3.25  ", D("3.25")), ("007.50", D("7.5")), ("-0", D(0))],
)
def test_the_ways_numbers_are_written_which_are_accepted(ddbc, tmp_path, text, expected):
    output = load(ddbc, write(tmp_path / "w.csv", f"v\n{text}\n"), columns=(("v", "FLOAT"),))
    assert output.fetchall() == [(expected,)]


@pytest.mark.parametrize("text", ["nan", "NaN", "inf", "-inf", "abc", "1,5", "1e500", "0x10", "--1", "1.2.3"])
def test_what_is_not_a_number_which_a_decimal_can_be_is_empty(ddbc, tmp_path, text):
    path = write(tmp_path / "n.csv", f'v,w\n"{text}",1.5\n')
    output = load(ddbc, path, columns=(("v", "FLOAT"), ("w", "FLOAT")))
    assert output.fetchall() == [(None, D("1.5"))]


def test_a_column_with_no_numbers_in_it_is_as_small_a_decimal_as_there_is(ddbc, tmp_path):
    output = load(ddbc, write(tmp_path / "e.csv", "v,w\nabc,1\n,2\n"), columns=(("v", "FLOAT"), ("w", "FLOAT")))
    assert types(output) == ["DECIMAL(1,0)", "DECIMAL(1,0)"]
    assert output.fetchall() == [(None, D(1)), (None, D(2))]


def test_a_column_which_is_in_the_settings_and_not_in_the_file_is_empty_and_the_same_type(ddbc, tmp_path):
    output = load(ddbc, write(tmp_path / "a.csv", "a\n1.5\n"), columns=(("a", "FLOAT"), ("extra", "FLOAT")))
    assert types(output) == ["DECIMAL(2,1)", "DECIMAL(1,0)"]
    assert output.fetchall() == [(D("1.5"), None)]


def test_numbers_too_big_for_a_decimal_make_it_a_double_and_say_so(ddbc, tmp_path, caplog):
    path = write(tmp_path / "b.csv", "a,b\n1e50,1\n2,2\n")
    with caplog.at_level(logging.WARNING):
        output = load(ddbc, path, columns=(("a", "FLOAT"), ("b", "FLOAT")))
    assert types(output) == ["DOUBLE", "DECIMAL(1,0)"]
    assert output.fetchall() == [(1e50, D(1)), (2.0, D(2))]
    assert "too big to be a DECIMAL" in caplog.text


def test_more_digits_than_a_decimal_has_are_rounded_to_fit_and_it_says_so(ddbc, tmp_path, caplog):
    path = write(
        tmp_path / "r.csv", "v\n123456789012345678901234567890.5\n0.1234567890123456789012345678901234567890\n"
    )
    with caplog.at_level(logging.WARNING):
        output = load(ddbc, path, columns=(("v", "FLOAT"),))
    assert types(output) == ["DECIMAL(38,8)"]  # (30 digits before the point leave 8 after it)
    assert "rounding them to fit" in caplog.text


def test_the_sizes_can_be_worked_out_of_text_columns(ddbc):
    ddbc.sql("create table t as select * from (values ('1.5', '100'), ('22.25', '-7'), (NULL, 'x')) v(a, b)")
    assert csv_decimal_sizes(ddbc.table("t"), ["a", "b"]) == {"a": (4, 2), "b": (3, 0)}
    assert csv_decimal_sizes(ddbc.table("t"), []) == {}


def test_the_cast_of_a_value_which_isnt_there_is_a_plain_one(ddbc):
    """(There's no text to look at for an exponent in a NULL, so there's no need for the slow way.)"""
    sql = str(csv_cast_sql(Raw("NULL"), "FLOAT", (5, 2), decimal=True))
    assert sql == "TRY_CAST(NULL AS DECIMAL(5,2))"
    assert "contains" in str(csv_cast_sql(Raw("'1e-5'"), "FLOAT", (5, 5), decimal=True))


def test_columns_of_the_other_types_are_cast_as_they_were(ddbc, tmp_path):
    path = write(tmp_path / "o.csv", "s,b,i\nhello,true,7\n")
    output = load(ddbc, path, columns=(("s", "STRING"), ("b", "BOOLEAN"), ("i", "INTEGER")))
    assert types(output) == ["VARCHAR", "BOOLEAN", "BIGINT"]
    assert output.fetchall() == [("hello", True, 7)]
