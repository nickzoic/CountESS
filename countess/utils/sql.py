"""SQL which is made from pieces which know what kind of thing they are.

Most of the SQL in CountESS is built with f-strings, and in an f-string everything is just text: nothing says
whether `{x}` is a column name which has to be quoted, a value which has to be escaped, a type, a function, or
a piece of SQL which was made earlier, and so nothing can check that it was dealt with properly.  Getting it
wrong has been quiet and has taken a while to find: a name with brackets in it was put into a regular
expression as it was, a string which should have been an f-string wasn't one so the query had {table_name} in
it, a missing name was written out as the word None, and a NaN as a column which doesn't exist.

Here each piece is a Fragment, of a class which says what it is and which makes it safe, or refuses:

    Id("count (n)")            "count (n)"          a table, column or alias
    Lit("it's")                'it''s'              a value
    Fn("sum")                  sum                  the name of a function or other keyword
    Type("DECIMAL(10, 2)")     DECIMAL(10,2)        a type
    Raw("a > b")               a > b                SQL which is known to be fine, said so on purpose

and a template only puts Fragments together, so something which is still just a str is refused instead of
being let through:

    Sql.template("SELECT {cols} FROM {table} WHERE {col} = {val}",
                 cols=join(Id(c) for c in columns), table=Id(alias), col=Id("n"), val=Lit(5))

The template isn't an f-string, so a name which isn't provided is an error, not text with braces in it.
"""

from __future__ import annotations

import datetime
import decimal
import logging
import math
import re
import string
from typing import Any, Iterable

logger = logging.getLogger(__name__)


class SqlError(ValueError):
    """A value which can't be put in SQL, such as a column which hasn't been chosen yet, so whatever has it isn't
    (yet) set up properly.  This isn't for a mistake in how the pieces are put together (see Sql.template): that is
    a TypeError or KeyError, which is a mistake in the code and not something to wait for somebody to fix."""


class Fragment:
    """A piece of SQL.  `str(fragment)` is the SQL; the class says what kind of piece it is."""

    def __init__(self, sql: str):
        self._sql = sql

    def __str__(self) -> str:
        return self._sql

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self._sql})"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Fragment) and type(self) is type(other) and self._sql == other._sql

    def __hash__(self) -> int:
        return hash((type(self), self._sql))

    def __bool__(self) -> bool:
        return bool(self._sql)


class Raw(Fragment):
    """SQL which is known to be safe.  It's here to be looked for: wherever it is used somebody has said that
    the text is not something which came from outside."""


class Id(Fragment):
    """The name of a table, column or alias, which is always quoted, so it can be anything DuckDB allows."""

    def __init__(self, name: str):
        if name is None:
            raise SqlError("there's no name for this identifier (a column which hasn't been chosen yet?)")
        if not isinstance(name, str):
            raise TypeError(f"an identifier has to be a str, not {name!r}")
        if not name:
            raise SqlError("DuckDB doesn't allow an identifier with nothing in it")
        if "\x00" in name:
            raise SqlError("DuckDB doesn't allow a NUL character in an identifier")
        super().__init__('"' + name.replace('"', '""') + '"')
        self.name = name

    @staticmethod
    def dotted(*names: str) -> Sql:
        """A name with some more in front of it, as in "table"."column" """
        return join((Id(name) for name in names), sep=".")


class Lit(Fragment):
    """A value, written as SQL.  A float is a DOUBLE and a Decimal a DECIMAL of just the size it needs."""

    def __init__(self, value: Any):
        super().__init__(literal_sql(value))
        self.value = value


_FUNCTION_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_TYPE_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_ ]*(\(\s*\d+\s*(,\s*\d+\s*)?\))?(\[\d*\])*")


class Fn(Fragment):
    """The name of a function, or a keyword such as an aggregation like SUM.  It has to look like one."""

    def __init__(self, name: str):
        if not isinstance(name, str) or not _FUNCTION_NAME.fullmatch(name):
            raise SqlError(f"{name!r} isn't the name of a function")
        super().__init__(name)


class Type(Fragment):
    """The name of a type, such as BIGINT, DECIMAL(10,2), TIMESTAMP WITH TIME ZONE or INTEGER[]."""

    def __init__(self, name: str):
        if not isinstance(name, str) or not _TYPE_NAME.fullmatch(name.strip()):
            raise SqlError(f"{name!r} isn't the name of a type: use Raw if it's one of the unusual ones")
        # (spaces made single, and none round brackets or commas, so `DECIMAL (10, 2)` is `DECIMAL(10,2)`)
        super().__init__(re.sub(r"\s*([(),])\s*", r"\1", re.sub(r"\s+", " ", name.strip())))


class Sql(Fragment):
    """A piece of SQL made from other pieces, with `template` or `join`: an expression, a clause, or a whole query."""

    @classmethod
    def template(cls, template: str, **fragments: Fragment) -> Sql:
        """Puts Fragments into a template where it has {name}s, and `{{` and `}}` for braces of its own.  Everything
        in the template has to be given, everything given has to be in it, and each has to be a Fragment: anything
        else is an error, so what kind of thing it is has to be said wherever it is used."""
        text = []
        used = set()
        for literal, field, spec, conversion in string.Formatter().parse(template):
            text.append(literal)
            if field is None:
                continue
            if spec or conversion or not field.isidentifier():
                raise ValueError(f"{{{field}}} in {template!r}: a template only has plain names in braces")
            if field not in fragments:
                raise KeyError(f"{{{field}}} is in the template {template!r} but wasn't given")
            fragment = fragments[field]
            if not isinstance(fragment, Fragment):
                raise TypeError(
                    f"{field} is {fragment!r}, which is a {type(fragment).__name__}: make it an Id, Lit, Fn, "
                    "Type or Sql (or Raw if it's known to be fine) to say what it is"
                )
            used.add(field)
            text.append(str(fragment))
        if unused := set(fragments) - used:
            raise TypeError(f"given to the template {template!r} but not in it: {sorted(unused)}")
        return cls("".join(text))

    def relation(self, ddbc: Any) -> Any:
        """Runs it, giving the relation, and makes a note of what it was if anyone is looking."""
        logger.debug("running %s", self)
        return ddbc.sql(str(self))


EMPTY = Sql("")


def join(fragments: Iterable[Fragment], sep: str = ", ") -> Sql:
    """Fragments with `sep` between them (which is plain text: it's put in as it is)."""
    items = list(fragments)
    for item in items:
        if not isinstance(item, Fragment):
            raise TypeError(f"{item!r} isn't a Fragment: make it an Id, Lit, Fn, Type or Sql")
    return Sql(sep.join(str(item) for item in items))


# ---- values


def literal_sql(value: Any) -> str:
    if value is None:
        return "NULL"
    if hasattr(value, "item") and type(value).__module__ == "numpy":
        value = value.item()  # (numpy numbers and bools, which are none of Python's own)
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if math.isnan(value):
            return "'nan'::DOUBLE"  # (and not nan::DOUBLE, which is a column called nan)
        if math.isinf(value):
            return "'inf'::DOUBLE" if value > 0 else "'-inf'::DOUBLE"
        return f"{value!r}::DOUBLE"
    if isinstance(value, decimal.Decimal):
        if not value.is_finite():
            raise SqlError(f"DuckDB can't have {value} as a DECIMAL")
        _sign, digits, exponent = value.as_tuple()
        assert isinstance(exponent, int)
        if exponent >= 0:
            return f"{value}::DECIMAL({len(digits) + exponent},0)"
        if -exponent > len(digits):
            return f"{value}::DECIMAL({-exponent},{-exponent})"
        return f"{value}::DECIMAL({len(digits)},{-exponent})"
    if isinstance(value, str):
        if "\x00" in value:
            raise SqlError("DuckDB can't have a NUL character in a string")
        return "'" + value.replace("'", "''") + "'"
    if isinstance(value, bytes):
        return "'" + "".join(f"\\x{b:02X}" for b in value) + "'::BLOB"
    if isinstance(value, datetime.datetime):
        return ("TIMESTAMPTZ '" if value.tzinfo else "TIMESTAMP '") + value.isoformat(sep=" ") + "'"
    if isinstance(value, datetime.date):
        return f"DATE '{value.isoformat()}'"
    if isinstance(value, datetime.time):
        return f"TIME '{value.isoformat()}'"
    if isinstance(value, (list, tuple)):
        return "[" + ",".join(literal_sql(item) for item in value) + "]"
    raise TypeError(f"can't write a {type(value).__name__} as a SQL value: {value!r}")


# ---- text which is going to be part of a pattern or a name which is made from a pattern


def regex_escape(text: str) -> str:
    """Text which matches `text` exactly when it is part of a regular expression of DuckDB's (RE2), which a name
    can't be as it is, since a bracket, a plus, a dot and so on are all something else there."""
    return re.escape(text)


def columns_alias(prefix: str = "", suffix: str = "") -> Lit:
    """The alias of a COLUMNS('regex') which has a name made of `prefix`, whatever it was that the first group of
    the regular expression matched, and `suffix`.  It's a template of DuckDB's in which \\1 is the match,
    so a backslash which is wanted as it is has to be doubled, or DuckDB refuses it."""
    return Lit(prefix.replace("\\", "\\\\") + "\\1" + suffix.replace("\\", "\\\\"))
