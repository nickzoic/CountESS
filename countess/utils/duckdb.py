import logging
import secrets
from typing import Iterable, Optional, Union

from duckdb import DuckDBPyConnection, DuckDBPyRelation
from duckdb.sqltypes import DuckDBPyType

from countess.utils.sql import Id, Lit, Sql, join

logger = logging.getLogger(__name__)


# XXX This stuff would probably be better as a DuckdbWrapper class


def duckdb_dtype_is_boolean(dtype: DuckDBPyType) -> bool:
    return dtype.id == "boolean"


def duckdb_dtype_is_integer(dtype: DuckDBPyType) -> bool:
    return dtype.id in (
        "bigint",
        "hugeint",
        "integer",
        "smallint",
        "tinyint",
        "ubigint",
        "uhugeint",
        "uinteger",
        "usmallint",
        "utinyint",
    )


def duckdb_dtype_is_numeric(dtype: DuckDBPyType) -> bool:
    return dtype.id in ("float", "decimal", "double") or duckdb_dtype_is_integer(dtype)


def duckdb_dtype_to_datatype_choice(dtype: DuckDBPyType) -> str:
    """Convert a duckdb dtype to make the choices of a DataTypeChoicesParam"""
    if duckdb_dtype_is_boolean(dtype):
        return "BOOLEAN"
    if duckdb_dtype_is_integer(dtype):
        return "INTEGER"
    elif duckdb_dtype_is_numeric(dtype):
        return "FLOAT"
    else:
        return "STRING"


def duckdb_escape_identifier(identifier: str) -> str:
    """Deprecated: it's countess.utils.sql.Id, which is what this uses, but this lets a None through (and an
    Id doesn't: a name which isn't there is an SqlError)."""
    if identifier is None:
        return None
    return str(Id(identifier))


def duckdb_escape_literal(literal: Union[str, int, float, list, None]) -> str:
    """Deprecated: use countess.utils.sql.Lit, which is what this uses.

    XXX this shouldn't be a thing *BUT* not all functions seem to accept parameter bindings (they can't be in a
    view, or in a PIVOT, or be an identifier), or at least it isn't clearly documented as such."""
    return str(Lit(literal))


def duckdb_combine(ddbc: DuckDBPyConnection, sources: Iterable[DuckDBPyRelation]) -> Optional[DuckDBPyRelation]:
    """Combine several tables into a new table using UNION ALL BY NAME"""

    sql = join(
        (Sql.template("SELECT * FROM {table}", table=Id(source.alias)) for source in sources if source is not None),
        sep=" UNION ALL BY NAME ",
    )
    if not sql:
        # there are no sources, and we can't have a table without columns, so we have to return None.
        return None

    logger.debug("duckdb_combine %s", sql)
    return duckdb_source_to_view(ddbc, sql.relation(ddbc))


def duckdb_source_to_view(cursor: DuckDBPyConnection, source: DuckDBPyRelation, view_name: Optional[str] = None):
    if view_name is None:
        view_name = "v_" + secrets.token_hex(16)
    logger.debug("creating view %s for %s", view_name, source.alias)
    source.to_view(view_name)
    return cursor.view(view_name)


def duckdb_source_to_table(cursor: DuckDBPyConnection, source: DuckDBPyRelation, table_name: Optional[str] = None):
    if table_name is None:
        table_name = "t_" + secrets.token_hex(16)
    logger.debug("creating table %s for %s", table_name, source.alias)
    source.to_table(table_name)
    return cursor.table(table_name)


def duckdb_choose_special(strings: Iterable[str]) -> str:
    """Pick a special string for use in identifiers etc, which is
    a unicode character which is not special for regexes (because of the
    COLUMN function) and is not in any of the `strings` and isn't some
    weird combining character which will make the logs unreadable."""
    chars = set(c for s in strings for c in s)
    for c in "~!@#%&_=:;,/":
        if c not in chars:
            return c
    for n in range(0xA1, 0x2FF):
        if chr(n) not in chars:
            return chr(n)
    raise NotImplementedError("couldn't choose separator")
