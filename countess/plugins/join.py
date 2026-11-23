import logging
from typing import Dict, Mapping, Optional

import duckdb
from duckdb import DuckDBPyConnection, DuckDBPyRelation
from duckdb.sqltypes import DuckDBPyType

from countess import VERSION
from countess.core.parameters import ArrayParam, BooleanParam, ColumnChoiceParam, ColumnOrNoneChoiceParam, MultiParam
from countess.core.plugins import DuckdbPlugin
from countess.utils.duckdb import duckdb_dtype_is_numeric
from countess.utils.sql import Fragment, Id, Lit, Raw, Sql, join

logger = logging.getLogger(__name__)


class JoinTypeError(TypeError):
    """Raised when columns which are being joined on have types which can't be compared with each other."""


# DuckDB types which can hold other values, which it makes no sense to compare with plain text or numbers
_NESTED_TYPE_IDS = {"list", "array", "struct", "map", "union"}


def _common_type(ddbc: DuckDBPyConnection, left_type: DuckDBPyType, right_type: DuckDBPyType) -> Optional[str]:
    """The type which DuckDB would combine two types into (so INTEGER and DOUBLE make DOUBLE, DATE and TIMESTAMP make
    TIMESTAMP), or None if it can't."""
    # (the types are written by DuckDB, so are fine as they are, nested ones like STRUCT(a INTEGER) as well)
    query = Sql.template(
        "SELECT typeof(CASE WHEN true THEN NULL::{left} ELSE NULL::{right} END)",
        left=Raw(str(left_type)),
        right=Raw(str(right_type)),
    )
    try:
        row = query.relation(ddbc).fetchone()
    except duckdb.Error:
        return None
    return row[0] if row else None


def _comparable(
    ddbc: DuckDBPyConnection, left: Fragment, left_type: DuckDBPyType, right: Fragment, right_type: DuckDBPyType
) -> tuple[Fragment, Fragment]:
    """Given SQL expressions for two columns and their types, work out from the types alone (without running
    anything) what to compare, returning an expression for each which can be compared with `=`, or raise
    JoinTypeError if there's no sensible way to do that.

    * the same type is compared as it is.
    * text is compared with anything which isn't a nested type by reading the text as that type (which is what
      DuckDB does on its own, except that it gives up with an error on the first value it can't read, but here
      text which can't be read as that type becomes NULL, which never matches anything).
    * otherwise both are converted to the type DuckDB makes of the two, so for example the 32 bit and 64 bit
      integers or the date and the timestamp are compared as the wider type.
    """

    if str(left_type) == str(right_type):
        return left, right

    left_is_text, right_is_text = left_type.id == "varchar", right_type.id == "varchar"
    if left_is_text != right_is_text:
        other_type = right_type if left_is_text else left_type
        if other_type.id not in _NESTED_TYPE_IDS:
            if left_is_text:
                return Sql.template("TRY_CAST({value} AS {type})", value=left, type=Raw(str(other_type))), right
            return left, Sql.template("TRY_CAST({value} AS {type})", value=right, type=Raw(str(other_type)))
    elif common := _common_type(ddbc, left_type, right_type):

        def converted(value: Fragment, value_type: DuckDBPyType) -> Fragment:
            if str(value_type) == common:
                return value
            return Sql.template("CAST({value} AS {type})", value=value, type=Raw(common))

        return converted(left, left_type), converted(right, right_type)

    raise JoinTypeError(f"can't compare {left_type} with {right_type}")


def _join_how(left_required: bool, right_required: bool) -> str:
    if left_required:
        return "INNER" if right_required else "LEFT OUTER"
    else:
        return "RIGHT OUTER" if right_required else "FULL OUTER"


class JoinPlugin(DuckdbPlugin):
    """Joins DuckDB tables"""

    name = "Join"
    description = "Joins by indexes or columns"
    version = VERSION
    link = "https://countess-project.github.io/CountESS/included-plugins/#join"

    class InputMultiParam(MultiParam):
        join_on = ColumnChoiceParam("Join On")
        join_2 = ColumnOrNoneChoiceParam("Join 2")
        join_3 = ColumnOrNoneChoiceParam("Join 3")
        required = BooleanParam("Required", True)
        drop = BooleanParam("Drop Column", False)

    inputs = ArrayParam("Inputs", InputMultiParam("Input"), read_only=True)

    join_params = None
    input_columns_1: Optional[Dict] = None
    input_columns_2: Optional[Dict] = None

    def execute_multi(
        self, ddbc: DuckDBPyConnection, sources: Mapping[str, DuckDBPyRelation], row_limit: Optional[int] = None
    ) -> Optional[DuckDBPyRelation]:
        if len(sources) <= 1 or any(table is None for table in sources.values()):
            return None

        while len(self.inputs) > len(sources):
            self.inputs.del_row(len(self.inputs) - 1)

        while len(self.inputs) < len(sources):
            self.inputs.add_row()

        for num, (label, source) in enumerate(sources.items()):
            if source is not None:
                logger.debug("JoinPlugin.execute_multi %d %s %s", num + 1, repr(label), source.alias)
                self.inputs[num].label = f"Input {num+1}: {label}"
                self.inputs[num].set_column_choices(
                    {name: duckdb_dtype_is_numeric(dt) for name, dt in zip(source.columns, source.dtypes)}
                )

        # XXX this isn't quite right for >2 tables where some
        # are required and some aren't.  I think what I need to
        # do is join all the required tables to each other and
        # then join the non-required ones on on top.
        tables = list(sources.values())
        required = [input_.required.value for input_ in self.inputs]

        select = join(
            Id.dotted(f"N_{num}", cname)
            for num, table in enumerate(tables)
            for cname in table.columns
            if not (
                (
                    self.inputs[num].drop
                    and (cname in (self.inputs[num].join_on, self.inputs[num].join_2, self.inputs[num].join_3))
                )
                or (
                    num != 0
                    and not self.inputs[0].drop
                    and (
                        cname == self.inputs[num].join_on == self.inputs[0].join_on
                        or cname == self.inputs[num].join_2 == self.inputs[0].join_2
                        or cname == self.inputs[num].join_3 == self.inputs[0].join_3
                    )
                )
            )
        )

        column_types = [dict(zip(table.columns, table.dtypes)) for table in tables]
        labels = list(sources.keys())

        def join_condition(num: int) -> Sql:
            """Compare the join column(s) of table 0 with those of table `num`, working out up front how to do
            that so there's no need to run the join to find out if it can be done."""
            conditions = []
            for attr in ("join_on", "join_2", "join_3"):
                left_param, right_param = getattr(self.inputs[0], attr), getattr(self.inputs[num], attr)
                if attr != "join_on" and not (left_param.is_not_none() and right_param.is_not_none()):
                    continue
                left_name, right_name = left_param.value, right_param.value
                left: Fragment = Id.dotted("N_0", left_name)
                right: Fragment = Id.dotted(f"N_{num}", right_name)
                if left_name in column_types[0] and right_name in column_types[num]:
                    try:
                        left, right = _comparable(
                            ddbc, left, column_types[0][left_name], right, column_types[num][right_name]
                        )
                    except JoinTypeError as exc:
                        raise JoinTypeError(
                            f"Can't join {labels[0]}.{left_name} to {labels[num]}.{right_name}: {exc}"
                        ) from exc
                conditions.append(Sql.template("{left} = {right}", left=left, right=right))
            return join(conditions, sep=" AND ")

        query = Sql.template(
            "SELECT {select} FROM {table} AS {alias}", select=select, table=Id(tables[0].alias), alias=Id("N_0")
        )
        for num, table in enumerate(tables[1:], 1):
            query = Sql.template(
                "{query} {how} JOIN {table} AS {alias} ON {on}",
                query=query,
                how=Raw(_join_how(required[0], required[num])),
                table=Id(table.alias),
                alias=Id(f"N_{num}"),
                on=join_condition(num),
            )

        if row_limit is not None:
            query = Sql.template("{query} LIMIT {limit}", query=query, limit=Lit(row_limit))

        logger.debug("JoinPlugin.execute_multi tables %s", [table.alias for table in tables])
        logger.debug("JoinPlugin.execute_multi query %s", query)

        return query.relation(ddbc)
