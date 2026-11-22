import logging
from typing import Dict, Mapping, Optional

import duckdb
from duckdb import DuckDBPyConnection, DuckDBPyRelation

from countess import VERSION
from countess.core.parameters import ArrayParam, BooleanParam, ColumnChoiceParam, ColumnOrNoneChoiceParam, MultiParam
from countess.core.plugins import DuckdbPlugin
from countess.utils.duckdb import duckdb_dtype_is_numeric
from countess.utils.sql import Id, Lit, Raw, Sql, join

logger = logging.getLogger(__name__)


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
            if source:
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

        query = Sql.template(
            "SELECT {select} FROM {table} AS {alias}", select=select, table=Id(tables[0].alias), alias=Id("N_0")
        )
        for num, table in enumerate(tables[1:], 1):
            # the first table's join columns are matched with those of this one, as far as there are two of them
            conditions = [
                Sql.template(
                    "{left} = {right}",
                    left=Id.dotted("N_0", self.inputs[0].join_on.value),
                    right=Id.dotted(f"N_{num}", self.inputs[num].join_on.value),
                )
            ]
            for key in ("join_2", "join_3"):
                first, this = getattr(self.inputs[0], key), getattr(self.inputs[num], key)
                if first.is_not_none() and this.is_not_none():
                    conditions.append(
                        Sql.template(
                            "{left} = {right}",
                            left=Id.dotted("N_0", first.value),
                            right=Id.dotted(f"N_{num}", this.value),
                        )
                    )

            query = Sql.template(
                "{query} {how} JOIN {table} AS {alias} ON {on}",
                query=query,
                how=Raw(_join_how(required[0], required[num])),
                table=Id(table.alias),
                alias=Id(f"N_{num}"),
                on=join(conditions, sep=" AND "),
            )

        if row_limit is not None:
            query = Sql.template("{query} LIMIT {limit}", query=query, limit=Lit(row_limit))

        logger.debug("JoinPlugin.execute_multi tables[0] %s %d", tables[0].alias, len(tables[0]))
        logger.debug("JoinPlugin.execute_multi tables[1] %s %d", tables[1].alias, len(tables[1]))
        logger.debug("JoinPlugin.execute_multi query %s", query)

        try:
            rel = query.relation(ddbc)
            logger.debug("JoinPlugin.execute_multi output %d", len(rel))
            return rel
        except duckdb.ConversionException as exc:
            logger.info(exc)
            return None
