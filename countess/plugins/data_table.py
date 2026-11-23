import logging
from typing import Optional

from duckdb import DuckDBPyConnection, DuckDBPyRelation

from countess import VERSION
from countess.core.parameters import ArrayParam, DataTypeChoiceParam, MultiParam, StringParam, TabularMultiParam
from countess.core.plugins import DuckdbInputPlugin
from countess.utils.sql import Id, Lit, Sql, join

logger = logging.getLogger(__name__)


class _ColumnsMultiParam(MultiParam):
    name = StringParam("Name")
    type = DataTypeChoiceParam("Type")


class DataTablePlugin(DuckdbInputPlugin):
    """DataTable"""

    name = "DataTable"
    description = "enter small amounts of data directly"
    link = "https://countess-project.github.io/CountESS/included-plugins/#data-table"
    version = VERSION
    num_inputs = 0

    columns = ArrayParam("Columns", _ColumnsMultiParam("Column"))
    rows = ArrayParam("Rows", TabularMultiParam("Row"))

    show_preview = False

    def fix_columns(self):
        old_rows = self.rows.params

        # XXX should deal with duplicate column names more generally
        for num, col in enumerate(self.columns):
            if not col.name:
                col.name = f"col_{num+1}"

        self.params["rows"] = self.rows = ArrayParam(
            "Rows",
            TabularMultiParam("Row", {str(col.name): col.type.get_parameter(str(col.name)) for col in self.columns}),
        )

        # fix rows to use the latest columns
        for old_row in old_rows:
            new_row = self.rows.add_row()
            for col in self.columns:
                try:
                    new_row[str(col.name)].value = old_row[str(col.name)].value
                except (KeyError, ValueError):
                    pass

    def set_parameter(self, key: str, *a, **k):
        if key.startswith("rows.0."):
            self.fix_columns()
        super().set_parameter(key, *a, **k)

    def set_json_data(self, value, base_dir="."):
        # The same applies as for set_parameter: the rows are made of whatever
        # the columns say they are, so the columns have to be loaded first and
        # the rows rebuilt to match, before any rows can be loaded.
        super().set_json_data({k: v for k, v in value.items() if k != "rows"}, base_dir)
        if "rows" in value:
            self.fix_columns()
            super().set_json_data({"rows": value["rows"]}, base_dir)

    def execute(
        self, ddbc: DuckDBPyConnection, source: None, row_limit: Optional[int] = None
    ) -> Optional[DuckDBPyRelation]:
        self.fix_columns()

        if len(self.rows) == 0:
            return None

        rows = join(
            Sql.template("({values})", values=join(Lit(val.value) for val in row.values())) for row in self.rows
        )
        columns = join(Id(col.name.value) for col in self.columns)
        sql = Sql.template("SELECT * FROM (VALUES {rows}) _({columns})", rows=rows, columns=columns)

        return sql.relation(ddbc)
