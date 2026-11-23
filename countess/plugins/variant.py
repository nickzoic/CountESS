import logging
import re
import secrets
import string
from functools import lru_cache
from typing import Any, Iterable, Optional

import duckdb
from countess_variant_caller import TooManyVariationsException, classify_protein_variant, find_variant_string
from duckdb import DuckDBPyConnection, DuckDBPyRelation
from duckdb.sqltypes import DuckDBPyType
from fqfa.constants.iupac.protein import AA_CODES

from countess import VERSION
from countess.core.parameters import (
    BooleanParam,
    ColumnChoiceParam,
    ColumnOrIntegerParam,
    ColumnOrStringParam,
    DictChoiceParam,
    FramedMultiParam,
    IntegerParam,
    StringCharacterSetParam,
    StringParam,
)
from countess.core.plugins import DuckdbSqlPlugin
from countess.utils.sql import Fn, Fragment, Id, Lit, Raw, Sql, join

logger = logging.getLogger(__name__)

REFERENCE_CHAR_SET = set(string.ascii_uppercase + string.digits + "_")

# XXX Should proabably support these other types as well but I don't
# know what I don't know ...
# XXX Supporting protein calls on mitochondrial (or other organisms)
# DNA will required expansion of the variant caller routine to handle
# different codon tables.  This opens up a can of worms of course.
# XXX There should probably also be a warning generated if you ask for a
# non-MT DNA call with an MT protein call or vice versa.

VARIANT_TYPE_CHOICES = {
    "g": "Genomic (g.)",
    # "o": "Circular Genomic",
    # "m": "Mitochondrial",
    "c": "Coding DNA (c.)",
    "n": "Non-Coding DNA (n.)",
}


class DnaVariantMultiParam(FramedMultiParam):
    prefix = StringCharacterSetParam("Prefix", "", character_set=REFERENCE_CHAR_SET)
    seq_type = DictChoiceParam("Type", choices=VARIANT_TYPE_CHOICES)
    offset = ColumnOrIntegerParam("Offset", 0)
    minus_strand = BooleanParam("Minus Strand", False)
    maxlen = IntegerParam("Max Variations", 10)
    output = StringParam("Output Column", "variant")


class ProteinVariantMultiParam(FramedMultiParam):
    prefix = StringCharacterSetParam("Prefix", "", character_set=REFERENCE_CHAR_SET)
    offset = ColumnOrIntegerParam("Offset", 0)
    # XXX different codon tables go here
    maxlen = IntegerParam("Max Variations", 10)
    output = StringParam("Output Column", "protein")


class VariantPlugin(DuckdbSqlPlugin):
    """Turns a DNA sequence into a HGVS variant code

    Working out a variant is expensive (compared to everything else in a pipeline) and has to be done in
    Python, and a typical experiment has the same sequence over and over again, so rather than calling
    `transform` for every row this works out each distinct combination of sequence, reference and offsets
    only once, in a UDF, and then joins the results back to the rows they belong to.  The UDF only ever sees the
    columns it needs, not the whole row.
    """

    name = "Variant Caller"
    description = "Turns a DNA sequence into a HGVS variant code"
    version = VERSION
    link = "https://countess-project.github.io/CountESS/included-plugins/#variant-caller"

    column = ColumnChoiceParam("Input Column", "sequence")
    reference = ColumnOrStringParam("Reference Sequence")

    variant = DnaVariantMultiParam("DNA Variant")
    protein = ProteinVariantMultiParam("Protein Variant")
    classify = StringParam("Classification Output Column", "class")

    drop = BooleanParam("Drop unmatched rows", False)
    drop_columns = BooleanParam("Drop Sequence / Reference Columns", False)

    function_name: Optional[str] = None

    @lru_cache(maxsize=10000)
    def find_variant_string_cached(self, *a, **k):
        return find_variant_string(*a, **k)

    def add_fields(self):
        return {
            self.variant.output.value: str,
            self.protein.output.value: str,
            self.classify.value: str,
        }

    def remove_fields(self, field_names: list[str]) -> list[Optional[str]]:
        if self.drop_columns:
            return [self.column.value, self.reference.get_column_name()]
        else:
            return []

    def transform(self, data: dict[str, Any]) -> Optional[dict[str, Any]]:
        """Call the variants for a single row, given as a dict of column name to value.  Returns that dict
        with the output columns added, or None if there's nothing to call."""
        sequence = data[str(self.column)]
        reference = self.reference.get_value_from_dict(data)
        if not sequence or not reference:
            return None

        # r: dict[str, Any] = data
        if self.variant.output:
            data[self.variant.output.value] = None
            try:
                prefix = self.variant.prefix + ":" if self.variant.prefix else ""
                data[self.variant.output.value] = find_variant_string(
                    f"{prefix}{self.variant.seq_type.get_choice()}.",
                    reference,
                    sequence,
                    max_mutations=self.variant.maxlen.value,
                    offset=int(self.variant.offset.get_value_from_dict(data) or 0),
                    minus_strand=self.variant.minus_strand.value,
                )
            except TooManyVariationsException:
                pass
            except (ValueError, TypeError, KeyError, IndexError) as exc:
                logger.warning("Exception", exc_info=exc)

        if self.protein.output or self.classify:
            if self.protein.output:
                data[self.protein.output.value] = None
            if self.classify:
                data[self.classify.value] = None
            try:
                protein_variant = find_variant_string(
                    "p.",
                    reference,
                    sequence,
                    max_mutations=self.protein.maxlen.value,
                    offset=int(self.protein.offset.get_value_from_dict(data) or 0),
                )

                if self.protein.output:
                    prefix = self.protein.prefix + ":" if self.protein.prefix else ""
                    data[self.protein.output.value] = prefix + protein_variant
                if self.classify:
                    data[self.classify.value] = classify_protein_variant(protein_variant)

            except TooManyVariationsException:
                pass
            except (ValueError, TypeError, KeyError, IndexError) as exc:
                logger.warning("Exception", exc_info=exc)

        if self.drop_columns:
            del data[self.column.value]
            reference_column_name = self.reference.get_column_name()
            if reference_column_name:
                del data[reference_column_name]

        return data

    def output_fields(self) -> list[str]:
        """Names of the columns which this plugin calculates (some can be turned off by leaving the name blank)"""
        return [name for name in self.add_fields() if name]

    def call(self, sequence: str, reference: str, dna_offset: int, protein_offset: int) -> Optional[dict[str, Any]]:
        """The UDF: `transform` for one distinct set of inputs, returning only the calculated columns.
        The offset and reference are only looked at if they come from a column, otherwise `transform`
        uses the constant from the plugin's parameters."""

        data: dict[str, Any] = {self.column.value: sequence}
        if reference_column := self.reference.get_column_name():
            data[reference_column] = reference
        if dna_offset_column := self.variant.offset.get_column_name():
            data[dna_offset_column] = dna_offset
        if protein_offset_column := self.protein.offset.get_column_name():
            data[protein_offset_column] = protein_offset

        try:
            result = self.transform(data)
        except RuntimeError as exc:
            logger.warning(str(exc))
            return None

        return {name: result.get(name) for name in self.output_fields()} if result is not None else None

    def execute(
        self, ddbc: DuckDBPyConnection, source: DuckDBPyRelation, row_limit: Optional[int] = None
    ) -> Optional[DuckDBPyRelation]:
        outputs = self.output_fields()
        if outputs:
            self.function_name = "f_" + secrets.token_hex(16)
            logger.debug("VariantPlugin creating function %s for plugin %s", self.function_name, self.name)
            ddbc.create_function(  # type: ignore[call-overload]
                self.function_name,
                self.call,
                parameters=[
                    DuckDBPyType("VARCHAR"),
                    DuckDBPyType("VARCHAR"),
                    DuckDBPyType("BIGINT"),
                    DuckDBPyType("BIGINT"),
                ],
                return_type=duckdb.struct_type({name: DuckDBPyType("VARCHAR") for name in outputs}),
            )
        return super().execute(ddbc, source, row_limit)

    def sql(self, table_name: str, columns: Iterable[str]) -> Optional[Sql]:
        columns = list(columns)
        outputs = self.output_fields()
        removed = {c for c in self.remove_fields(columns) if c}
        kept_columns = [c for c in columns if c not in removed]
        table = Id(table_name)

        if not outputs:
            kept: Fragment = join(Id(c) for c in kept_columns) or Raw("*")
            return Sql.template("SELECT {columns} FROM {table}", columns=kept, table=table)

        # The inputs to the UDF.  Anything which is a constant (the reference, perhaps the offsets) isn't
        # part of the key, as it's the same for every row, and `call` doesn't look at it.
        sequence_expr = Sql.template("CAST({column} AS VARCHAR)", column=Id.dotted("S", self.column.value))
        keys: dict[str, Fragment] = {"k_sequence": sequence_expr}
        args: dict[str, Fragment] = {
            "k_sequence": Id("k_sequence"),
            "k_reference": Lit(""),
            "k_dna_offset": Lit(0),
            "k_protein_offset": Lit(0),
        }

        if reference_column := self.reference.get_column_name():
            keys["k_reference"] = Sql.template("CAST({column} AS VARCHAR)", column=Id.dotted("S", reference_column))
        if dna_offset_column := self.variant.offset.get_column_name():
            keys["k_dna_offset"] = Sql.template(
                "COALESCE(CAST({column} AS BIGINT), 0)", column=Id.dotted("S", dna_offset_column)
            )
        if protein_offset_column := self.protein.offset.get_column_name():
            keys["k_protein_offset"] = Sql.template(
                "COALESCE(CAST({column} AS BIGINT), 0)", column=Id.dotted("S", protein_offset_column)
            )
        for key in keys:
            args[key] = Id(key)

        # There's nothing to call if there's no sequence or no reference (see `transform`): those rows
        # don't match anything in the join, so the whole row comes out NULL, as it always has.
        nothing_to_call: list[Fragment] = [Sql.template("({key} IS NOT NULL AND {key} != '')", key=keys["k_sequence"])]
        if "k_reference" in keys:
            nothing_to_call.append(Sql.template("({key} IS NOT NULL AND {key} != '')", key=keys["k_reference"]))
        elif not self.reference.get_value_from_dict({}):
            nothing_to_call.append(Raw("FALSE"))

        key_select = join(Sql.template("{expr} AS {key}", expr=expr, key=Id(key)) for key, expr in keys.items())
        join_on = join(
            (Sql.template("{expr} = {column}", expr=expr, column=Id.dotted("C", key)) for key, expr in keys.items()),
            sep=" AND ",
        )

        # Output columns replace input columns of the same name where there are any, otherwise they
        # go at the end.
        called: dict[str, Fragment] = {
            name: Sql.template("struct_extract({result}, {name})", result=Id.dotted("C", "_result"), name=Lit(name))
            for name in outputs
        }

        def select_column(column: str) -> Fragment:
            if column in called:
                return Sql.template("{expr} AS {name}", expr=called[column], name=Id(column))
            # (no match in the join means there was nothing to call, which blanks the whole row)
            return Sql.template(
                "CASE WHEN {key} IS NOT NULL THEN {value} END AS {name}",
                key=Id.dotted("C", "k_sequence"),
                value=Id.dotted("S", column),
                name=Id(column),
            )

        select_list = [select_column(c) for c in kept_columns]
        select_list += [
            Sql.template("{expr} AS {name}", expr=expr, name=Id(name))
            for name, expr in called.items()
            if name not in kept_columns
        ]

        return Sql.template(
            """
            WITH _source AS MATERIALIZED (
                SELECT * FROM {table}
            ), _keys AS (
                SELECT DISTINCT {key_select} FROM _source AS S WHERE {nothing_to_call}
            ), _called AS (
                SELECT *, {function}({udf_args}) AS _result FROM _keys
            )
            SELECT {select_list}
            FROM _source AS S LEFT JOIN _called AS C ON {join_on}
            """,
            table=table,
            key_select=key_select,
            nothing_to_call=join(nothing_to_call, sep=" AND "),
            function=Fn(self.function_name),
            udf_args=join(args[k] for k in ("k_sequence", "k_reference", "k_dna_offset", "k_protein_offset")),
            select_list=join(select_list),
            join_on=join_on,
        )


class VariantClassifier(DuckdbSqlPlugin):
    name = "Protein Variant Classifier"
    description = "Classifies protein variants into simple types"
    version = VERSION
    link = "https://countess-project.github.io/CountESS/included-plugins/#variant-classifier"

    variant_col = ColumnChoiceParam("Protein variant Column", "variant")

    def sql(self, table_name: str, columns: Iterable[str]) -> Optional[Sql]:
        variant = Id(self.variant_col.value)
        output = Id(self.variant_col + "_type")

        # the nested query stops any of the pattern matching columns
        # from colliding with source table columns, plus hopefully
        # duckdb can use the 'group by z' to only match the regex
        # once for each distinct variant string.  Then the cases
        # in the outer select use the parts of the regex match to
        # classify the variant.

        hgvs_aa_re = "(?:" + "|".join(v for v in AA_CODES.values() if v != "Ter") + ")"
        short_aa_re = "[" + "".join(k for k in AA_CODES if k != "*") + "]"

        classifier_re = re.sub(
            r"\s+",
            "",
            rf"""
            ^(_?[Ww][Tt]|p.=)$|
            ^p.{hgvs_aa_re}\d+({hgvs_aa_re}|=|del|dup|Ter)$|
            ^p.{hgvs_aa_re}\d+_{hgvs_aa_re}\d+(del|dup|ins{hgvs_aa_re}+)$|
            ^({short_aa_re})\d+({short_aa_re}|[=*X-])$
        """,
        )

        return Sql.template(
            r"""
            select S.*, case
               when T.wildtype != '' then 'W'
               when T.hgvs_1 = '=' then 'S'
               when T.hgvs_1 = 'del' then 'D'
               when T.hgvs_1 = 'dup' then 'I'
               when T.hgvs_1 = 'Ter' then 'N'
               when T.hgvs_1 != '' then 'M'
               when T.hgvs_2 = 'del' then 'D'
               when T.hgvs_2 != '' then 'I'
               when T.short_2 = '=' then 'S'
               when T.short_2 = '*' then 'N'
               when T.short_2 = 'X' then 'N'
               when T.short_2 = '-' then 'D'
               when T.short_2 != '' then case
                   when T.short_1 = T.short_2 then 'S'
                   else 'M'
               end
               else warning(concat({plugin_label},
                   'unclassifiable variant: "', z, '"'), '?')
               end as {output}
            from {table} S join (
                select {variant} as z, unnest(regexp_extract(
                    {variant},
                    {classifier_re},
                    ['wildtype','hgvs_1','hgvs_2','short_1', 'short_2']
                ))
                from {table}
                group by z
            ) T on {source_variant} = T.z
            """,
            plugin_label=Lit(self.name + ": "),
            output=output,
            table=Id(table_name),
            variant=variant,
            classifier_re=Lit(classifier_re),
            source_variant=Id.dotted("S", self.variant_col.value),
        )


def _translate_aa(expr: Fragment, expr1: Optional[Fragment] = None) -> Sql:
    """A CASE which turns the one letter code in `expr` into the three letter one (or, if there's an `expr1`, which
    gives = if they are the same)"""
    # This looks ludicrous but it pushes all the work down into SQL so that
    # duckdb can run it without translating rows into Python etc.
    whens = [
        Sql.template("WHEN {expr} = {code} THEN {name}", expr=expr, code=Lit(k), name=Lit(v))
        for k, v in list(AA_CODES.items()) + [("X", "Ter"), ("-", "del")]
    ]
    if expr1 is not None:
        whens.insert(0, Sql.template("WHEN {expr} = {expr1} THEN '='", expr=expr, expr1=expr1))
    return Sql.template("CASE {whens} ELSE NULL END", whens=join(whens, sep=" "))


class VariantConverter(DuckdbSqlPlugin):
    name = "Variant Converter"
    description = "Convert various forms of variant strings to HGVS"
    version = VERSION

    variant_col = ColumnChoiceParam("Variant Column", "variant")
    variant_out = StringParam("Variant Output Column", "variant")
    prefix_aa = StringParam("Variant AA Prefix", "")
    prefix_nt = StringParam("Variant NT Prefix", "")
    offset = IntegerParam("Variant Location Offset", 0)

    def sql(self, table_name: str, columns: Iterable[str]) -> Optional[Sql]:
        variant = Id(self.variant_col.value)
        output = Id(self.variant_out.value)
        offset = Lit(self.offset.value)
        prefix_aa = Lit((self.prefix_aa.value + ":" if self.prefix_aa else "") + "p.")
        prefix_nt = Lit((self.prefix_nt.value + ":" if self.prefix_nt else "") + "c.")

        # (the first and last letter, the digits between them, and everything after syn_)
        first = Sql.template("{variant}[1]", variant=variant)
        last = Sql.template("{variant}[-1]", variant=variant)
        digits = Sql.template("{variant}[2:-2]", variant=variant)
        after_syn = Sql.template("{variant}[5:]", variant=variant)

        # XXX handle other short forms like _synNNNX>Y or whatever
        converted = Sql.template(
            r"""CASE
                WHEN regexp_matches({variant}, 'wt', 'i') THEN {prefix_aa} || '='
                WHEN regexp_matches({variant}, '[A-Z]\d+[*A-Z-]')
                    THEN {prefix_aa} || {first_aa} || ({digits}::INTEGER + {offset})::TEXT || {last_aa}
                WHEN regexp_matches({variant}, 'syn_\d+[A-Z]>[A-Z]')
                    THEN {prefix_nt} || {after_syn}
                ELSE NULL
            END AS {output}""",
            variant=variant,
            prefix_aa=prefix_aa,
            first_aa=_translate_aa(first),
            digits=digits,
            offset=offset,
            last_aa=_translate_aa(last, first),
            prefix_nt=prefix_nt,
            after_syn=after_syn,
            output=output,
        )

        fields = join([*(Id(c) for c in columns if c != self.variant_out.value), converted])
        return Sql.template("SELECT {fields} FROM {table}", fields=fields, table=Id(table_name))
