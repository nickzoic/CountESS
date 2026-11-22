import logging
import re
import string
from functools import lru_cache
from typing import Any, Iterable, Optional

from countess_variant_caller import TooManyVariationsException, classify_protein_variant, find_variant_string
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
from countess.core.plugins import DuckdbParallelTransformPlugin, DuckdbSqlPlugin
from countess.utils.sql import Fragment, Id, Lit, Sql, join

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


class VariantPlugin(DuckdbParallelTransformPlugin):
    """Turns a DNA sequence into a HGVS variant code"""

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
