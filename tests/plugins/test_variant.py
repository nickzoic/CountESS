import duckdb
import numpy as np
import pandas as pd
import pytest

from countess.plugins.variant import VariantClassifier, VariantConverter, VariantPlugin
from countess.utils.duckdb import duckdb_escape_literal


def test_variant_ref_value():
    plugin = VariantPlugin()
    plugin.set_parameter("column", "seq")
    plugin.set_parameter("reference", "AGAAGTAGAGG")
    plugin.set_parameter("variant.seq_type", "g")
    plugin.set_parameter("variant.output", "out")
    plugin.set_column_choices({"seq": False})

    assert plugin.transform({"seq": "TGAAGTAGAGG"})["out"] == "g.1A>T"
    assert plugin.transform({"seq": "AGAAGTTGTGG"})["out"] == "g.[7A>T;9A>T]"
    assert plugin.transform({"seq": "ATAAGAAGAGG"})["out"] == "g.[2G>T;6T>A]"


def test_variant_ref_column():
    plugin = VariantPlugin()
    plugin.set_parameter("column", "seq")
    plugin.set_parameter("reference", "— ref")
    plugin.set_parameter("variant.seq_type", "g")
    plugin.set_parameter("variant.output", "out")
    plugin.set_column_choices({"seq": False, "ref": False})

    assert plugin.transform({"ref": "TACACACAG", "seq": "TACAGACAG"})["out"] == "g.5C>G"
    assert plugin.transform({"ref": "ATGGTTGGTTC", "seq": "ATGGTTGGTGGTTCG"})["out"] == "g.[7_9dup;11_12insG]"


def test_variant_ref_offset():
    plugin = VariantPlugin()
    plugin.set_parameter("column", "seq")
    plugin.set_parameter("reference", "AGAAGTAGAGG")
    plugin.set_parameter("variant.offset", "— offs")
    plugin.set_parameter("variant.seq_type", "g")
    plugin.set_parameter("variant.output", "out")
    plugin.set_column_choices({"seq": False, "offs": True})

    assert plugin.transform({"seq": "TGAAGTAGAGG", "offs": "0"})["out"] == "g.1A>T"
    assert plugin.transform({"seq": "AGAAGTTGTGG", "offs": "10"})["out"] == "g.[17A>T;19A>T]"
    assert plugin.transform({"seq": "ATAAGAAGAGG", "offs": "100"})["out"] == "g.[102G>T;106T>A]"


def test_variant_ref_offset_minus():
    """check that the reverse-complement behaviour works on the minus strand."""
    # genes on the minus strand are reverse-complemented, so what we're actually
    # comparing is the reverse-complemented sequences:
    #
    #      00000000011
    # num  12345678901
    # ref  CCTCTACTTCT
    # seq1 CCTCTACTTCA => 11T>A
    # seq2 CCACAACTTCT => 3T>A;5T>A
    # seq3 CCTCTTCTTAT => 6A>T;10C>A
    #
    # plus the offset

    plugin = VariantPlugin()
    plugin.set_parameter("column", "seq")
    plugin.set_parameter("reference", "AGAAGTAGAGG")
    plugin.set_parameter("variant.offset", "1000")
    plugin.set_parameter("variant.seq_type", "g")
    plugin.set_parameter("variant.minus_strand", True)
    plugin.set_parameter("variant.output", "out")
    plugin.set_column_choices({"seq": False, "offs": True})

    assert plugin.transform({"seq": "TGAAGTAGAGG"})["out"] == "g.1011T>A"
    assert plugin.transform({"seq": "AGAAGTTGTGG"})["out"] == "g.[1003T>A;1005T>A]"
    assert plugin.transform({"seq": "ATAAGAAGAGG"})["out"] == "g.[1006A>T;1010C>A]"


def test_variant_too_many():
    plugin = VariantPlugin()
    plugin.set_parameter("column", "seq")
    plugin.set_parameter("reference", "AGAAGTAGAGG")
    plugin.set_parameter("variant.seq_type", "g")
    plugin.set_parameter("variant.output", "out")
    plugin.set_parameter("variant.maxlen", 2)
    plugin.set_column_choices({"seq": False, "offs": True})

    assert plugin.transform({"seq": "TGAAGTAGAGG"})["out"] == "g.1A>T"
    assert plugin.transform({"seq": "AGAAGTTGTGG"})["out"] == "g.[7A>T;9A>T]"
    assert plugin.transform({"seq": "ATAAGAAGACG"})["out"] is None
    assert plugin.transform({"seq": "AGAATTAGAGG"})["out"] == "g.5G>T"


@pytest.mark.parametrize(
    "variant,expected,warnings_expected",
    [
        ("WT", "W", 0),
        ("_WT", "W", 0),
        ("A107P", "M", 0),
        ("A107A", "S", 0),
        ("A107=", "S", 0),
        ("A107X", "N", 0),
        ("A107*", "N", 0),
        ("A107Z", "?", 1),  # there's no such short AA code
        ("A107AA", "?", 1),  # not a valid substitution
        ("A107-", "D", 0),
        ("p.=", "W", 0),
        ("p.Ala107His", "M", 0),
        ("p.Ala107=", "S", 0),
        ("p.Ala107del", "D", 0),
        ("p.Ala107insAla", "?", 1),  # not valid, need both ends
        ("p.Ala107_His108insGluGln", "I", 0),
        ("p.Ala107dup", "I", 0),
        ("p.Ala107_His109del", "D", 0),
        ("p.Ala3Foo", "?", 1),  # no such long AA code
    ],
)
def test_classifier(variant, expected, warnings_expected):
    warnings = []

    def mock_warning(message: str, value: str) -> str:
        warnings.append(message)
        return value

    plugin = VariantClassifier()
    plugin.set_parameter("variant_col", "variant")
    ddbc = duckdb.connect()
    ddbc.create_function("warning", mock_warning)

    variant_literal = duckdb_escape_literal(variant)
    ddbc.sql(f"create table test as select {variant_literal} as variant")
    plugin.execute(ddbc, ddbc.table("test"), None).to_view("test2")
    value = ddbc.sql("select variant_type from test2").fetchone()[0]
    assert value == expected
    assert len(warnings) == warnings_expected


# ---- more of the variant caller

# a reference of  Met Ala Lys Stop  to make protein calls against
REFERENCE = "ATGGCTAAATAG"


def variant_plugin(**params):
    plugin = VariantPlugin()
    plugin.set_parameter("column", "seq")
    plugin.set_parameter("reference", REFERENCE)
    for key, value in params.items():
        plugin.set_parameter(key, value)
    plugin.set_column_choices({"seq": False, "ref": False})
    return plugin


@pytest.mark.parametrize(
    "sequence,dna,protein,classification",
    [
        ("ATGGCTAAATAG", "g.=", "p.=", "W"),  # nothing different
        ("ATGGTTAAATAG", "g.5C>T", "p.Ala2Val", "M"),  # a missense change
        ("ATGGCCAAATAG", "g.6T>C", "p.Ala2=", "S"),  # a different codon for the same amino acid
        ("ATGGCTTAATAG", "g.7A>T", "p.Lys3Ter", "N"),  # a stop codon
        ("ATGGCTAAA", "g.10_12del", "p.Ter4del", "D"),  # (it's not got a stop codon any more)
        ("atggttaaatag", "g.5C>T", "p.Ala2Val", "M"),  # (the case doesn't matter)
    ],
)
def test_dna_protein_and_classification(sequence, dna, protein, classification):
    result = variant_plugin().transform({"seq": sequence})
    assert (result["variant"], result["protein"], result["class"]) == (dna, protein, classification)
    assert result["seq"] == sequence  # (the input is kept)


def test_prefixes_and_output_names():
    plugin = variant_plugin(
        **{
            "variant.prefix": "NC_1",
            "variant.output": "hgvs_dna",
            "protein.prefix": "ENSP2",
            "protein.output": "hgvs_protein",
            "classify": "kind",
        }
    )
    result = plugin.transform({"seq": "ATGGTTAAATAG"})
    assert result == {
        "seq": "ATGGTTAAATAG",
        "hgvs_dna": "NC_1:g.5C>T",
        "hgvs_protein": "ENSP2:p.Ala2Val",
        "kind": "M",
    }
    assert plugin.add_fields() == {"hgvs_dna": str, "hgvs_protein": str, "kind": str}


def test_dna_variant_types():
    for seq_type, expected in (("g", "g.5C>T"), ("c", "c.5C>T"), ("n", "n.5C>T")):
        assert (
            variant_plugin(**{"variant.seq_type": seq_type}).transform({"seq": "ATGGTTAAATAG"})["variant"] == expected
        )


@pytest.mark.parametrize("sequence", ["", None])
def test_no_sequence_gives_nothing(sequence):
    assert variant_plugin().transform({"seq": sequence}) is None


def test_no_reference_gives_nothing():
    plugin = variant_plugin(reference="— ref")  # (a reference taken from a column, which is empty)
    assert plugin.transform({"seq": "ATGGCTAAATAG", "ref": ""}) is None
    assert plugin.transform({"seq": "ATGGCTAAATAG", "ref": None}) is None
    assert plugin.transform({"seq": "ATGGTTAAATAG", "ref": REFERENCE})["variant"] == "g.5C>T"


def test_a_sequence_which_cant_be_compared_gives_nulls_and_a_warning(caplog):
    with caplog.at_level("WARNING"):
        result = variant_plugin().transform({"seq": "ATGNCTAAATAG"})  # (N isn't a base)
    assert (result["variant"], result["protein"], result["class"]) == (None, None, None)
    assert "Invalid variant sequence" in caplog.text


def test_too_many_protein_changes_gives_no_protein_but_still_the_dna():
    reference = "ATGGCTAAAGGTTAG"  # Met Ala Lys Gly Stop
    sequence = "ATGGTTAAAGATTAG"  # two changes, Ala2Val and Gly4Asp, with an unchanged codon between them
    everything = variant_plugin(reference=reference).transform({"seq": sequence})
    assert everything["protein"] == "p.[Ala2Val;Gly4Asp]"
    assert everything["class"] == "?"  # (a change in more than one place isn't classified)

    # (changes in neighbouring codons are one change, so wouldn't be too many)
    limited = variant_plugin(reference=reference, **{"protein.maxlen": 1}).transform({"seq": sequence})
    assert limited["variant"] == "g.[5C>T;11G>A]"
    assert (limited["protein"], limited["class"]) == (None, None)


def test_sequence_and_reference_columns_can_be_dropped():
    plugin = variant_plugin(reference="— ref", drop_columns=True)
    result = plugin.transform({"seq": "ATGGTTAAATAG", "ref": REFERENCE, "other": 7})
    assert result == {"other": 7, "variant": "g.5C>T", "protein": "p.Ala2Val", "class": "M"}
    assert plugin.remove_fields(["seq", "ref", "other"]) == ["seq", "ref"]

    keeping = variant_plugin(reference="— ref")
    assert keeping.remove_fields(["seq", "ref", "other"]) == []
    assert "seq" in keeping.transform({"seq": "ATGGTTAAATAG", "ref": REFERENCE})


def test_a_fixed_reference_has_no_column_to_drop():
    result = variant_plugin(drop_columns=True).transform({"seq": "ATGGTTAAATAG"})
    assert "seq" not in result


def test_in_a_query(ddbc, make_source):
    source = make_source(
        "select * from (values ('ATGGTTAAATAG', 1), ('ATGGCTAAATAG', 2), ('', 3), ('ATGGCTTAATAG', 4)) t(seq, n)"
    )
    plugin = variant_plugin()
    plugin.prepare(ddbc, source)
    output = plugin.execute(ddbc, source)
    assert output.columns == ["seq", "n", "variant", "protein", "class"]
    assert sorted(output.fetchall(), key=repr) == sorted(
        [
            ("ATGGTTAAATAG", 1, "g.5C>T", "p.Ala2Val", "M"),
            ("ATGGCTAAATAG", 2, "g.=", "p.=", "W"),
            (None, None, None, None, None),  # (nothing to call: a whole row of nulls, the number too)
            ("ATGGCTTAATAG", 4, "g.7A>T", "p.Lys3Ter", "N"),
        ],
        key=repr,
    )


def test_in_a_query_with_columns_dropped(ddbc, make_source):
    source = make_source("select * from (values ('ATGGTTAAATAG', 1)) t(seq, n)")
    plugin = variant_plugin(drop_columns=True)
    plugin.prepare(ddbc, source)
    output = plugin.execute(ddbc, source)
    assert output.columns == ["n", "variant", "protein", "class"]


def test_each_distinct_input_is_only_called_once(ddbc, make_source, monkeypatch):
    """The whole point of calling through a join: a sequence which turns up in many rows is only worked out once."""
    source = make_source(
        "select * from (values ('ATGGTTAAATAG', 1), ('ATGGCTAAATAG', 2), ('ATGGTTAAATAG', 3), ('ATGGTTAAATAG', 4),"
        " ('ATGGCTAAATAG', 5), ('', 6)) t(seq, n)"
    )
    plugin = variant_plugin()
    calls = []
    real_call = plugin.call

    def spy(sequence, reference, dna_offset, protein_offset):  # (duckdb needs to see the real parameters)
        calls.append((sequence, reference, dna_offset, protein_offset))
        return real_call(sequence, reference, dna_offset, protein_offset)

    monkeypatch.setattr(plugin, "call", spy)
    plugin.prepare(ddbc, source)
    rows = plugin.execute(ddbc, source).fetchall()

    assert len(rows) == 6
    assert sorted(a[0] for a in calls) == ["ATGGCTAAATAG", "ATGGTTAAATAG"]  # (and not the empty one)
    assert sorted(r[1:] for r in rows if r[0] == "ATGGTTAAATAG") == [(n, "g.5C>T", "p.Ala2Val", "M") for n in (1, 3, 4)]


def test_an_upstream_which_is_a_view_is_only_evaluated_once(ddbc):
    """The query needs the source twice (to find the distinct inputs, and to join the results back to), which
    must not mean doing everything upstream of it twice."""
    evaluations = []

    def upstream(seq: str) -> str:
        evaluations.append(seq)
        return seq

    ddbc.create_function("upstream", upstream)
    ddbc.sql("create table base as select 'ATGGTTAAATAG' as seq from range(100)")
    ddbc.sql("create view lazy as select upstream(seq) as seq from base")
    source = ddbc.table("lazy")
    plugin = variant_plugin()
    plugin.prepare(ddbc, source)
    assert len(plugin.execute(ddbc, source).fetchall()) == 100
    assert len(evaluations) == 100


def test_reference_and_offsets_from_columns_in_a_query(ddbc, make_source):
    """Rows with the same sequence but a different reference or offset are different calls, and each row gets
    exactly what calling `transform` on it would have given."""
    values = [
        ("ATGGTTAAATAG", REFERENCE, 0, 0),
        ("ATGGTTAAATAG", REFERENCE, 99, 33),  # (same sequence, different offsets)
        ("ATGGTTAAATAG", "ATGGTTAAATAG", 0, 0),  # (same sequence, different reference)
        ("ATGGCTTAATAG", REFERENCE, 3, 1),
        ("ATGGCTTAATAG", REFERENCE, None, None),  # (a null offset is no offset)
        ("ATGGCTAAATAG", "", 0, 0),  # (no reference)
    ]
    source = make_source(
        "select * from (values "
        + ", ".join(f"({', '.join(duckdb_escape_literal(v) for v in row)}, {n})" for n, row in enumerate(values))
        + ") t(seq, ref, doff, poff, n)"
    )
    params = {"reference": "— ref", "variant.offset": "— doff", "protein.offset": "— poff"}
    plugin = variant_plugin(**params)
    plugin.set_column_choices({"seq": False, "ref": False, "doff": True, "poff": True, "n": True})
    plugin.prepare(ddbc, source)
    output = plugin.execute(ddbc, source)
    assert output.columns == ["seq", "ref", "doff", "poff", "n", "variant", "protein", "class"]

    expected = []
    reference_plugin = variant_plugin(**params)
    reference_plugin.set_column_choices({"seq": False, "ref": False, "doff": True, "poff": True, "n": True})
    for n, (seq, ref, doff, poff) in enumerate(values):
        result = reference_plugin.transform({"seq": seq, "ref": ref, "doff": doff, "poff": poff, "n": n})
        expected.append((None,) * 8 if result is None else tuple(result.values()))
    assert sorted(output.fetchall(), key=repr) == sorted(expected, key=repr)


def test_an_output_with_the_same_name_as_an_input_replaces_it_in_place(ddbc, make_source):
    source = make_source("select * from (values ('ATGGTTAAATAG', 'x', 1)) t(seq, protein, n)")
    plugin = variant_plugin()
    plugin.prepare(ddbc, source)
    output = plugin.execute(ddbc, source)
    assert output.columns == ["seq", "protein", "n", "variant", "class"]
    assert output.fetchall() == [("ATGGTTAAATAG", "p.Ala2Val", 1, "g.5C>T", "M")]


@pytest.mark.xfail(
    strict=True,
    reason='the "Drop unmatched rows" option is never looked at: switching it on makes no difference to what comes out',
)
def test_unmatched_rows_can_be_dropped(ddbc, make_source):
    source = make_source("select * from (values ('ATGGTTAAATAG'), ('ATGNCTAAATAG')) t(seq)")  # (one can't be matched)
    outputs = []
    for drop in (False, True):
        plugin = variant_plugin(drop=drop)
        plugin.prepare(ddbc, source)
        outputs.append(sorted(plugin.execute(ddbc, source).fetchall(), key=repr))
    assert outputs[0] != outputs[1]


def test_the_cached_finder_agrees_with_the_real_one():
    """(Nothing calls this yet: it's here in case it's put to use)"""
    from countess_variant_caller import find_variant_string  # pylint: disable=import-outside-toplevel

    plugin = variant_plugin()
    for _ in range(2):  # (the second time it's from the cache)
        assert plugin.find_variant_string_cached("g.", REFERENCE, "ATGGTTAAATAG") == find_variant_string(
            "g.", REFERENCE, "ATGGTTAAATAG"
        )


# ---- the variant converter

CONVERSIONS = [
    ("wt", "p.="),
    ("WT", "p.="),
    ("A12G", "p.Ala12Gly"),
    ("A12A", "p.Ala12="),  # (the same amino acid)
    ("A12*", "p.Ala12Ter"),
    ("A12-", "p.Ala12del"),
    ("M1X", "p.Met1Ter"),
    ("syn_12A>G", "c.12A>G"),
    ("garbage", None),
    (None, None),
]


def convert(ddbc, make_source, **params):
    values = ", ".join("(NULL)" if v is None else f"('{v}')" for v, _ in CONVERSIONS)
    source = make_source(f"select * from (values {values}) t(variant)")
    plugin = VariantConverter()
    plugin.prepare(ddbc, source)
    for key, value in params.items():
        plugin.set_parameter(key, value)
    return plugin.execute(ddbc, source)


def test_variant_converter(ddbc, make_source):
    output = convert(ddbc, make_source)
    assert output.columns == ["variant"]  # (the converted ones replace the originals)
    assert sorted(output.fetchall(), key=repr) == sorted([(e,) for _, e in CONVERSIONS], key=repr)


def test_each_variant_converts_to_the_right_thing(ddbc, make_source):
    for variant, expected in CONVERSIONS:
        source = make_source("select " + ("NULL::varchar" if variant is None else f"'{variant}'") + " as variant")
        plugin = VariantConverter()
        plugin.prepare(ddbc, source)
        assert plugin.execute(ddbc, source).fetchall() == [(expected,)], variant


def test_variant_converter_offset_and_prefixes(ddbc, make_source):
    output = convert(ddbc, make_source, offset=100, prefix_aa="ENST1", prefix_nt="NM_1")
    expected = {
        "ENST1:p.=",
        "ENST1:p.Ala112=",
        "ENST1:p.Ala112Gly",
        "ENST1:p.Ala112Ter",
        "ENST1:p.Ala112del",
        "ENST1:p.Met101Ter",
        "NM_1:c.12A>G",  # (the offset is only for protein positions, not these)
        None,
    }
    assert {r[0] for r in output.fetchall()} == expected


def test_variant_converter_negative_offset(ddbc, make_source):
    source = make_source("select 'A12G' as variant")
    plugin = VariantConverter()
    plugin.prepare(ddbc, source)
    plugin.set_parameter("offset", -2)
    assert plugin.execute(ddbc, source).fetchall() == [("p.Ala10Gly",)]


def test_variant_converter_into_another_column(ddbc, make_source):
    output = convert(ddbc, make_source, variant_out="hgvs")
    assert output.columns == ["variant", "hgvs"]
    rows = dict(output.fetchall())
    assert rows["A12G"] == "p.Ala12Gly" and rows["garbage"] is None and rows["syn_12A>G"] == "c.12A>G"


def test_variant_converter_keeps_the_other_columns(ddbc, make_source):
    source = make_source("select * from (values ('x', 'A12G', 5)) t(name, variant, n)")
    plugin = VariantConverter()
    plugin.prepare(ddbc, source)
    output = plugin.execute(ddbc, source)
    assert output.columns == ["name", "n", "variant"]
    assert output.fetchall() == [("x", 5, "p.Ala12Gly")]


@pytest.mark.parametrize(
    "params,expected_keys",
    [
        ({}, ["seq", "variant", "protein", "class"]),
        ({"variant.output": ""}, ["seq", "protein", "class"]),  # (no DNA variant column)
        ({"protein.output": ""}, ["seq", "variant", "class"]),  # (just the classification of the protein change)
        ({"classify": ""}, ["seq", "variant", "protein"]),
        ({"variant.output": "", "protein.output": "", "classify": ""}, ["seq"]),  # (nothing to call)
    ],
)
def test_outputs_can_be_left_out(params, expected_keys):
    result = variant_plugin(**params).transform({"seq": "ATGGTTAAATAG"})
    assert list(result) == expected_keys


def test_outputs_which_are_left_out_arent_added_to_a_query(ddbc, make_source):
    source = make_source("select 'ATGGTTAAATAG' as seq")
    plugin = variant_plugin(**{"variant.output": "", "classify": ""})
    plugin.prepare(ddbc, source)
    assert plugin.execute(ddbc, source).columns == ["seq", "protein"]
