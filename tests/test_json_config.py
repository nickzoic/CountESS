import ast
import json
import logging
import math
import os
import shutil
import subprocess
import sys

import pytest

from countess.core import parameters as P
from countess.core.cmd import run
from countess.core.config import (
    graph_to_json,
    json_to_graph,
    read_config,
    read_json_config,
    write_config,
    write_json_config,
)
from countess.core.pipeline import PipelineGraph, PipelineNode
from countess.core.plugins import get_plugin_classes
from countess.plugins.data_table import DataTablePlugin

INI_FILES = ["tests/simple.ini", "tests/test_datatable.ini"]
COLUMNS = {"a": True, "b": False, "a_x": True, "a_y": True, "c": True}


def strict_json_loads(text):
    def refuse(constant):
        raise ValueError(f"{constant} is not valid JSON")

    return json.loads(text, parse_constant=refuse)


def perturb(param):
    """Give a parameter (and everything in it) some non-default values."""
    if isinstance(param, P.PerColumnArrayParam):
        param.set_column_choices(COLUMNS)
        for p in param.params:
            perturb(p)
    elif isinstance(param, P.ArrayParam):
        for _ in range(2):
            row = param.add_row()
            if row is not None:
                perturb(row)
    elif isinstance(param, P.HasSubParametersMixin):
        for p in list(param.params.values()):
            perturb(p)
    elif isinstance(param, P.MultiChoiceParam):
        param.set_choices(["a", "b", "c"])
        param.set_value("a")
        param.set_value("c")
    elif isinstance(param, P.MultiValueParam):
        param.set_value("x")
        param.set_value("y")
    elif isinstance(param, P.DictChoiceParam):
        if len(param.choices) > 1:
            param.set_choice(list(param.choices)[1])
    elif isinstance(param, P.ColumnOrIntegerParam):
        param.set_column_choices(COLUMNS)
        param.value = param.PREFIX + "a"
    elif isinstance(param, P.ColumnOrStringParam):
        param.set_column_choices(COLUMNS)
        param.value = param.PREFIX + "a_x"
    elif isinstance(param, P.ChoiceParam):
        param.set_column_choices(COLUMNS)
        if len(param.choices) > 1:
            param.set_choice(1)
        else:
            param.set_value("x*")
    elif isinstance(param, P.BooleanParam):
        param.value = not param.value
    elif isinstance(param, P.FloatParam):
        param.value = 2.5
    elif isinstance(param, P.IntegerParam):
        param.value = 7
    elif isinstance(param, P.FileBaseParam):
        param.value = "sub/file.txt"
    elif isinstance(param, P.ScalarParam):
        param.value = 'text "quoted" ünï\ncode'


def load_via_ini_method(plugin, base_dir):
    """What loading a .ini file does with a plugin's parameters."""
    new = plugin.__class__()
    for key, value in plugin.get_parameters("", base_dir):
        new.set_parameter(key, ast.literal_eval(repr(value)), base_dir)
    return new


def parameters_of(plugin, base_dir):
    return sorted(repr(kv) for kv in plugin.get_parameters("", base_dir))


PLUGINS = sorted(get_plugin_classes(), key=lambda c: c.__name__)


@pytest.mark.parametrize("perturbed", [False, True], ids=["default", "perturbed"])
@pytest.mark.parametrize("plugin_class", PLUGINS, ids=lambda c: c.__name__)
def test_plugin_round_trip(plugin_class, perturbed, tmp_path):
    """Every plugin must write valid JSON, load it back to exactly what loading
    the equivalent .ini file gives, and write the same JSON again."""
    base_dir = str(tmp_path)
    plugin = plugin_class()
    if perturbed:
        if plugin_class is DataTablePlugin:
            pytest.skip("covered by test_data_table_round_trip")
        for param in plugin.params.values():
            perturb(param)

    text = json.dumps(plugin.get_json_data(base_dir), allow_nan=False)

    from_json = plugin_class()
    from_json.set_json_data(strict_json_loads(text), base_dir)

    assert parameters_of(from_json, base_dir) == parameters_of(load_via_ini_method(plugin, base_dir), base_dir)
    assert json.dumps(from_json.get_json_data(base_dir), allow_nan=False) == text


@pytest.mark.parametrize("filename", INI_FILES)
def test_graph_round_trip(filename):
    original = read_config([filename])
    base_dir = os.path.dirname(os.path.abspath(filename))

    text = json.dumps(graph_to_json(original, base_dir), allow_nan=False)
    loaded = json_to_graph(strict_json_loads(text), base_dir)

    def summary(graph):
        result = {}
        for node in graph.nodes:
            if node.plugin:
                node.load_config()
            result[node.name] = (
                node.uuid,
                node.plugin.__class__ if node.plugin else None,
                sorted(p.name for p in node.parent_nodes),
                tuple(round(c, 3) for c in node.position),
                node.notes,
                node.sort_column,
                node.sort_descending,
                parameters_of(node.plugin, base_dir) if node.plugin else None,
            )
        return result

    assert len(loaded.nodes) == len(original.nodes)
    assert summary(loaded) == summary(original)

    # ... and saving the loaded graph gives the same file again
    assert json.dumps(graph_to_json(loaded, base_dir), allow_nan=False) == text


def test_json_is_readable():
    data = graph_to_json(read_config(["tests/simple.ini"]), "tests")
    assert data["format"] == "countess-config" and data["version"] == 1
    node = next(n for n in data["nodes"] if n["name"] == "CSV Load 1")
    assert node["plugin"]["module"] == "countess.plugins.csv"
    assert node["plugin"]["class"] == "LoadCsvPlugin"
    assert node["parameters"]["delimiter"] == ","
    assert node["parameters"]["files"] == [{"filename": "input1.csv"}] or "input1.csv" in json.dumps(
        node["parameters"]["files"]
    )
    # nodes come after the nodes they depend on
    names = [n["name"] for n in data["nodes"]]
    for n in data["nodes"]:
        for parent in n.get("parents", []):
            assert names.index(parent) < names.index(n["name"])


def test_filenames_relative_to_config(tmp_path):
    """Filenames in the file are relative to the file, so a project directory can be moved."""
    project = tmp_path / "project"
    (project / "conf").mkdir(parents=True)
    (project / "data").mkdir()
    for name in ("input1.csv", "input2.txt"):
        shutil.copy(os.path.join("tests", name), project / "data" / name)

    ini = open("tests/simple.ini", encoding="utf-8").read()
    for name in ("input1.csv", "input2.txt", "output.csv"):
        ini = ini.replace(f"'{name}'", f"'../data/{name}'")
    (project / "conf" / "c.ini").write_text(ini, encoding="utf-8")
    graph = read_config([str(project / "conf" / "c.ini")])
    write_json_config(graph, str(project / "conf" / "c.json"))

    text = (project / "conf" / "c.json").read_text(encoding="utf-8")
    assert "../data/input1.csv" in text.replace("\\\\", "/")
    assert "../data/output.csv" in text.replace("\\\\", "/")
    assert str(project) not in text

    moved = tmp_path / "moved"
    shutil.copytree(project, moved)
    loaded = read_json_config(str(moved / "conf" / "c.json"))
    paths = [
        f.filename.get_file_path()
        for n in loaded.nodes
        if n.plugin.__class__.__name__ == "LoadCsvPlugin"
        for f in n.plugin.files
    ]
    assert sorted(paths) == sorted(str(moved / "data" / name) for name in ("input1.csv", "input2.txt"))
    assert all(os.path.exists(p) for p in paths)


EXPECTED_OUTPUT = """"thing","result"
"foo",23512.798828125
"baz",12769.0
"qux",14728.1142578125
"""


def make_json_project(tmp_path):
    for name in ("input1.csv", "input2.txt"):
        shutil.copy(os.path.join("tests", name), tmp_path / name)
    shutil.copy("tests/simple.ini", tmp_path / "simple.ini")
    graph = read_config([str(tmp_path / "simple.ini")])
    write_config(graph, str(tmp_path / "simple.json"))
    return tmp_path / "simple.json"


def test_run_json_config_from_command_line(tmp_path):
    path = make_json_project(tmp_path)
    run([str(path)])
    assert (tmp_path / "output.csv").read_text(encoding="utf-8") == EXPECTED_OUTPUT


def test_command_line_set_overrides_json_config(tmp_path):
    path = make_json_project(tmp_path)
    other = tmp_path / "other.csv"
    run(["--set", f"CSV Save 5.filename={str(other)!r}", str(path)])
    assert other.read_text(encoding="utf-8") == EXPECTED_OUTPUT
    assert not (tmp_path / "output.csv").exists()


def test_missing_or_bad_json_from_command_line(tmp_path, caplog):
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    for filename in (bad, tmp_path / "missing.json"):
        with pytest.raises(SystemExit) as excinfo:
            run([str(filename)])
        assert excinfo.value.code == 2
    assert "not valid JSON" in caplog.text


def test_read_and_write_config_choose_format_by_extension(tmp_path):
    graph = read_config(["tests/simple.ini"])
    for name in ("g.json", "g.JSON"):
        write_config(graph, str(tmp_path / name))
        assert json.loads((tmp_path / name).read_text(encoding="utf-8"))["format"] == "countess-config"
        assert len(read_config([str(tmp_path / name)]).nodes) == len(graph.nodes)

    write_config(graph, str(tmp_path / "g.ini"))
    assert (tmp_path / "g.ini").read_text(encoding="utf-8").startswith("[")

    with pytest.raises(ValueError):
        read_config([str(tmp_path / "g.json"), str(tmp_path / "g.ini")])


def test_nodes_in_any_order_and_unconfigured_nodes():
    graph = PipelineGraph()
    new = PipelineNode(name="NEW", position=(0.25, 0.75), notes="note\nwith lines")
    graph.nodes.append(new)
    data = graph_to_json(graph)
    assert "plugin" not in data["nodes"][0] and "parameters" not in data["nodes"][0]

    data["nodes"].append({"name": "child", "parents": ["NEW", "later"]})
    data["nodes"].append({"name": "later"})  # a parent which is defined after its child
    loaded = json_to_graph(data)
    child = loaded.find_node("child")
    assert sorted(p.name for p in child.parent_nodes) == ["NEW", "later"]
    assert loaded.find_node("NEW").notes == "note\nwith lines"
    assert loaded.find_node("NEW").position == (0.25, 0.75)


@pytest.mark.parametrize(
    "data,message",
    [
        ([], "Not a CountESS"),
        ({"format": "something-else", "version": 1}, "Not a CountESS"),
        ({"format": "countess-config", "version": 99}, "Unsupported"),
        ({"format": "countess-config", "version": 1, "nodes": {}}, "should be a list"),
        ({"format": "countess-config", "version": 1, "nodes": [{"plugin": {}}]}, "needs a name"),
        ({"format": "countess-config", "version": 1, "nodes": [{"name": "a"}, {"name": "a"}]}, "Duplicate"),
        ({"format": "countess-config", "version": 1, "nodes": [{"name": "a", "parents": ["x"]}]}, "unknown parent"),
    ],
)
def test_bad_documents(data, message):
    with pytest.raises(ValueError, match=message):
        json_to_graph(data)


def test_invalid_json_file(tmp_path):
    path = tmp_path / "x.json"
    path.write_text("{ nope", encoding="utf-8")
    with pytest.raises(ValueError, match="not valid JSON"):
        read_json_config(str(path))


def test_bad_parameters_are_warnings(caplog):
    data = graph_to_json(read_config(["tests/simple.ini"]), "tests")
    node = next(n for n in data["nodes"] if n["plugin"]["class"] == "LoadCsvPlugin")
    node["parameters"]["no_such_parameter"] = 1
    node["parameters"]["header"] = "this is not a boolean"
    node["parameters"]["delimiter"] = ";"
    other = {"name": "gone", "plugin": {"module": "no.such.module", "class": "Nothing"}, "parameters": {"a": 1}}
    data["nodes"].append(other)

    with caplog.at_level(logging.WARNING):
        graph = json_to_graph(data, "tests")

    assert "no_such_parameter" in caplog.text
    assert "header" in caplog.text
    assert "Module Not Found: no.such.module" in caplog.text
    assert graph.find_node("gone").plugin is None
    loaded = graph.find_node(node["name"])
    assert loaded.plugin.delimiter.value == ";"  # everything else still got loaded


def test_non_finite_floats_are_valid_json(tmp_path):
    param = P.FloatParam("f")
    for value in (math.nan, math.inf, -math.inf, 1.5, 3):
        param.value = value
        data = param.get_json_data()
        assert strict_json_loads(json.dumps(data, allow_nan=False)) == data
        other = P.FloatParam("f")
        other.set_json_data(data)
        assert repr(float(other.value)) == repr(float(value))
    assert P.FloatParam("f", 3).get_json_data() == 3.0 and isinstance(P.FloatParam("f", 3).get_json_data(), float)

    # and a whole file containing them
    graph = PipelineGraph()
    plugin = DataTablePlugin()
    graph.nodes.append(PipelineNode(name="t", plugin=plugin))
    plugin.columns.add_row().name = "x"
    plugin.columns.params[0].type = "FLOAT"
    plugin.fix_columns()
    plugin.rows.add_row()["x"].value = math.nan
    write_json_config(graph, str(tmp_path / "nan.json"))
    strict_json_loads((tmp_path / "nan.json").read_text(encoding="utf-8"))
    loaded = read_json_config(str(tmp_path / "nan.json"))
    assert math.isnan(loaded.find_node("t").plugin.rows[0]["x"].value)


def test_data_table_round_trip():
    plugin = DataTablePlugin()
    for name, type_ in (("aaa", "STRING"), ("bbb", "INTEGER"), ("ccc", "BOOLEAN")):
        row = plugin.columns.add_row()
        row.name = name
        row.type = type_
    plugin.fix_columns()
    for values in (("one", 1, True), ("two", 2, False), ("three", 3, True)):
        new_row = plugin.rows.add_row()
        for name, value in zip(("aaa", "bbb", "ccc"), values):
            new_row[name].value = value

    data = plugin.get_json_data()
    assert data["rows"] == [
        {"aaa": "one", "bbb": 1, "ccc": True},
        {"aaa": "two", "bbb": 2, "ccc": False},
        {"aaa": "three", "bbb": 3, "ccc": True},
    ]

    loaded = DataTablePlugin()
    loaded.set_json_data(strict_json_loads(json.dumps(data)))
    assert loaded.get_json_data() == data
    assert [type(c.value) for c in loaded.rows[0].params.values()] == [str, int, bool]


# ---- parameter level


def test_multi_choice_values_in_choice_order_and_unknown_values_kept():
    param = P.MultiChoiceParam("m", choices=["z", "y", "x"])
    param.set_value("x")
    param.set_value("z")
    assert param.get_json_data() == ["z", "x"]

    other = P.MultiChoiceParam("m")
    other.set_json_data(["q", "p"])  # choices aren't known yet when a file is loaded
    assert other.get_json_data() == ["q", "p"] or other.get_json_data() == ["p", "q"]
    assert set(other.choices) == {"p", "q"}
    other.set_json_data(["p"])  # loading replaces what was there
    assert other.get_json_data() == ["p"]


def test_column_group_keeps_its_star():
    param = P.ColumnGroupChoiceParam("g")
    param.set_column_choices({"abc_1": True, "abc_2": True, "xyz": True})
    param.set_value("abc_*")
    other = P.ColumnGroupChoiceParam("g")
    other.set_json_data(param.get_json_data())
    assert other.value == "abc_*" and other.get_column_prefix() == "abc_"


def test_column_or_index_none():
    param = P.ColumnOrIndexChoiceParam("c")
    assert param.get_json_data() is None
    param.set_column_choices({"a": True})
    param.value = "a"
    assert param.get_json_data() == "a"
    other = P.ColumnOrIndexChoiceParam("c")
    other.set_json_data("a")
    other.set_json_data(None)
    assert other.is_index()


def test_per_column_array_keeps_column_names():
    param = P.PerColumnArrayParam("cols", P.BooleanParam("flag"))
    param.set_column_choices({"alpha": True, "beta": True, "gamma": True})
    param.params[1].value = True
    data = param.get_json_data()
    assert list(data) == ["alpha", "beta", "gamma"]

    other = P.PerColumnArrayParam("cols", P.BooleanParam("flag"))
    other.set_json_data(data)
    assert [(p.label, p.value) for p in other.params] == [("alpha", False), ("beta", True), ("gamma", False)]


def test_array_param_row_limits():
    padded = P.ArrayParam("a", P.StringParam("s"), min_size=3)
    padded.set_json_data(["x", "y", "z", "w"])
    assert [p.value for p in padded.params] == ["x", "y", "z", "w"]  # padding doesn't shift rows
    short = P.ArrayParam("a", P.StringParam("s"), min_size=3)
    short.set_json_data(["x"])
    assert len(short.params) == 3 and short.params[0].value == "x"  # padded up to min_size

    capped = P.ArrayParam("a", P.StringParam("s"), max_size=2)
    capped.set_json_data(["x", "y", "z"])
    assert [p.value for p in capped.params] == ["x", "y"]


def test_file_param_paths(tmp_path):
    param = P.FileParam("f")
    param.set_json_data("sub/file.txt", str(tmp_path))
    assert param.get_file_path() == str(tmp_path / "sub" / "file.txt")
    assert param.get_json_data(str(tmp_path)) == os.path.join("sub", "file.txt")
    assert param.get_json_data(str(tmp_path / "elsewhere")) == os.path.join("..", "sub", "file.txt")
    param.set_json_data("", str(tmp_path))
    assert param.get_json_data(str(tmp_path)) == ""


def test_dict_choice_round_trip():
    param = P.DictChoiceParam("d", choices={"one": "First", "two": "Second"})
    param.set_choice("two")
    other = P.DictChoiceParam("d", choices={"one": "First", "two": "Second"})
    other.set_json_data(param.get_json_data())
    assert other.value == "Second"


def test_no_debug_output_from_choice_params(capsys):
    param = P.ChoiceParam("c", choices=["a"])
    param.set_choices(["a", "b"])
    param.get_json_schema()
    assert capsys.readouterr().out == ""


def test_multi_value_param_is_a_sorted_list():
    param = P.MultiValueParam("m", {"b", "c", "a"})
    data = param.get_json_data()
    assert data == ["a", "b", "c"]
    assert json.dumps(data) == '["a", "b", "c"]'  # (a set would raise TypeError)

    other = P.MultiValueParam("m", {"left over"})
    other.set_json_data(data)
    assert other.get_values() == {"a", "b", "c"}


MULTI_VALUE_SCRIPT = """
import json
from countess.core import parameters as P
words = ["alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel", "india", "juliet"]
plain = P.MultiValueParam("m", words)
choice = P.MultiChoiceParam("c", choices=words)
for w in reversed(words[::2]):
    choice.set_value(w)
print(json.dumps([list(plain.get_parameters("k")), plain.get_hash_value(),
                  list(choice.get_parameters("k")), choice.get_hash_value()]))
"""


def test_multi_value_output_does_not_depend_on_hash_randomization():
    """The values of a MultiValueParam are kept in a set, whose order depends on the
    per-process string hash seed: what's written to a config file (and the hash of the
    plugin) mustn't."""
    outputs = set()
    for seed in ("0", "1", "2", "3", "4", "5"):
        result = subprocess.run(
            [sys.executable, "-c", MULTI_VALUE_SCRIPT],
            env={**os.environ, "PYTHONHASHSEED": seed},
            capture_output=True,
            text=True,
            check=True,
        )
        outputs.add(result.stdout.strip().splitlines()[-1])
    assert len(outputs) == 1

    plain_params, _, choice_params, _ = json.loads(outputs.pop())
    assert [v for _, v in plain_params] == sorted(v for _, v in plain_params)  # (sorted, when there are no choices)
    assert [v for _, v in choice_params] == [
        "alpha",
        "charlie",
        "echo",
        "golf",
        "india",
    ]  # in the order of the choices
