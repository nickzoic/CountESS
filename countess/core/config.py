import ast
import io
import json
import logging
import os.path
import re
from configparser import ConfigParser
from typing import Any

from countess import VERSION
from countess.core.pipeline import PipelineGraph, PipelineNode
from countess.core.plugins import load_plugin

logger = logging.getLogger(__name__)


def read_config_dict(name: str, base_dir: str, config_dict: dict) -> PipelineNode:
    if "_module" in config_dict:
        module_name = config_dict["_module"]
        class_name = config_dict["_class"]
        # XXX version = config_dict.get("_version")
        # XXX hash_digest = config_dict.get("_hash")
        plugin = load_plugin(module_name, class_name)
    else:
        plugin = None

    position_str = config_dict.get("_position")
    notes = config_dict.get("_notes")

    position = None
    if position_str:
        position_match = re.match(r"(\d+) (\d+)$", position_str)
        if position_match:
            position = (
                int(position_match.group(1)) / 1000,
                int(position_match.group(2)) / 1000,
            )

    sort = config_dict.get("_sort", "0 0").split()

    # XXX check version and hash_digest and emit warnings.

    node = PipelineNode(
        name=name,
        uuid=config_dict.get("_uuid"),
        plugin=plugin,
        position=position,
        notes=notes,
        sort_column=int(sort[0]),
        sort_descending=bool(int(sort[1])),
    )
    for key, val in config_dict.items():
        if not key.startswith("_"):
            node.set_config(key, ast.literal_eval(val), base_dir)
    return node


def read_config(
    filenames: list[str],
) -> PipelineGraph:
    """Reads `filenames` and returns a PipelineGraph.  A file with a name ending
    in .json is read as JSON (see read_json_config), anything else as an .ini file."""

    if any(is_json_filename(f) for f in filenames):
        if len(filenames) != 1:
            raise ValueError("A JSON config file can't be combined with other config files")
        return read_json_config(filenames[0])

    cp = ConfigParser(interpolation=None)
    for filename in filenames:
        with open(filename, "r", encoding="utf-8") as fh:
            cp.read_file(fh)

    base_dir = os.path.dirname(filenames[0] if filenames else ".")

    return config_to_graph(cp, base_dir)


def config_to_graph(cp: ConfigParser, base_dir: str = ".") -> PipelineGraph:
    pipeline_graph = PipelineGraph()
    nodes_by_name: dict[str, PipelineNode] = {}

    for section_name in cp.sections():
        config_dict = dict(cp[section_name])
        node = read_config_dict(section_name, base_dir, config_dict)

        for key, val in config_dict.items():
            if key.startswith("_parent."):
                node.add_parent(nodes_by_name[val])

        pipeline_graph.nodes.append(node)
        nodes_by_name[section_name] = node

    return pipeline_graph


def graph_to_config(pipeline_graph: PipelineGraph, base_dir: str = ".") -> ConfigParser:
    pipeline_graph.reset_node_names()
    cp = ConfigParser(interpolation=None)
    for node in pipeline_graph.traverse_nodes():
        write_config_node(node, cp, base_dir)
    return cp


def write_config(pipeline_graph: PipelineGraph, filename: str):
    """Write `pipeline_graph`'s configuration out to `filename`, as JSON if
    its name ends in .json (see write_json_config) or as an .ini file otherwise."""
    if is_json_filename(filename):
        return write_json_config(pipeline_graph, filename)

    base_dir = os.path.dirname(filename)
    cp = graph_to_config(pipeline_graph, base_dir)

    with open(filename, "w", encoding="utf-8") as fh:
        cp.write(fh)


def is_json_filename(filename: str) -> bool:
    return str(filename).lower().endswith(".json")


# The JSON format is a single object:
#
#   {
#     "format": "countess-config",
#     "version": 1,
#     "countess_version": "0.1.28",
#     "nodes": [
#       {
#         "name": "Load",                      (unique)
#         "uuid": "...",
#         "plugin": {"module": ..., "class": ..., "version": ..., "hash": ...},
#         "parameters": { ... plugin.get_json_data() ... },
#         "parents": ["name of another node", ...],
#         "position": [0.25, 0.5],
#         "sort": {"column": 0, "descending": false},
#         "notes": "..."
#       }, ...
#     ]
#   }
#
# Everything but "name" is optional, and a node with no "plugin" is one which
# hasn't been set up yet.  Filenames in "parameters" are relative to the directory
# the config file is in, as they are in .ini files.

JSON_FORMAT = "countess-config"
JSON_VERSION = 1


def node_to_json(node: PipelineNode, base_dir: str = ".") -> dict[str, Any]:
    data: dict[str, Any] = {"name": node.name, "uuid": node.uuid}
    if node.plugin:
        node.load_config()
        data["plugin"] = {
            "module": node.plugin.__module__,
            "class": node.plugin.__class__.__name__,
            "version": node.plugin.version,
            "hash": node.plugin.hash(),
        }
        data["sort"] = {"column": node.sort_column, "descending": bool(node.sort_descending)}
    if node.position:
        data["position"] = [round(node.position[0], 4), round(node.position[1], 4)]
    if node.notes:
        data["notes"] = node.notes
    if node.parent_nodes:
        # sorted, so that the same graph always makes the same file
        data["parents"] = sorted(parent.name for parent in node.parent_nodes)
    if node.plugin:
        data["parameters"] = node.plugin.get_json_data(base_dir)
    return data


def graph_to_json(pipeline_graph: PipelineGraph, base_dir: str = ".") -> dict[str, Any]:
    pipeline_graph.reset_node_names()
    return {
        "format": JSON_FORMAT,
        "version": JSON_VERSION,
        "countess_version": VERSION,
        "nodes": [node_to_json(node, base_dir) for node in pipeline_graph.traverse_nodes()],
    }


def json_to_graph(data: Any, base_dir: str = ".") -> PipelineGraph:
    """Make a PipelineGraph from what graph_to_json() makes.  Raises ValueError
    if `data` isn't a CountESS config.  Problems with the parameters of
    individual plugins (ones which no longer exist, say) are logged as warnings
    rather than being fatal, as they are with .ini files."""

    if not isinstance(data, dict) or data.get("format") != JSON_FORMAT:
        raise ValueError("Not a CountESS JSON config file")
    version = data.get("version")
    if not isinstance(version, int) or version > JSON_VERSION:
        raise ValueError(f"Unsupported CountESS JSON config version {version!r}")
    node_list = data.get("nodes", [])
    if not isinstance(node_list, list):
        raise ValueError('"nodes" should be a list')

    pipeline_graph = PipelineGraph()
    nodes_by_name: dict[str, PipelineNode] = {}

    for node_data in node_list:
        if not isinstance(node_data, dict) or not isinstance(node_data.get("name"), str):
            raise ValueError("Every node needs a name")
        name = node_data["name"]
        if name in nodes_by_name:
            raise ValueError(f"Duplicate node name {name!r}")

        plugin = None
        if "plugin" in node_data:
            plugin_data = node_data["plugin"]
            # XXX check "version" and "hash" and emit warnings, as for .ini files
            plugin = load_plugin(plugin_data["module"], plugin_data["class"])
            if plugin and "parameters" in node_data:
                plugin.set_json_data(node_data["parameters"], base_dir)

        position = node_data.get("position")
        sort = node_data.get("sort", {})
        node = PipelineNode(
            name=name,
            uuid=node_data.get("uuid"),
            plugin=plugin,
            position=(float(position[0]), float(position[1])) if position else None,
            notes=node_data.get("notes"),
            sort_column=int(sort.get("column", 0)),
            sort_descending=bool(sort.get("descending", False)),
        )
        pipeline_graph.nodes.append(node)
        nodes_by_name[name] = node

    # links are made afterwards so that nodes can be in any order.
    for node_data in node_list:
        for parent_name in node_data.get("parents", []):
            try:
                nodes_by_name[node_data["name"]].add_parent(nodes_by_name[parent_name])
            except KeyError as exc:
                raise ValueError(f"Node {node_data['name']!r} has unknown parent {parent_name!r}") from exc

    return pipeline_graph


def read_json_config(filename: str) -> PipelineGraph:
    """Reads the JSON config file `filename` and returns a PipelineGraph"""
    with open(filename, "r", encoding="utf-8") as fh:
        try:
            data = json.load(fh)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{filename} is not valid JSON: {exc}") from exc
    return json_to_graph(data, os.path.dirname(os.path.abspath(filename)))


def write_json_config(pipeline_graph: PipelineGraph, filename: str):
    """Write `pipeline_graph`'s configuration out to `filename` as JSON"""
    data = graph_to_json(pipeline_graph, os.path.dirname(os.path.abspath(filename)))
    # allow_nan=False makes sure that we only ever write real JSON.
    text = json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False)
    with open(filename, "w", encoding="utf-8") as fh:
        fh.write(text + "\n")


def write_config_node_string(node: PipelineNode, base_dir: str = ""):
    cp = ConfigParser(interpolation=None)
    write_config_node(node, cp, base_dir)
    buf = io.StringIO()
    cp.write(buf)
    return buf.getvalue()


def write_config_node(node: PipelineNode, cp: ConfigParser, base_dir: str):
    cp.add_section(node.name)
    if node.plugin:
        cp[node.name].update(
            {
                "_uuid": node.uuid,
                "_module": node.plugin.__module__,
                "_class": node.plugin.__class__.__name__,
                "_version": node.plugin.version,
                "_hash": node.plugin.hash(),
                "_sort": "%d %d" % (node.sort_column, 1 if node.sort_descending else 0),
            }
        )
    if node.position:
        xx, yy = node.position
        cp[node.name]["_position"] = "%d %d" % (xx * 1000, yy * 1000)
    if node.notes:
        cp[node.name]["_notes"] = node.notes
    for n, parent in enumerate(node.parent_nodes):
        cp[node.name][f"_parent.{n}"] = parent.name
    if node.plugin:
        node.load_config()
        for k, v in node.plugin.get_parameters("", base_dir):
            cp[node.name][k] = repr(v)


def export_config_graphviz(pipeline_graph: PipelineGraph, filename: str):
    with open(filename, "w", encoding="utf-8") as fh:
        fh.write("digraph {\n")
        for node in pipeline_graph.traverse_nodes():
            label = node.name.replace('"', r"\"")
            if node.child_nodes and not node.parent_nodes:
                fh.write(f'\t"{label}" [ shape="invhouse" ];\n')
            elif node.parent_nodes and not node.child_nodes:
                fh.write(f'\t"{label}" [ shape="house" ];\n')
            else:
                fh.write(f'\t"{label}" [ shape="box" ];\n')

            for child_node in node.child_nodes:
                label2 = child_node.name.replace('"', r"\"")
                fh.write(f'\t"{label}" -> "{label2}";\n')

        fh.write("}\n")
