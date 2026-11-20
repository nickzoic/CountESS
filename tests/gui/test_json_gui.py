import json
import os
import shutil
from unittest.mock import patch

import pytest

from countess.gui.main import CONFIG_FILE_TYPES, MainWindow, make_root


@pytest.fixture(name="window")
def fixture_window(tmp_path):
    for name in ("input1.csv", "input2.txt", "simple.ini"):
        shutil.copy(os.path.join("tests", name), tmp_path / name)
    root = make_root()
    mw = MainWindow(root, str(tmp_path / "simple.ini"))
    root.update()
    yield mw, tmp_path
    mw.graph.stop_thread()
    root.destroy()


def node_names(mw):
    return sorted(n.name for n in mw.graph.nodes)


@pytest.mark.gui
def test_save_and_load_json(window):
    mw, tmp_path = window
    names = node_names(mw)
    path = str(tmp_path / "saved.json")

    mw.config_save(path)
    assert mw.config_filename == path and not mw.config_changed
    data = json.loads(open(path, encoding="utf-8").read())
    assert data["format"] == "countess-config"
    assert sorted(n["name"] for n in data["nodes"]) == names

    old_graph = mw.graph
    mw.config_load(path)
    assert mw.graph is not old_graph
    assert node_names(mw) == names
    assert mw.config_filename == path


@pytest.mark.gui
def test_save_without_extension_keeps_format(window):
    mw, tmp_path = window
    mw.config_save(str(tmp_path / "first.json"))
    mw.config_save(str(tmp_path / "second"))
    assert (tmp_path / "second.json").exists() and not (tmp_path / "second.ini").exists()

    mw.config_save(str(tmp_path / "third.ini"))
    mw.config_save(str(tmp_path / "fourth"))
    assert (tmp_path / "fourth.ini").exists()


@pytest.mark.gui
def test_ini_still_saves_and_loads(window):
    mw, tmp_path = window
    names = node_names(mw)
    mw.config_save(str(tmp_path / "again.ini"))
    assert (tmp_path / "again.ini").read_text(encoding="utf-8").startswith("[")
    mw.config_load(str(tmp_path / "again.ini"))
    assert node_names(mw) == names


@pytest.mark.gui
def test_bad_file_leaves_current_config_alone(window):
    mw, tmp_path = window
    names = node_names(mw)
    graph = mw.graph
    filename = mw.config_filename

    bad = tmp_path / "bad.json"
    bad.write_text('{"format": "something else"}', encoding="utf-8")
    with patch("countess.gui.main.messagebox.showerror") as showerror:
        mw.config_load(str(bad))

    showerror.assert_called_once()
    assert "Not a CountESS" in showerror.call_args[0][1]
    assert mw.graph is graph and node_names(mw) == names and mw.config_filename == filename


@pytest.mark.gui
def test_export_name_from_json_config(window):
    mw, tmp_path = window
    mw.config_save(str(tmp_path / "graph.json"))
    with patch("countess.gui.main.ask_saveas_filename", return_value="") as ask:
        mw.config_export()
    assert ask.call_args.kwargs["initial_file"] == str(tmp_path / "graph.dot")


def test_file_dialog_offers_both_formats():
    assert any("*.json" in (ext if isinstance(ext, list) else [ext]) for _, ext in CONFIG_FILE_TYPES)
    assert any("*.ini" in (ext if isinstance(ext, list) else [ext]) for _, ext in CONFIG_FILE_TYPES)
    first_label, first_exts = CONFIG_FILE_TYPES[0]
    assert set(first_exts) == {"*.ini", "*.json"}  # the "everything" entry is the one for opening
