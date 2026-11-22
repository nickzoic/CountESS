import time
import tkinter as tk
from unittest.mock import MagicMock, patch

import pytest

from countess.core.config import read_config
from countess.core.pipeline import PipelineNode
from countess.gui.main import ConfiguratorWrapper, MainWindow, PluginChooserFrame, make_root
from countess.plugins.data_table import DataTablePlugin


def descendants(widget):
    for w in widget.winfo_children():
        yield w
        yield from descendants(w)


@pytest.mark.gui
def test_chooser():
    callback = MagicMock()

    root = make_root()
    choose = PluginChooserFrame(root, "X", callback, True, True)

    for x in descendants(choose):
        if isinstance(x, tk.Button):
            x.invoke()
            break

    callback.assert_called_once()

    root.destroy()


@pytest.mark.gui
def test_main():
    root = make_root()
    node = PipelineNode(name="NEW", plugin=DataTablePlugin())
    callback = MagicMock()

    wrap = ConfiguratorWrapper(root, node, None, callback)
    time.sleep(0.1)
    root.update()
    wrap.on_add_notes()

    root.update()

    wrap.destroy()
    root.destroy()


@pytest.mark.gui
def test_mac_quit_runs_the_exit_button_code():
    """On macOS Cmd-Q runs ::tk::mac::Quit if it exists, and that has to be the same as
    pressing Exit or DuckDB's static destructors crash the interpreter."""
    root = make_root()
    try:
        with patch.object(MainWindow, "program_exit") as program_exit:  # (the real one quits the application)
            window = MainWindow(root)
            root.tk.call("::tk::mac::Quit")
        program_exit.assert_called_once_with()
        window.graph.stop_thread(timeout=10)
    finally:
        root.destroy()
