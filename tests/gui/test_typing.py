import time
import tkinter as tk
from tkinter import ttk

import pytest

from countess.core.parameters import IntegerParam, StringParam, TextParam
from countess.core.plugins import DuckdbPlugin
from countess.gui.config import PluginConfigurator
from countess.gui.main import make_root


class TypingPlugin(DuckdbPlugin):
    version = "0"
    num_inputs = 0
    num_outputs = 1
    number = IntegerParam("number", 0)
    name_ = StringParam("string", "hello")
    text = TextParam("text", "some text")


def descendants(widget):
    for w in widget.winfo_children():
        yield w
        yield from descendants(w)


def pump(root, seconds=0.05):
    deadline = time.time() + seconds
    while time.time() < deadline:
        root.update()
        time.sleep(0.01)


def focus(root, widget):
    widget.focus_force()
    pump(root)
    assert widget.focus_get() is widget, "could not get keyboard focus under this display"


@pytest.fixture(name="setup")
def fixture_setup():
    root = make_root()
    plugin = TypingPlugin()
    changes = []
    configurator = PluginConfigurator(root, plugin, lambda *a: changes.append(a))
    pump(root)
    entries = [w for w in descendants(configurator.frame) if isinstance(w, tk.Entry)]
    texts = [w for w in descendants(configurator.frame) if isinstance(w, tk.Text)]
    yield root, plugin, configurator, entries, texts
    root.destroy()


@pytest.mark.gui
def test_refresh_does_not_overwrite_entry_being_typed(setup):
    """ "-" can't be stored in an IntegerParam (it stays 0) but that mustn't stop you getting to "-5"."""
    root, plugin, configurator, entries, _ = setup
    number_entry = configurator.wrapper_cache["number"].entry
    assert number_entry in entries
    focus(root, number_entry)

    number_entry.delete(0, tk.END)
    number_entry.insert(0, "-")
    pump(root)
    assert plugin.number.value == 0  # parameter couldn't take it

    configurator.update()  # this is what happens when a recalculation finishes
    pump(root)
    assert number_entry.get() == "-"

    number_entry.insert(tk.END, "5")
    pump(root)
    configurator.update()
    assert number_entry.get() == "-5"
    assert plugin.number.value == -5


@pytest.mark.gui
def test_entry_resyncs_when_focus_leaves(setup):
    root, plugin, configurator, entries, _ = setup
    number_entry = configurator.wrapper_cache["number"].entry
    string_entry = configurator.wrapper_cache["name_"].entry
    focus(root, number_entry)

    number_entry.delete(0, tk.END)
    number_entry.insert(0, "-")
    pump(root)
    configurator.update()
    assert number_entry.get() == "-"

    focus(root, string_entry)  # leave the field: it should show what the parameter really holds
    assert number_entry.get() == "0"


@pytest.mark.gui
def test_refresh_still_updates_unfocused_entries(setup):
    root, plugin, configurator, entries, _ = setup
    number_entry = configurator.wrapper_cache["number"].entry
    string_entry = configurator.wrapper_cache["name_"].entry
    focus(root, string_entry)

    plugin.number.value = 42  # changed from elsewhere
    configurator.update()
    assert number_entry.get() == "42"
    assert string_entry.get() == "hello"


@pytest.mark.gui
def test_refresh_does_not_overwrite_text_being_typed(setup):
    root, plugin, configurator, _, texts = setup
    text = configurator.wrapper_cache["text"].entry
    assert text in texts
    focus(root, text)

    text.insert("1.0", "typed ")
    text.mark_set(tk.INSERT, "1.3")
    pump(root)

    plugin.text.value = "something else entirely"
    configurator.update()
    pump(root)

    assert text.get("1.0", "end-1c").startswith("typed ")
    assert text.index(tk.INSERT) == "1.3"  # cursor didn't jump either


@pytest.mark.gui
def test_editable_combobox_not_overwritten_but_values_refreshed():
    from countess.core.parameters import ColumnOrStringParam

    class ComboPlugin(TypingPlugin):
        col = ColumnOrStringParam("col")

    root = make_root()
    try:
        plugin = ComboPlugin()
        configurator = PluginConfigurator(root, plugin, lambda *a: None)
        pump(root)
        wrapper = configurator.wrapper_cache["col"]
        combo = wrapper.entry
        assert isinstance(combo, ttk.Combobox)
        focus(root, combo)

        combo.set("partial")
        pump(root)
        plugin.col.value = "something else"  # the parameter has moved on from what's being typed
        configurator.update()
        assert combo.get() == "partial"

        focus(root, configurator.wrapper_cache["number"].entry)  # leave the field
        assert combo.get() == "something else"
    finally:
        root.destroy()
