import getopt
import logging
import logging.handlers
import multiprocessing
import os
import queue
import re
import sys
import time

try:
    import tkinter as tk
except ImportError:
    msg = """countess_gui uses python's "tkinter" library.
This is generally supplied by a system package called "python3-full" or something similar."""
    print(msg)  # pylint: disable=bad-builtin
    sys.exit(1)

from collections.abc import Callable
from tkinter import messagebox, ttk
from typing import Optional, Union

import duckdb
import psutil

from countess import VERSION
from countess.core.config import (
    config_to_graph,
    export_config_graphviz,
    graph_to_config,
    is_json_filename,
    read_config,
    write_config,
)
from countess.core.pipeline import PipelineGraph, PipelineNode, PipelineNodeStatus
from countess.core.plugins import get_plugin_classes
from countess.gui.config import PluginConfigurator
from countess.gui.mini_browser import mini_browser_open
from countess.gui.tabular import TabularDataFrame
from countess.gui.tree import FlippyCanvas, GraphWrapper
from countess.gui.widgets import (
    LabeledProgressbar,
    ResizingFrame,
    ask_open_filename,
    ask_saveas_filename,
    get_icon,
    info_button,
)

preview_row_limit: Optional[int] = int(psutil.virtual_memory().available / 1024 / 1024 / 1024) * 10000

usage = """usage: countess_gui [--log LEVEL] [CONFIGFILE]

Start the CountESS GUI.

options:
    --help                         show this message & exit.
    --version                      show version
    --preview LIMIT                set preview mode row limit (default: %s)
    --log LEVEL                    set log level to LEVEL
    CONFIGFILE                     load configuration file (.ini or .json)
""" % (
    preview_row_limit or "NONE"
)


# import faulthandler
# faulthandler.enable(all_threads=True)

# logging.basicConfig(format="%(threadName)s:%(message)s")
logger = logging.getLogger(__name__)

logging_queue: multiprocessing.Queue = multiprocessing.Queue()

plugin_classes = sorted(get_plugin_classes(), key=lambda x: x.name)


class PluginChooserFrame(tk.Frame):
    def __init__(self, master, title, callback, has_parents, has_children, *a, **k):
        super().__init__(master, *a, **k)

        self.columnconfigure(0, weight=1)

        label_frame = tk.LabelFrame(self, text=title, padx=10, pady=10)
        label_frame.grid(row=1, column=0, sticky=tk.EW, padx=10, pady=10)

        for n, plugin_class in enumerate(plugin_classes):
            if (
                (has_parents and plugin_class.num_inputs == 0)
                or (not has_parents and plugin_class.num_inputs > 0)
                or (has_children and plugin_class.num_outputs == 0)
            ):
                continue

            label_text = plugin_class.description
            tk.Button(
                label_frame,
                text=plugin_class.name,
                command=lambda plugin_class=plugin_class: callback(plugin_class),
            ).grid(row=n + 1, column=0, sticky=tk.EW)
            tk.Label(label_frame, text=label_text).grid(row=n + 1, column=1, sticky=tk.W, padx=10)


class ConfiguratorWrapper:
    """Wraps up the PluginConfigurator or PluginChooserFrame, plus a
    DataFramePreview, to make up the larger part of the main window"""

    config_canvas: Optional[tk.Canvas] = None
    config_scrollbar: Optional[tk.Scrollbar] = None
    config_subframe: Optional[tk.Frame] = None
    config_subframe_id = None
    preview_subframe: Optional[tk.Frame] = None
    notes_widget: Optional[tk.Widget] = None
    info_toplevel: Optional[tk.Toplevel] = None
    info_frame: Optional[tk.Frame] = None
    configurator: Optional[PluginConfigurator] = None

    # Timings, in milliseconds.
    CHANGE_DEBOUNCE_MS = 200  # how long config changes have to stop for before the pipeline is told
    RESULT_POLL_MS = 250  # how often to check if a node has finished calculating
    PREVIEW_POLL_MS = 100  # how often to check if there's a result to show a preview of

    def __init__(self, tk_parent, node, ddbc, change_callback):
        logger.debug("ConfiguratorWrapper %s %s", node.name, node.uuid)
        self.node = node
        self.ddbc = ddbc
        self.change_callback = change_callback

        # Every deferred callback this wrapper schedules is tracked so that
        # destroy() can cancel them: otherwise they fire later against widgets
        # which no longer exist.  There is also at most one pending timer for
        # each of the "config changed" debounce, the "wait for result" poll and
        # the "wait for preview" poll, so repeated changes don't multiply them.
        self._destroyed = False
        self._after_ids: set[str] = set()
        self._change_after_id: Optional[str] = None
        self._poll_after_id: Optional[str] = None
        self._preview_after_id: Optional[str] = None
        # the node's generation (see PipelineNode.generation) when the preview was made
        self._preview_generation: Optional[int] = None

        self.frame = ResizingFrame(tk_parent, orientation=ResizingFrame.Orientation.VERTICAL, bg="darkgrey")
        self.subframe = self.frame.add_frame()
        self.subframe.columnconfigure(0, weight=1)
        self.subframe.rowconfigure(3, weight=1)

        self.name_var = tk.StringVar(self.subframe, value=node.name)
        tk.Entry(self.subframe, textvariable=self.name_var, font=("TkHeadingFont", 14, "bold")).grid(
            row=0, columnspan=2, sticky=tk.EW, padx=10, pady=5
        )
        self.name_var.trace_add("write", self.name_changed_callback)

        self.label = tk.Label(self.subframe, justify=tk.LEFT, wraplength=500)
        self.label.grid(sticky=tk.EW, row=1, padx=10, pady=5)
        self.label.bind("<Configure>", self.on_label_configure)

        self._after(50, self.show_config_subframe)
        self._after(500, self.replace_preview_subframe)

    def _after(self, delay_ms: int, func, *args) -> str:
        """Like frame.after() but cancelled by destroy()"""

        def _run():
            self._after_ids.discard(after_id)
            if not self._destroyed:
                func(*args)

        after_id = self.frame.after(delay_ms, _run)
        self._after_ids.add(after_id)
        return after_id

    def _cancel(self, after_id: Optional[str]) -> None:
        if after_id:
            self._after_ids.discard(after_id)
            try:
                self.frame.after_cancel(after_id)
            except tk.TclError:
                pass

    def show_config_subframe(self):
        if self.config_subframe:
            self.config_subframe.destroy()
        if self.config_canvas:
            self.config_canvas.destroy()
        self.config_canvas = tk.Canvas(self.subframe)
        self.config_scrollbar = ttk.Scrollbar(self.subframe, orient=tk.VERTICAL, command=self.config_canvas.yview)
        self.config_canvas.configure(yscrollcommand=self.config_scrollbar.set, bd=0)
        self.config_canvas.grid(row=3, column=0, sticky=tk.NSEW)
        self.config_scrollbar.grid(row=3, column=1, sticky=tk.NS)

        if self.node.plugin:
            self.node.load_config()
            if self.node.notes:
                self.show_notes_widget(self.node.notes)
            else:
                self.notes_widget = tk.Button(self.subframe, text="add notes", command=self.on_add_notes)
                self.notes_widget.grid(row=2, columnspan=2, padx=10, pady=5)

            descr = re.sub(r"\s+", " ", self.node.plugin.description)
            if self.node.plugin.additional:
                descr += "\n" + re.sub(r"\s+", " ", self.node.plugin.additional)

            self.label["text"] = "%s %s — %s" % (
                self.node.plugin.name,
                self.node.plugin.version,
                descr,
            )
            if self.node.plugin.link:
                info_button(self.frame, command=self.on_info_button_press).place(anchor=tk.NE, relx=1, y=50)
            self._after(100, self.show_configurator)

        else:
            has_parents = len(self.node.parent_nodes) > 0
            has_children = len(self.node.child_nodes) > 0
            self.config_subframe = PluginChooserFrame(
                self.config_canvas, "Choose Plugin", self.choose_plugin, has_parents, has_children
            )
            self.config_subframe.grid(sticky=tk.NSEW)

    def show_configurator(self):
        self.configurator = PluginConfigurator(self.config_canvas, self.node.plugin, self.config_change_callback)
        self.config_subframe = self.configurator.frame
        self.config_subframe_id = self.config_canvas.create_window((0, 0), window=self.config_subframe, anchor=tk.NW)
        self.config_subframe.bind(
            "<Configure>",
            lambda e: self.config_canvas.configure(scrollregion=self.config_canvas.bbox("all")),
        )
        self.config_canvas.bind("<Configure>", self.on_config_canvas_configure)

    def on_config_canvas_configure(self, *_):
        self.config_canvas.itemconfigure(self.config_subframe_id, width=self.config_canvas.winfo_width())

    def on_label_configure(self, *_):
        self.label["wraplength"] = self.label.winfo_width() - 20

    def on_info_button_press(self, *_):
        mini_browser_open(self.node.plugin.link)

    def on_info_toplevel_close(self):
        self.info_toplevel.destroy()
        self.info_toplevel = None

    def on_add_notes(self, *_):
        self.notes_widget.destroy()
        self.show_notes_widget()

    def choose_plugin(self, plugin_class):
        self.node.plugin = plugin_class()
        self.change_callback(self.node)
        self.name_var.set(self.node.name)
        self.show_config_subframe()
        self.config_change_callback()

    def show_notes_widget(self, notes=""):
        self.notes_widget = tk.Text(self.subframe, height=5)
        self.notes_widget.insert("1.0", notes)
        self.notes_widget.bind("<<Modified>>", self.notes_modified_callback)
        self.notes_widget.grid(row=2, columnspan=2, sticky=tk.EW, padx=10, pady=5)

    def show_preview_subframe(self):
        logger.debug("ConfiguratorWrapper.show_preview_subframe %s %s", self.node.name, self.node.uuid)

        # only ever one pending attempt at a time
        self._cancel(self._preview_after_id)
        self._preview_after_id = None

        if self.preview_subframe or not self.node.plugin or not self.node.plugin.show_preview:
            return

        # (noted before looking at the status: if the node changes after this
        # the generation will no longer match and the preview will be remade)
        generation = self.node.generation

        if self.node.status == PipelineNodeStatus.ERROR:
            logger.debug("ConfiguratorWrapper.show_preview_subframe %s", self.node.message)
            self.preview_subframe = tk.Frame(self.frame)
            text = tk.Text(self.preview_subframe)
            text.insert("1.0", self.node.message or "")
            text["state"] = "disabled"
            text.grid(sticky=tk.NSEW)
            self.frame.add_child(self.preview_subframe)
            self._preview_generation = generation

        elif self.node.status == PipelineNodeStatus.RESULT:
            logger.debug("ConfiguratorWrapper.show_preview_subframe %s", self.node.table_name)
            preview = TabularDataFrame(self.frame, cursor="arrow")
            try:
                preview.set_table(self.ddbc, self.ddbc.table(self.node.table_name))
                preview.set_sort_order(self.node.sort_column or 0, self.node.sort_descending)
            except duckdb.Error:
                # the worker dropped the table between us checking the status and
                # reading it: the node is about to be DIRTY, so try again shortly.
                logger.debug("ConfiguratorWrapper.show_preview_subframe table not available, retrying")
                preview.destroy()
                self._preview_after_id = self._after(self.PREVIEW_POLL_MS, self.show_preview_subframe)
                return
            preview.set_callback(self.preview_changed_callback)
            self.preview_subframe = preview
            self.frame.add_child(self.preview_subframe)
            self._preview_generation = generation

        else:
            self._preview_after_id = self._after(self.PREVIEW_POLL_MS, self.show_preview_subframe)

    def hide_preview_subframe(self):
        logger.debug("ConfiguratorWrapper.hide_preview_subframe %s %s", self.node.name, self.node.uuid)
        self._cancel(self._preview_after_id)
        self._preview_after_id = None
        if self.preview_subframe:
            self.frame.remove_child(self.preview_subframe).destroy()
            self.preview_subframe = None
        self._preview_generation = None

    def replace_preview_subframe(self):
        self.hide_preview_subframe()
        self.show_preview_subframe()

    def update_preview_subframe(self):
        """Make the preview show the node's current result.  Making a preview is
        slow (it gets as many text widgets as the table has columns) so if the node
        hasn't changed since it was made, as when going back to a node which was
        selected earlier, it's left as it is."""
        if self.preview_subframe is not None and self._preview_generation == self.node.generation:
            return
        self.replace_preview_subframe()

    def notify(self):
        self._cancel(self._poll_after_id)
        self._poll_after_id = None
        self.config_change_poll_callback()

    def preview_changed_callback(self, offset: int, sort_col: int, sort_desc: bool) -> None:
        self.node.sort_column = sort_col
        self.node.sort_descending = sort_desc

    def name_changed_callback(self, *_):
        name = self.name_var.get()
        self.node.name = name
        self.change_callback(self.node)

    def notes_modified_callback(self, *_):
        self.node.notes = self.notes_widget.get("1.0", tk.END).strip()
        self.notes_widget.edit_modified(False)

    def config_change_callback(self, *_):
        """Called immediately if a change to config has occurred.  Changes tend
        to come in bursts (every keystroke in an entry field is one) so we wait
        for things to settle down before disturbing the pipeline."""
        logger.debug("config_change_callback %s", self.node.name)
        self._cancel(self._change_after_id)
        self._change_after_id = self._after(self.CHANGE_DEBOUNCE_MS, self._config_change_fire)

    def _config_change_fire(self):
        self._change_after_id = None
        self.change_callback(self.node, True)
        self._start_poll()

    def _start_poll(self):
        if self._poll_after_id is None:
            self._poll_after_id = self._after(self.RESULT_POLL_MS, self.config_change_poll_callback)

    def config_change_poll_callback(self):
        logger.debug("ConfiguratorWrapper.config_change_poll_callback")
        self._poll_after_id = None
        if self._change_after_id is not None or self.node.status in (
            PipelineNodeStatus.DIRTY,
            PipelineNodeStatus.WORKING,
        ):
            self._start_poll()
        else:
            self.config_change_poll_done()

    def config_change_poll_done(self):
        logger.debug("ConfiguratorWrapper.config_change_poll_done")
        self.update_preview_subframe()
        if self.config_scrollbar and self.configurator:
            pos1, pos2 = self.config_scrollbar.get()
            self.configurator.update()
            # update_idletasks (as opposed to update) lays out the widgets without
            # running arbitrary other event handlers in the middle of this one.
            self.frame.update_idletasks()
            self.config_canvas.yview_moveto(pos1)
            self.config_scrollbar.set(pos1, pos2)
        else:
            self.frame.update_idletasks()

    def destroy(self):
        self._destroyed = True
        for after_id in list(self._after_ids):
            self._cancel(after_id)
        self.frame.destroy()


class SplashWrapper:
    def __init__(self, tk_parent):
        self.frame = tk.Frame(tk_parent)
        self.frame.configure(cursor="heart")

        font = ("TkHeadingFont", 16, "bold")
        subframe = tk.Frame(self.frame, highlightbackground="black", highlightthickness=1)
        subframe.grid(padx=10, pady=10)
        self.frame.columnconfigure(0, weight=1)
        self.frame.rowconfigure(0, weight=1)
        tk.Label(subframe, image=get_icon(tk_parent, "countess")).grid(padx=10, pady=10)
        tk.Label(subframe, text=f"CountESS {VERSION}", font=font).grid(padx=10, pady=10)

    def notify(self):
        pass

    def destroy(self):
        self.frame.destroy()


class ButtonMenu:  # pylint: disable=R0903
    def __init__(self, tk_parent, buttons):
        self.frame = tk.Frame(tk_parent)
        for button_number, (button_label, button_command) in enumerate(buttons):
            tk.Button(self.frame, text=button_label, command=button_command).grid(
                row=0, column=button_number, sticky=tk.EW
            )
        self.frame.grid(sticky=tk.NSEW)


def runwindow_subprocess_target(config):
    logger.debug("Subprocess started")
    graph = config_to_graph(config)
    graph.run()
    logger.debug("Subprocess finished")


class _CallbackLoggingHandler(logging.Handler):
    """This is a tiny stub of a logging.Handler which just lets you
    direct LogRecords to a callback instead of a handler, eg:
    a method in some other class."""

    def __init__(self, callback: Callable[[logging.LogRecord], None], *a, **k):
        super().__init__(*a, **k)
        self.__callback = callback

    def emit(self, record: logging.LogRecord) -> None:
        self.__callback(record)


class LoggerFrame(tk.Frame):
    LOGGING_TIME_BUDGET = 0.02  # seconds per call to logging_event
    LOGGING_MAX_LINES = 5000  # lines of log to keep

    def __init__(self, *a, log_queue=None, **k) -> None:
        """`log_queue` is where log records come from, by default the one which the
        main program's logging is sent to."""
        super().__init__(*a, **k)
        self.count = 0
        self.text = tk.Text(self)
        self.scrollbar = ttk.Scrollbar(self, orient="vertical", command=self.text.yview)
        self.text["yscrollcommand"] = self.scrollbar.set
        self.text.grid(sticky=tk.NSEW)
        self.scrollbar.grid(row=0, column=1, sticky=tk.NS)
        self.progress_bars: dict[str, LabeledProgressbar] = {}

        # Start a QueueListener in its own thread,
        # logging_callback gets called for each record
        # received.
        # logging.handlers.QueueListener(logging_queue, _CallbackLoggingHandler(self.logging_callback)).start()

        # XXX tkinter doesn't seem happy with updating widgets from threads,
        # so here's an alternative, we poll from a tkinter event.
        self.logging_handler = logging.handlers.QueueListener(log_queue if log_queue is not None else logging_queue)

        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)
        # self.rowconfigure(1, weight=0)

        self.logging_event()

    def logging_event(self, ev=None):
        """Show whatever's arrived on the log queue.  Called regularly from the Tk
        event loop, it works through the queue for a short time (so the GUI stays
        responsive even if something is logging thousands of messages) and adds
        the text to the log in one go."""
        lines: list[str] = []
        deadline = time.monotonic() + self.LOGGING_TIME_BUDGET
        backlog = False
        while True:
            try:
                rec = self.logging_handler.dequeue(block=False)
            except queue.Empty:
                break
            if rec:
                self.logging_callback(rec, lines)
            if time.monotonic() > deadline:
                backlog = True
                break
        if lines:
            self.add_lines(lines)
        # if there's more to do, come back to it soon.
        self.after(10 if backlog else 100, self.logging_event)

    def logging_callback(self, record: logging.LogRecord, lines: Optional[list[str]] = None) -> None:
        """Handle one record: progress messages update their progress bar, anything
        else is added to `lines` (or shown straight away if no list is given)"""
        message = record.getMessage()
        if m := re.match(r"(.*): (\d+(?:\.\d+)?)(%|/\d+)", message):
            name, n1, n2 = m.groups()
            try:
                pbar = self.progress_bars[name]
            except KeyError:
                pbar = LabeledProgressbar(self, value=0)
                self.progress_bars[name] = pbar
                pbar.grid(sticky=tk.EW, row=len(self.progress_bars) + 1, column=0)
            if n2 == "%":
                percent = int(float(n1))
                pbar.progress_update(message, percent)
                if percent == 100:
                    self.after(5000, self.hide_event, name)
            else:
                pbar.progress_update(message)
        elif lines is not None:
            lines.append(message + "\n")
        else:
            self.add_lines([message + "\n"])

    def add_lines(self, lines: list[str]) -> None:
        # Only follow the end of the log if that's where we were looking already.
        following = self.text.yview()[1] >= 0.999
        self.text.insert(tk.END, "".join(lines))
        self.count += len(lines)

        # Don't let a long run's worth of log grow without limit.
        n_lines = int(self.text.index("end-1c").split(".")[0]) - 1  # (the last line is empty)
        if n_lines > self.LOGGING_MAX_LINES:
            self.text.delete("1.0", f"{n_lines - self.LOGGING_MAX_LINES + 1}.0")
        if following:
            self.text.see(tk.END)

    def hide_event(self, name):
        pbar = self.progress_bars.pop(name, None)
        if pbar:
            pbar.destroy()


# the first is for loading only, the rest are also what can be saved.
CONFIG_FILE_TYPES = [
    ("Config Files", ["*.ini", "*.json"]),
    (".INI Config File", "*.ini"),
    ("JSON Config File", "*.json"),
]


class MainWindow:
    """Arrange the parts of the main window, with the FrameWrapper on the top
    left and the ConfiguratorWrapper either beside it or under it depending on
    the window dimenstions"""

    graph = None
    graph_wrapper = None
    preview_frame = None
    config_wrapper = None
    config_changed = False
    config_filename = None

    def __init__(self, tk_parent: tk.Widget, config_filename: Optional[str] = None):
        self.tk_parent = tk_parent

        logger.debug("MainWindow.__init__")

        ButtonMenu(
            tk_parent,
            [
                ("New Config", self.config_new),
                ("Load Config", self.config_load),
                ("Save Config", self.config_save),
                ("Export Config", self.config_export),
                ("Tidy Graph", self.graph_tidy),
                ("Run", self.program_run),
                ("Exit", self.program_exit),
            ],
        )

        # +--------------------------------------------+
        # | ButtonMenu                                 |
        # +--------------+-----------------------------+
        # | tree_canvas  | config_wrapper.subframe     |
        # |              |                             |
        # |              +-----------------------------+
        # |              | config_wrapper.preview      |
        # |              |                             |
        # +--------------+-----------------------------+
        # | logger_subframe                            |
        # +--------------------------------------------+

        # top level is a vertical layout with logs along the bottom
        self.frame = ResizingFrame(tk_parent, orientation=ResizingFrame.Orientation.VERTICAL, bg="darkgrey")
        self.frame.grid(sticky=tk.NSEW)

        # next is a division between the tree view and the configuration
        self.main_subframe = ResizingFrame(self.frame, orientation=ResizingFrame.Orientation.AUTOMATIC, bg="darkgrey")
        self.frame.add_child(self.main_subframe, weight=4)

        self.logger_subframe = LoggerFrame(self.frame)
        # self.logger = self.logger_subframe.get_logger("")
        # self.logger_subframe_show_task()
        self.frame.add_child(self.logger_subframe)

        self.tree_canvas = FlippyCanvas(self.main_subframe, bg="skyblue")
        self.main_subframe.add_child(self.tree_canvas)

        self.config_wrapper_cache: dict[PipelineNode, ConfiguratorWrapper] = {}
        self.config_wrapper = SplashWrapper(self.main_subframe)
        self.main_subframe.add_child(self.config_wrapper.frame, weight=4)

        # On macOS, Quit (Cmd-Q or the application menu) runs ::tk::mac::Quit if it exists;
        # otherwise Tk calls exit() from inside mainloop(), and DuckDB's static destructors
        # then crash the interpreter. Quit the way the Exit button does instead.
        tk_parent.createcommand("::tk::mac::Quit", self.program_exit)

        logger.debug("MainWindow.__init__(%s)", repr(config_filename))
        if config_filename:
            self.config_load(config_filename)
        else:
            self.config_new()

    def logger_subframe_show_task(self):
        if self.logger_subframe.count > 0:
            self.frame.add_child(self.logger_subframe)
            self.frame.after(100, self.logger_subframe_hide_task)
        else:
            self.frame.after(100, self.logger_subframe_show_task)

    def logger_subframe_hide_task(self):
        if self.logger_subframe.count == 0:
            self.frame.remove_child(self.logger_subframe)
            self.frame.after(100, self.logger_subframe_show_task)
        else:
            self.frame.after(100, self.logger_subframe_hide_task)

    def _discard_graph(self):
        """Get rid of the current graph, and everything in the GUI that refers to it."""
        if self.graph_wrapper:
            self.graph_wrapper.destroy()
            self.graph_wrapper = None

        # Replace whatever is showing in the configuration pane with the splash,
        # then throw away the cached wrappers: they hold on to nodes (and the
        # database connection) of the old graph and have timers still running.
        splash = SplashWrapper(self.main_subframe)
        old_wrapper = self.config_wrapper
        if old_wrapper:
            self.main_subframe.replace_child(old_wrapper.frame, splash.frame)
        else:
            self.main_subframe.add_child(splash.frame, weight=4)
        self.config_wrapper = splash
        for wrapper in list(self.config_wrapper_cache.values()):
            wrapper.destroy()
        if old_wrapper and old_wrapper not in self.config_wrapper_cache.values():
            old_wrapper.destroy()
        self.config_wrapper_cache.clear()

        # Tell the old graph's worker to stop but don't wait for it, it'll
        # notice (and interrupt any running query) in its own time.
        if self.graph:
            self.graph.stop_thread(wait=False)
            self.graph = None

    def config_new(self):
        if self.config_changed:
            if not messagebox.askokcancel("Clear Config", "Clear unsaved config?"):
                return
        self.config_changed = False
        self.config_filename = None
        self.update_title()
        self._discard_graph()
        self.graph = PipelineGraph(preview_row_limit=preview_row_limit)
        self.graph_wrapper = GraphWrapper(self.tree_canvas, self.graph, self.node_select)
        self.graph_wrapper.add_new_node(select=False)

    def config_load(self, filename=None):
        if not filename:
            filename = ask_open_filename(file_types=CONFIG_FILE_TYPES)
        if not filename:
            return

        # read it before touching anything, so a bad file leaves the current config alone
        try:
            graph = read_config([filename])
        except Exception as exc:  # pylint: disable=broad-exception-caught
            logger.warning("Couldn't load %s: %s", filename, exc)
            messagebox.showerror("Load Config", f"Couldn't load {filename}:\n{exc}")
            return

        self.config_filename = filename
        self.config_changed = False
        self.update_title()

        self._discard_graph()
        self.graph = graph
        self.graph.preview_row_limit = preview_row_limit
        self.graph.reset()
        self.graph_wrapper = GraphWrapper(self.tree_canvas, self.graph, self.node_select)
        self.node_select(None)

    def config_save(self, filename=None):
        if not filename:
            filename = ask_saveas_filename(
                initial_file=self.config_filename,
                file_types=CONFIG_FILE_TYPES[1:],
            )
        if not filename:
            return
        if not filename.lower().endswith((".ini", ".json")):
            # no extension given: keep to the format of the file we have
            filename += ".json" if self.config_filename and is_json_filename(self.config_filename) else ".ini"
        write_config(self.graph, filename)
        self.config_filename = filename
        self.config_changed = False
        self.update_title()
        # Names may have changed on save so redraw the tree
        self.graph_wrapper.destroy()
        self.graph_wrapper = GraphWrapper(self.tree_canvas, self.graph, self.node_select)

    def config_export(self, filename=None):
        if not filename:
            if self.config_filename:
                initialfile = os.path.splitext(self.config_filename)[0] + ".dot"
            else:
                initialfile = None
            filename = ask_saveas_filename(initial_file=initialfile, file_types=[("Graphviz File", "*.dot")])
        if not filename:
            return
        export_config_graphviz(self.graph, filename)

    def graph_tidy(self):
        self.graph.tidy()
        self.graph_wrapper.destroy()
        self.graph_wrapper = GraphWrapper(self.tree_canvas, self.graph, self.node_select)

    def program_run(self):
        config = graph_to_config(self.graph)
        process = multiprocessing.Process(target=runwindow_subprocess_target, args=[config])
        process.start()

        def on_button(*_):
            if process.is_alive():
                process.terminate()
                logger.warning("Process Terminated")

        button = tk.Button(self.logger_subframe, text="Stop", command=on_button)
        button.grid(row=1, column=0, sticky=tk.EW)

        def on_poll(*_):
            if process.is_alive():
                self.frame.after(100, on_poll)
            else:
                logger.info("Finished")
                button.destroy()

        self.frame.after(100, on_poll)

    def program_exit(self):
        if not self.config_changed or messagebox.askokcancel("Exit", "Exit CountESS without saving config?"):
            if self.graph:
                # give the worker a moment to wind down so DuckDB isn't torn
                # down underneath it, but don't hang if it won't.
                self.graph.stop_thread(timeout=5)
            self.tk_parent.quit()

    def node_select(self, node):
        if not node:
            new_config_wrapper = SplashWrapper(self.main_subframe)
        else:
            try:
                new_config_wrapper = self.config_wrapper_cache[node]
            except KeyError:
                new_config_wrapper = ConfiguratorWrapper(self.main_subframe, node, self.graph.ddbc, self.node_changed)
                self.config_wrapper_cache[node] = new_config_wrapper

        if self.config_wrapper:
            self.main_subframe.replace_child(self.config_wrapper.frame, new_config_wrapper.frame)
        self.config_wrapper = new_config_wrapper
        new_config_wrapper.notify()

    def node_changed(self, node, recalc=False):
        logger.debug("MainWindow.node_changed(%s)", node)
        self.graph_wrapper.node_changed(node, recalc)
        self.config_changed = True
        self.update_title()

    def update_title(self):
        if self.config_filename:
            try:
                title = "CountESS: " + os.path.relpath(self.config_filename)
            except ValueError:
                # if relpath doesn't work, eg: on windows with multiple drive letters
                title = "CountESS: " + self.config_filename
        else:
            title = "CountESS " + VERSION

        if self.config_changed:
            title += " *"

        self.tk_parent.title(title)


class SplashScreen:
    def __init__(self, tk_root):
        bg = "skyblue"
        self.splash = tk.Toplevel(tk_root, bg=bg)
        self.splash.attributes("-type", "dialog")

        font = ("TkHeadingFont", 16, "bold")
        tk.Label(self.splash, text=f"CountESS {VERSION}", font=font, bg=bg).grid(padx=10, pady=10)
        tk.Label(self.splash, image=get_icon(tk_root, "countess"), bg=bg).grid(padx=10)

        self.splash.after(2500, self.destroy)

    def destroy(self):
        self.splash.destroy()


def make_root():
    try:
        import ttkthemes  # pylint: disable=C0415

        root = ttkthemes.ThemedTk()
        themes = set(root.get_themes())
        for t in ["clam", "aqua", "winnative"]:
            if t in themes:
                root.set_theme(t)
                break
    except ImportError:
        print(  # pylint: disable=bad-builtin
            "ttkthemes library not found.\nFor better GUI support, `pip install countess[gui]`."
        )

        root = tk.Tk()
        # XXX some kind of ttk style setup goes here as a fallback

    root.rowconfigure(0, weight=0)
    root.rowconfigure(1, weight=1)
    root.columnconfigure(0, weight=1)

    # Try to start off maximized, but this option doesn't exist
    # in all Tks or all platforms.
    try:
        root.state("zoomed")
    except tk.TclError:
        pass

    return root


def main() -> None:
    args = sys.argv[1:]
    global preview_row_limit  # pylint: disable=global-statement

    logging.getLogger().setLevel("INFO")

    try:
        options, args = getopt.getopt(args, "", ["help", "version", "preview=", "log="])
    except getopt.GetoptError as exc:
        logger.error(str(exc))
        sys.exit(1)

    for opt_key, opt_val in options:
        if opt_key == "--help":
            print(usage)  # pylint: disable=bad-builtin
            sys.exit(0)
        elif opt_key == "--version":
            print(f"CountESS {VERSION}")  # pylint: disable=bad-builtin
            sys.exit(0)
        elif opt_key == "--preview":
            if opt_val.upper() == "NONE":
                preview_row_limit = None
            elif re.match(r"\d+$", opt_val):
                preview_row_limit = int(opt_val) or None
            else:
                logger.warning("Bad --preview value: %s, expected a positive integer or 'NONE'", opt_val)
        elif opt_key == "--log":
            try:
                log_level: Union[int, str]
                if re.match(r"\d+$", opt_val):
                    log_level = int(opt_val)
                else:
                    log_level = str(opt_val).upper()
                logging.getLogger().setLevel(log_level)
                logger.warning("Log level set to %s", log_level)
            except ValueError:
                logger.error("Bad --log level: %s", opt_val)

    logger.info("Preview Limit set to %d", preview_row_limit)

    # set up a multiprocessing-compatible logging queue to bring all logging
    # messages back to the main process.
    logging.getLogger().addHandler(logging.handlers.QueueHandler(logging_queue))
    logging.getLogger().addHandler(logging.StreamHandler())

    root = make_root()
    MainWindow(root, args[0] if args else None)

    root.mainloop()


if __name__ == "__main__":
    main()  # pragma: no cover
