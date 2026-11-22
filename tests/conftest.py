import gc
import os

import duckdb
import pytest

# DuckDB uses as many threads as the computer has cores.  That makes no difference to most things, but it does to
# some: how a write which fails part way through is left, and the order which rows come out in when nothing says
# which.  Those can't be seen on a computer with one core, and a test run there passes where one on a
# computer with several fails, as the checks which run on GitHub do.  So
#
#     COUNTESS_TEST_DUCKDB_THREADS=4 pytest
#
# runs the tests as if on a computer with four cores, wherever they are.
if os.environ.get("COUNTESS_TEST_DUCKDB_THREADS"):
    _real_connect = duckdb.connect

    def _connect_with_threads(*args, **kwargs):
        connection = _real_connect(*args, **kwargs)
        connection.execute(f"SET threads = {int(os.environ['COUNTESS_TEST_DUCKDB_THREADS'])}")
        return connection

    duckdb.connect = _connect_with_threads


@pytest.fixture(autouse=True)
def collect_garbage_on_the_main_thread(monkeypatch):
    """Tests make a lot of Tk windows.  Once destroyed they're only reference-cycle
    garbage until Python's garbage collector next runs, and if that happens in a
    thread other than the one which made the window (say one of the threads the
    pipeline tests start, which make lots of objects) then Tcl aborts the whole
    test run with "Tcl_AsyncDelete: async handler deleted by the wrong thread".
    So after each test which made a Tk window, clear the garbage away here, in the
    main thread.  (Not after every test: collecting takes a good fraction of a second.)"""
    try:
        import tkinter  # pylint: disable=import-outside-toplevel
    except ImportError:  # no Tk, no problem
        yield
        return

    made_a_window = []
    original_init = tkinter.Tk.__init__

    def tracking_init(self, *args, **kwargs):
        made_a_window.append(True)
        original_init(self, *args, **kwargs)

    monkeypatch.setattr(tkinter.Tk, "__init__", tracking_init)
    yield
    if made_a_window:
        gc.collect()
