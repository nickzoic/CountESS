import logging
import re
import secrets
import threading
import time
import traceback
from dataclasses import dataclass
from enum import Enum
from typing import Iterable, Optional

import duckdb

from countess.core.plugins import BasePlugin, DuckdbPlugin, get_plugin_classes

logger = logging.getLogger(__name__)

# Threading model
# ---------------
# In the GUI there are two threads: the Tk thread, where the user edits nodes,
# links and parameters, and a worker thread (see PipelineGraph._thread_target)
# which recalculates whichever nodes have gone DIRTY.  The rules for keeping them
# out of each other's way are:
#
# * Neither ever waits for the other for long.  The Tk thread calls
#   PipelineGraph.update() whenever something changes, which only asks for a
#   recalculation (and interrupts a running query if it has gone out of date).
# * Each thread has its own DuckDB connection: PipelineGraph.ddbc is for the Tk
#   thread, PipelineGraph.thread_cursor is for the worker.
# * A change is never lost: PipelineNode.mark_dirty() bumps the node's
#   `generation`, and a run which started before that discards its result
#   (PipelineNode._settle) and leaves the node DIRTY to be run again.
# * Structural changes to the graph, and anything which walks it, take
#   GRAPH_LOCK.  It is held for a few set/list operations, never for a query.
# * Plugin parameters are not locked.  The Tk thread may edit them while a run
#   is reading them, but any such edit makes the node stale (see above), so
#   whatever the run read is thrown away.
GRAPH_LOCK = threading.RLock()


class PipelineNodeStatus(Enum):
    DIRTY = 0
    WORKING = 1
    RESULT = 2
    ERROR = 3


class PipelineNode:
    name: str
    uuid: str
    plugin: Optional[BasePlugin] = None
    position: Optional[tuple[float, float]] = None
    sort_column: int = 0
    sort_descending: bool = False
    notes: Optional[str] = None
    parent_nodes: set["PipelineNode"]
    child_nodes: set["PipelineNode"]
    status: PipelineNodeStatus = PipelineNodeStatus.DIRTY

    # XXX config is a cache for config loaded from the file
    # at config load time, if it is present it is loaded the
    # first time the plugin is prerun.
    config: list[tuple[str, str, str]] = []

    def __init__(
        self,
        name: str,
        uuid: Optional[str] = None,
        plugin: Optional[BasePlugin] = None,
        position: Optional[tuple[float, float]] = None,
        notes: Optional[str] = None,
        sort_column: int = 0,
        sort_descending: bool = False,
    ):
        self.name = name
        self.uuid = uuid or secrets.token_hex(16)
        self.plugin = plugin
        self.position = position or (0.5, 0.5)
        self.sort_column = sort_column
        self.sort_descending = sort_descending
        self.notes = notes
        self.parent_nodes = set()
        self.child_nodes = set()
        self.config: list[tuple[str, str, str]] = []
        self.table_name: str = "n_%x" % id(self)
        self.message: Optional[str] = None
        self.finished = threading.Event()

        # `generation` is bumped every time something that affects this node's
        # result changes (see mark_dirty).  run() remembers the generation it
        # started with and will only record its result if it is unchanged, so
        # an edit made while the node is running can never be overwritten by
        # the (now stale) result.  `_lock` makes the check-and-set atomic.
        self.generation: int = 0
        self._lock = threading.RLock()

    def set_config(self, key, value, base_dir):
        self.config.append((key, value, base_dir))

    def __hash__(self):
        return id(self)

    def is_ancestor_of(self, node: "PipelineNode") -> bool:
        return (self in node.parent_nodes) or any((self.is_ancestor_of(n) for n in node.parent_nodes))

    def is_descendant_of(self, node: "PipelineNode") -> bool:
        return (self in node.child_nodes) or any((self.is_descendant_of(n) for n in node.child_nodes))

    def load_config(self):
        assert isinstance(self.plugin, BasePlugin)
        # Called from both the GUI thread (when a node is first shown) and the
        # worker thread (first run): make sure the cached config is applied once.
        with self._lock:
            if self.config:
                config, self.config = self.config, []
                for key, val, base_dir in config:
                    try:
                        self.plugin.set_parameter(key, val, base_dir)
                        logger.debug("Parameter %s=%s", key, val)
                    except KeyError:
                        logger.warning("Parameter %s=%s Not Found", key, val)
                    except ValueError:
                        logger.warning("Parameter %s=%s Not Valid", key, val)

    def run(self, ddbc, row_limit: Optional[int] = None) -> bool:
        """Recalculate this node.  Returns True if the node was left DIRTY
        and needs to be run again (because it was changed while running, or
        the query was interrupted), False if it has settled."""
        logger.debug("PipelineNode.run %s", self.uuid)
        if not isinstance(self.plugin, DuckdbPlugin):
            return False
        self.load_config()

        # Note the generation and set the node to WORKING atomically.  Any
        # change which happens from here on bumps the generation (and sets
        # the node DIRTY), and _settle() will refuse to overwrite that.
        with self._lock:
            generation = self.generation
            self.status = PipelineNodeStatus.WORKING

        try:
            if ddbc and self.table_name:
                ddbc.sql(f"DROP TABLE IF EXISTS {self.table_name}")

            sources = {pn.name: ddbc.table(pn.table_name) for pn in list(self.parent_nodes) if pn.table_name}
            self.plugin.prepare_multi(ddbc, sources)
            logger.info("running %s", self.name)
            result = self.plugin.execute_multi(ddbc, sources, row_limit)
            if result is not None:
                result.to_table(self.table_name)
                # (the table is already materialized so this count is cheap, whereas
                # len(result) on the relation would run the whole query again.)
                rows = len(ddbc.table(self.table_name))
                logger.debug("PipelineNode.run got result %d rows", rows)
                if rows:
                    return self._settle(generation, PipelineNodeStatus.RESULT, "")
            logger.debug("PipelineNode.run got no result")
            return self._settle(generation, PipelineNodeStatus.ERROR, "")
        except (duckdb.InterruptException, duckdb.TransactionException):
            # Interrupted because we're stale, or lost a race for the catalog
            # with a reader: either way the right thing to do is try again.
            logger.debug("Query interrupted: retrying")
            return self._settle(generation, None)
        except Exception as exc:  # pylint: disable=broad-exception-caught
            logger.warning("PipelineNode.run %s exception %s", self.uuid, repr(exc))
            return self._settle(generation, PipelineNodeStatus.ERROR, "\n".join(traceback.format_exception(exc)))

    def _settle(self, generation: int, status: Optional[PipelineNodeStatus], message: Optional[str] = None) -> bool:
        """Record the outcome of a run, unless the node changed while it was running.
        `status` of None means "retry".  Returns True if the node is DIRTY."""
        with self._lock:
            if self.generation != generation:
                # changed under our feet: the node is already DIRTY and
                # whatever we calculated (or whatever went wrong) is moot.
                return True
            if status is None:
                self.message = None
                self.status = PipelineNodeStatus.DIRTY
                return True
            self.message = message
            self.status = status
            return False

    def mark_dirty(self, _seen: Optional[set] = None):
        if _seen is None:
            _seen = set()
        if self in _seen:
            return
        _seen.add(self)
        with self._lock:
            self.generation += 1
            self.status = PipelineNodeStatus.DIRTY
        for child_node in list(self.child_nodes):
            child_node.mark_dirty(_seen)

    def add_parent(self, parent):
        if (not self.plugin or self.plugin.num_inputs) and (not parent.plugin or parent.plugin.num_outputs):
            with GRAPH_LOCK:
                self.parent_nodes.add(parent)
                parent.child_nodes.add(self)
            self.mark_dirty()

    def del_parent(self, parent):
        with GRAPH_LOCK:
            self.parent_nodes.discard(parent)
            parent.child_nodes.discard(self)
        self.mark_dirty()

    def configure_plugin(self, key, value, base_dir="."):
        self.plugin.set_parameter(key, value, base_dir)
        self.mark_dirty()

    def detach(self):
        self.mark_dirty()
        with GRAPH_LOCK:
            for parent_node in self.parent_nodes:
                parent_node.child_nodes.discard(self)
            for child_node in self.child_nodes:
                child_node.parent_nodes.discard(self)


@dataclass
class NodeProgress:
    """How far a node which is being calculated has got"""

    node: PipelineNode
    # a percentage, or None if it's running but DuckDB can't say how far it's got (yet)
    percent: Optional[float]


@dataclass
class GraphProgress:
    """What the background worker is up to, see PipelineGraph.get_progress()"""

    running: list[NodeProgress]
    waiting: int  # how many more nodes are waiting their turn

    @property
    def active(self) -> bool:
        return bool(self.running or self.waiting)


class PipelineGraph:
    def __init__(self, nodes: Optional[list[PipelineNode]] = None, preview_row_limit: int = 1000000):
        self.plugin_classes = get_plugin_classes()
        self.nodes = nodes or []
        self.ddbc = duckdb.connect()
        self.ddbc.sql("SET python_enable_replacements = false")
        self.ddbc.create_function("warning", self.sql_warning)
        self.preview_row_limit = preview_row_limit

        # self.ddbc is for the GUI thread only.  The background worker has a
        # cursor of its own, which is created here (on the creating thread) rather
        # than in the worker, so that nothing ever calls self.ddbc.cursor() while
        # the GUI is using self.ddbc.  The worker is the only thread which
        # executes queries on it; other threads may only call interrupt() or
        # query_progress() on it.
        self.thread_cursor: duckdb.DuckDBPyConnection = self.ddbc.cursor()
        self.thread_cursor.sql("set enable_progress_bar=true")
        self.thread_cursor.sql("set enable_progress_bar_print=false")

        # The worker thread runs only while there is work to do: update() asks
        # for a sweep and starts a worker if there isn't one running.
        self.thread: Optional[threading.Thread] = None
        self._state_lock = threading.Lock()
        self._pending = False  # a sweep has been requested
        self._worker_running = False
        self._abort = threading.Event()  # the current sweep is out of date
        self._quit = False
        self._current_node: Optional[PipelineNode] = None

    def reset(self):
        for node in self.nodes:
            node.mark_dirty()
        self.update()

    def __del__(self):
        try:
            self.stop_thread(wait=False)
        except Exception:  # pylint: disable=broad-exception-caught
            pass

    def sql_warning(self, message: str, value: Optional[str]) -> Optional[str]:
        logger.warning(message)
        return value

    def reset_node_name(self, node: PipelineNode):
        node_names_seen = set(n.name for n in self.nodes if n != node)
        while node.name in node_names_seen:
            num = 1
            if match := re.match(r"(.*?)\s+(\d+)$", node.name):
                node.name = match.group(1)
                num = int(match.group(2))
            node.name += f" {num + 1}"

    def add_node(self, node: PipelineNode):
        self.reset_node_name(node)
        with GRAPH_LOCK:
            self.nodes.append(node)

    def del_node(self, node: PipelineNode):
        node.detach()
        self.remove_node(node)

    def remove_node(self, node: PipelineNode):
        """Remove a node from the graph, without touching its links."""
        with GRAPH_LOCK:
            self.nodes.remove(node)

    def find_node(self, name: str) -> Optional[PipelineNode]:
        for node in self.nodes:
            if node.name == name:
                return node
        return None

    def find_node_by_uuid(self, uuid: str) -> Optional[PipelineNode]:
        for node in self.nodes:
            if node.uuid == uuid:
                return node
        raise KeyError(uuid)

    def traverse_nodes(self) -> Iterable[PipelineNode]:
        """Nodes in dependency order (parents before children).  The order is
        worked out all at once, under the lock, so it is safe to iterate over
        while another thread is editing the graph.  Nodes which can't be
        ordered (because of a cycle) are left out rather than looping forever."""
        with GRAPH_LOCK:
            nodes = list(self.nodes)
            found_nodes = set(node for node in nodes if not node.parent_nodes)
            order = sorted(found_nodes, key=lambda n: n.uuid)

            progress = True
            while len(found_nodes) < len(nodes) and progress:
                progress = False
                for node in sorted(nodes, key=lambda n: n.uuid):
                    if node not in found_nodes and node.parent_nodes.issubset(found_nodes):
                        order.append(node)
                        found_nodes.add(node)
                        progress = True

        yield from order

    def traverse_nodes_backwards(self) -> Iterable[PipelineNode]:
        with GRAPH_LOCK:
            nodes = list(self.nodes)
            found_nodes = set(node for node in nodes if not node.child_nodes)
            order = list(found_nodes)

            progress = True
            while len(found_nodes) < len(nodes) and progress:
                progress = False
                for node in nodes:
                    if node not in found_nodes and node.child_nodes.issubset(found_nodes):
                        order.append(node)
                        found_nodes.add(node)
                        progress = True

        yield from order

    def run_node(self, node: PipelineNode):
        return node.run(self.ddbc)

    def stop_thread(self, wait: bool = True, timeout: Optional[float] = None):
        """Ask the worker to stop and (optionally) wait for it to do so.  Only
        needed when the graph is being thrown away: for ordinary changes call
        update(), which never blocks."""
        logger.debug("PipelineGraph.stop_thread")
        with self._state_lock:
            self._quit = True
            self._abort.set()
            thread = self.thread
        self._interrupt_current(force=True)
        if wait and thread and thread.is_alive() and thread is not threading.current_thread():
            thread.join(timeout)

    def start_thread(self):
        """Ensure that a sweep happens.  Equivalent to update()."""
        self.update()

    def update(self):
        """Something has changed: make sure the dirty nodes get recalculated.
        This never blocks, so it is safe to call from the GUI thread.  If a
        sweep is in progress it is abandoned (and an interrupt sent, if the
        node it is working on is the one which has gone out of date) and a
        fresh one is started as soon as possible."""
        logger.debug("PipelineGraph.update")
        with self._state_lock:
            if self._quit:
                return
            self._pending = True
            self._abort.set()
            if not self._worker_running:
                self._worker_running = True
                self.thread = threading.Thread(target=self._thread_target, name="PipelineGraph", daemon=True)
                self.thread.start()
        self._interrupt_current()

    def _interrupt_current(self, force: bool = False):
        # Interrupting is only an optimization: a node which is changed while
        # running is discarded anyway (see PipelineNode._settle), but there's
        # no point waiting for a long query whose result we'll throw away.
        # Don't kill queries which are still valid though.
        node = self._current_node
        if node is not None and (force or node.status == PipelineNodeStatus.DIRTY):
            try:
                self.thread_cursor.interrupt()
            except duckdb.Error:
                pass

    def get_progress(self) -> GraphProgress:
        """What the worker is up to: the node it is running, with how far through its
        query it has got, and how many others are waiting.  Safe to call from any thread
        and cheap enough to call a few times a second to drive a progress bar."""
        running: list[NodeProgress] = []
        waiting = 0
        if not self._worker_running:
            return GraphProgress(running, waiting)

        for node in self.traverse_nodes():
            if node.status == PipelineNodeStatus.WORKING:
                running.append(NodeProgress(node, self._query_percent(node)))
            elif node.status == PipelineNodeStatus.DIRTY and isinstance(node.plugin, DuckdbPlugin):
                waiting += 1
        return GraphProgress(running, waiting)

    def _query_percent(self, node: PipelineNode) -> Optional[float]:
        # The worker's cursor is only ever running the current node's query.
        # DuckDB lets another thread ask how far a query has got.
        if node is not self._current_node or not isinstance(node.plugin, DuckdbPlugin):
            return None
        try:
            percent = node.plugin.query_progress(self.thread_cursor)
        except duckdb.Error:
            return None
        # DuckDB says -1 when it doesn't know: before the query starts, between one
        # query and the next, and when the node has just finished.
        return float(percent) if percent is not None and 0 <= percent <= 100 else None

    def _thread_target(self):
        """Worker: sweep for dirty nodes and update their results, until
        there's nothing left to do."""
        logger.debug("Pipelinegraph._thread_target starting")
        while True:
            with self._state_lock:
                if self._quit or not self._pending:
                    self._worker_running = False
                    logger.debug("Pipelinegraph._thread_target finished")
                    return
                self._pending = False
                self._abort.clear()

            try:
                retry = self._sweep()
            except Exception:  # pylint: disable=broad-exception-caught
                logger.exception("Pipelinegraph._thread_target sweep failed")
                retry = False

            if retry:
                # something was interrupted or lost a race: back off briefly and go again.
                self._abort.wait(0.1)
                with self._state_lock:
                    self._pending = True

    def _sweep(self) -> bool:
        """One pass over the graph, recalculating dirty nodes.  Returns True if
        any node needs another go."""
        retry = False
        blocked: set[PipelineNode] = set()
        try:
            for node in self.traverse_nodes():
                if self._quit or self._abort.is_set():
                    logger.debug("Pipelinegraph._sweep abandoned")
                    return False
                if node.status != PipelineNodeStatus.DIRTY:
                    logger.debug("Pipelinegraph._sweep %s is clean", node.uuid)
                    continue
                if any(p in blocked for p in list(node.parent_nodes)):
                    # an input isn't ready yet, we'll get to this on the next pass.
                    blocked.add(node)
                    continue
                logger.debug("Pipelinegraph._sweep recalculating %s", node.uuid)
                self._current_node = node
                try:
                    if node.run(self.thread_cursor, self.preview_row_limit):
                        blocked.add(node)
                        retry = True
                finally:
                    self._current_node = None
        except duckdb.InterruptException:
            logger.debug("Pipelinegraph._sweep interrupted")
            retry = True
        return retry

    def simplify(self):
        for node in self.traverse_nodes():
            if len(node.child_nodes) != 1:
                pass

    def run_group(self, node, cursor):
        logger.debug(
            "%s: Node %s cursor %s table_name %s", threading.current_thread(), node.name, cursor, node.table_name
        )
        cursor.sql("set enable_progress_bar=true")
        cursor.sql("set enable_progress_bar_print=false")

        def _r(pnode):
            if len(pnode.child_nodes) != 1:
                logger.debug("%s: Group '%s' Waiting on '%s'", threading.current_thread(), node.name, pnode.name)
                pnode.finished.wait()
                return cursor.table(pnode.table_name)
            else:
                logger.debug("%s: Group '%s' Preparing '%s'", threading.current_thread(), node.name, pnode.name)
                pnode.load_config()
                sources = {n.name: _r(n) for n in pnode.parent_nodes}
                pnode.plugin.prepare_multi(cursor, sources)
                result = pnode.plugin.execute_multi(cursor, sources)
                if result is None:
                    return None
                result.to_view(pnode.table_name)
                return cursor.table(pnode.table_name)

        node.load_config()
        sources = {n.name: _r(n) for n in node.parent_nodes}
        node.plugin.prepare_multi(cursor, sources)
        result = node.plugin.execute_multi(cursor, sources)
        if result:
            logger.debug("%s: saving table %s", threading.current_thread(), node.table_name)
            result.to_table(node.table_name)
        node.finished.set()
        logger.info("%s: 100%%", node.name)
        logger.debug("%s: finished node '%s'", threading.current_thread(), node.name)

    def run(self):
        # We let DuckDB work out how to run everything in parallel.
        # The exception is any nodes with more than one child_node ...
        # rather than have each child_node read the view over again,
        # we materialize the view into a table.

        start_time = time.time()
        logger.info("Starting: %s", start_time)

        threads = []
        cursors = {}

        with duckdb.connect() as ddbc:
            for node in self.traverse_nodes():
                if len(node.child_nodes) != 1:
                    node.finished.clear()
                    cursor = ddbc.cursor()
                    threads.append(threading.Thread(target=self.run_group, args=(node, cursor)))
                    cursors[node.name] = cursor

            for thread in threads:
                thread.start()

            while any(thread.is_alive() for thread in threads):
                for node_name, cursor in cursors.items():
                    try:
                        qp = cursor.query_progress()
                        if qp == 0:
                            logger.info("%s: 0/0", node_name)
                        elif qp >= 0:
                            logger.info("%s: %.1f%%", node_name, qp)
                    except duckdb.ConnectionException:
                        pass
                time.sleep(1)

            for thread in threads:
                thread.join()

        finish_time = time.time()
        logger.info("Finished: %s, elapsed time: %d", finish_time, finish_time - start_time)

    def reset_node_names(self):
        node_names_seen = set()
        for node in self.traverse_nodes():
            while node.name in node_names_seen:
                num = 0
                if match := re.match(r"(.*?)\s+(\d+)$", node.name):
                    node.name = match.group(1)
                    num = int(match.group(2))
                node.name += f" {num + 1}"
            node_names_seen.add(node.name)

    def tidy(self):
        """Tidies the graph (sets all the node positions)"""

        # XXX This is very arbitrary and not particularly efficient.
        # Some kind of FDP-like algorithm might be nice.
        # Especially if it could include node/line collisions.
        # See #24

        nodes = list(self.traverse_nodes())

        # first calculate a stratum for each node.

        stratum = {}
        for node in nodes:
            if not node.parent_nodes:
                stratum[node] = 0
            else:
                stratum[node] = max(stratum[n] for n in node.parent_nodes) + 1

        # shuffle nodes back down to avoid really long connections.

        for node in nodes[::-1]:
            if node.child_nodes:
                if len(node.parent_nodes) == 0:
                    stratum[node] = min(stratum[n] for n in node.child_nodes) - 1
                else:
                    stratum[node] = (
                        min(stratum[n] for n in node.child_nodes) + max(stratum[n] for n in node.parent_nodes)
                    ) // 2

        max_stratum = max(stratum.values())

        position = {}
        for s in range(0, max_stratum + 1):
            # now sort all the nodes by the average position of their parents,
            # to try and stop them forming a big tangle.  The current position
            # is included as a "tie breaker" and to keep some memory of the user's
            # preference for position (eg: ordering of branches)

            def avg_pos_parents(node):
                return sum(position[p] for p in node.parent_nodes) / len(node.parent_nodes)

            snodes = [
                (
                    avg_pos_parents(node) if node.parent_nodes else 0.5,
                    node.position[1],
                    n,
                )
                for n, node in enumerate(nodes)
                if stratum[node] == s
            ]
            snodes.sort()

            # Assign node positions with the stratums placed
            # evenly and the nodes spaced evenly per stratum.

            y = (s + 0.5) / (max_stratum + 1)
            for p, (_, _, n) in enumerate(snodes):
                x = (p + 0.5) / len(snodes)
                nodes[n].position = (y, x)
                position[nodes[n]] = x
