"""Writing compressed files as they are generated, in blocks, using several threads.

DuckDB can write a CSV file in blocks, to a plain file or to a Python file-like object, but it only knows how to
compress as gzip (and not very quickly: one thread, whatever the number of cores).  Writing it uncompressed and
compressing that afterwards would mean storing, and reading back, all of the uncompressed file, which can be big
and slow if the filesystem is a shared one.

Instead, `compressed_output` gives DuckDB somewhere to write to which cuts what it's given into blocks, compresses
each block, a number of them at a time in different threads, and writes the compressed blocks to the file in order.
Each block is compressed as a complete gzip member, bzip2 stream or xz stream of its own.  Those all concatenate to
a file which any decompressor decompresses to the original: gzip, bzip2 and xz say so in their documentation, and
Python, pandas and DuckDB read them that way, so what's decompressed is exactly what was written, whatever the
block size.  The compression is a little less good than for one stream, as each block starts again.

Memory used is a few blocks per thread however big the file is.  zlib, bz2 and lzma all let go of the GIL while
they're compressing, so the threads do run at the same time.

For reading, `open_for_reading` gives a file to read the uncompressed bytes from, using faster decoders for .bz2 and
.xz files if they're installed (see the "fast" extra)."""

import bz2
import contextlib
import gzip
import io
import logging
import lzma
import os
import queue
import secrets
import threading
import zlib
from concurrent.futures import ThreadPoolExecutor
from typing import IO, Callable, Iterator, Optional, Union, cast

import fsspec  # type: ignore[import-untyped]
from duckdb import DuckDBPyConnection

from countess.utils.files import atomic_output

# Decoders which can use several cores, and which for bzip2 is a good deal faster even on one.  Neither is needed:
# without them the standard library's bz2 and lzma are used.
try:
    import indexed_bzip2  # type: ignore[import-not-found,import-untyped]
except ImportError:  # pragma: no cover
    indexed_bzip2 = None  # type: ignore[assignment]

try:
    import lzma_mt  # type: ignore[import-not-found,import-untyped]
except ImportError:  # pragma: no cover
    lzma_mt = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

# Each block is as big as this before it is compressed.  Bigger blocks compress a little better (about 4% more
# for 16MB blocks than for one stream, in a CSV file of the kind CountESS writes) and use more memory.
DEFAULT_BLOCK_SIZE = 8 * 1024 * 1024

# at most this many compressors at once, however many cores there are (xz can want 100MB for each)
MAX_WORKERS = 8

GZIP_LEVEL = 6
BZIP2_LEVEL = 9


def _gzip_member(data: bytes) -> bytes:
    compressor = zlib.compressobj(GZIP_LEVEL, zlib.DEFLATED, 31)  # (31 = 16 + 15: with a gzip header and trailer)
    return compressor.compress(data) + compressor.flush()


# What to compress a block with, for each file extension.  Whatever the block is, even if it's empty, the result is
# a complete gzip file, bzip2 file or xz file.
COMPRESSORS: dict[str, Callable[[bytes], bytes]] = {
    ".gz": _gzip_member,
    ".bz2": lambda data: bz2.compress(data, BZIP2_LEVEL),
    ".xz": lzma.compress,
}


def compressor_for(filename: Union[str, "os.PathLike[str]"]) -> Optional[Callable[[bytes], bytes]]:
    """The function which compresses a block for a file of this name, or None if it isn't to be compressed"""
    name = str(filename).lower()
    for extension, compressor in COMPRESSORS.items():
        if name.endswith(extension):
            return compressor
    return None


def default_workers() -> int:
    """How many compressors to run at once: one for each core this process may use (which on a shared server may
    be fewer than the machine has), up to MAX_WORKERS."""
    try:
        cores = len(os.sched_getaffinity(0))  # type: ignore[attr-defined]
    except (AttributeError, OSError):  # pragma: no cover  (not on every platform)
        cores = os.cpu_count() or 1
    return max(1, min(cores, MAX_WORKERS))


def open_for_reading(filename: Union[str, "os.PathLike[str]"], workers: Optional[int] = None) -> IO[bytes]:
    """Open a file to read its bytes, decompressed if its name ends in .gz, .bz2 or .xz.

    A .bz2 file is read with indexed_bzip2 if that's installed, which is about 6 times faster than the standard
    library's bz2 on one core and uses `workers` cores (by default as many as `default_workers` says) for the many
    independent blocks in a bzip2 file.  A .xz file is read with lzma_mt if that's installed, which is a little
    faster on one core and can use `workers` cores if the file was made with several blocks (as `xz -T` does),
    which it isn't if it was made with just `xz`.  If they're not installed it's the standard library's modules,
    which use one core.  Whichever is used it reads the same bytes, and a damaged file is an error and not a short
    file.  (The other kinds of file are just opened, and not for text.)"""

    name = str(filename).lower()
    workers = workers or default_workers()

    if name.endswith(".bz2"):
        if indexed_bzip2 is None:
            return bz2.open(filename, "rb")
        # (bz2 says an empty file is incomplete, but indexed_bzip2 reads it as nothing, as it would an empty stream)
        if os.path.getsize(filename) == 0:
            raise EOFError("Compressed file ended before the end-of-stream marker was reached")
        logger.debug("reading %s with indexed_bzip2, %d workers", filename, workers)
        return indexed_bzip2.open(str(filename), parallelization=workers)

    if name.endswith(".xz"):
        if lzma_mt is None:
            return lzma.open(filename, "rb")
        logger.debug("reading %s with lzma_mt, %d workers", filename, workers)
        return cast(IO[bytes], lzma_mt.open(filename, "rb", threads=workers))  # (mode "rb" is bytes)

    if name.endswith(".gz"):
        return cast(IO[bytes], gzip.open(filename, "rb"))  # (it is one, but typeshed doesn't say so)

    # (it's for the caller to close, not for a `with` here)
    return open(filename, "rb")  # pylint: disable=consider-using-with


class BlockCompressedFile(io.RawIOBase):
    """A file which compresses what's written to it, a block at a time with a number of blocks at once.

    What's written, in whatever sized pieces, is cut into blocks of `block_size` and each is compressed by
    `compress` in a thread of its own.  A thread of its own writes what they make to `path`, in order.  There is
    never more than `2 * workers` blocks being compressed or waiting to be written, so a writer which is quicker
    than the compression is made to wait for it.

    If anything goes wrong (`compress` fails, or `path` can't be written to) the next `write` or the `close` raises
    the exception, and nothing hangs."""

    def __init__(self, path: str, compress: Callable[[bytes], bytes], block_size: int, workers: int):
        super().__init__()
        self._compress = compress
        self._block_size = block_size
        self._buffer = bytearray()
        self._blocks = 0
        self._written = 0
        self._error: Optional[BaseException] = None
        self._lock = threading.Lock()
        self._slots = threading.BoundedSemaphore(2 * workers)
        self._results: queue.Queue = queue.Queue()
        # (this file is closed by `_finish`, not by a `with`)
        self._out = open(path, "wb")  # pylint: disable=consider-using-with
        self._pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="countess-compress")
        self._writer = threading.Thread(target=self._write_results, name="countess-compress-writer", daemon=True)
        self._finished = False
        self._writer.start()

    def writable(self) -> bool:
        return True

    def tell(self) -> int:
        """How much has been written, uncompressed"""
        return self._written

    def flush(self) -> None:
        # (There's nothing useful to do: a block is only compressed once it's all there.)
        pass

    def write(self, data) -> int:  # type: ignore[override]
        with self._lock:
            self._raise_any_error()
            if self.closed:  # pylint: disable=using-constant-test
                raise ValueError("write to closed file")
            self._buffer += data
            self._written += len(data)
            while len(self._buffer) >= self._block_size:
                block = bytes(self._buffer[: self._block_size])
                del self._buffer[: self._block_size]
                self._submit(block)
        return len(data)

    def close(self) -> None:
        if self.closed:  # pylint: disable=using-constant-test
            return
        try:
            with self._lock:
                if self._error is None:
                    last_block = bytes(self._buffer)
                    self._buffer.clear()
                    # (a file with nothing in it is still a whole gzip or bzip2 or xz file, with one empty block)
                    if last_block or self._blocks == 0:
                        self._submit(last_block)
        except BaseException as exc:  # pylint: disable=broad-exception-caught
            self._error = self._error or exc
        finally:
            self._finish()
            super().close()
        self._raise_any_error()

    def abort(self) -> None:
        """Stop, whatever has been written so far, without raising anything (for when something else has failed)"""
        self._error = self._error or RuntimeError("aborted")
        with contextlib.suppress(BaseException):
            self.close()

    def _raise_any_error(self) -> None:
        if self._error is not None:
            raise self._error

    def _submit(self, block: bytes) -> None:
        # (the slot is released by `_write_results`, or just below if there's an error, not by a `with`)
        self._slots.acquire()  # pylint: disable=consider-using-with
        try:
            self._raise_any_error()
            self._results.put(self._pool.submit(self._compress, block))
            self._blocks += 1
        except BaseException:
            self._slots.release()
            raise

    def _write_results(self) -> None:
        while True:
            future = self._results.get()
            if future is None:
                return
            try:
                if self._error is None:
                    self._out.write(future.result())
                else:
                    future.cancel()  # (nothing more is wanted, just emptying the queue so nobody waits for it)
            except BaseException as exc:  # pylint: disable=broad-exception-caught
                self._error = self._error or exc
            finally:
                self._slots.release()

    def _finish(self) -> None:
        if self._finished:
            return
        self._finished = True
        self._results.put(None)
        self._writer.join()
        self._pool.shutdown(wait=True)
        try:
            self._out.close()
        except OSError as exc:
            self._error = self._error or exc


class CompressingFilesystem(fsspec.AbstractFileSystem):  # pylint: disable=abstract-method
    """A filesystem for DuckDB to write one file to, which is compressed as it's written.  (DuckDB sends a COPY
    to a filesystem it has been given, if the filename starts with the filesystem's protocol.)  It only does that:
    the file isn't there to be read back or looked at, and as the name is only used once there's nothing there
    already."""

    cachable = False  # (each of these is for one file, and has to be a new one)

    def __init__(self, protocol: str, compress: Callable[[bytes], bytes], block_size: int, workers: int):
        super().__init__()
        self.protocol = protocol
        self._compress = compress
        self._block_size = block_size
        self._workers = workers
        self._files: list[BlockCompressedFile] = []

    @classmethod
    def _strip_protocol(cls, path):
        path = str(path)
        return path.split("://", 1)[1] if "://" in path else path

    def info(self, path, **kwargs):
        raise FileNotFoundError(path)

    def exists(self, path, **kwargs):
        return False

    def _open(self, path, mode="rb", block_size=None, autocommit=True, cache_options=None, **kwargs):
        if "w" not in mode or "r" in mode or "a" in mode:
            raise NotImplementedError(f"{self.protocol} can only be written to, not opened with mode {mode!r}")
        file = BlockCompressedFile(self._strip_protocol(path), self._compress, self._block_size, self._workers)
        self._files.append(file)
        return file

    def finish(self) -> None:
        """Make sure every file which was opened is finished and all of it is written, raising any error there was"""
        for file in self._files:
            file.close()

    def abort(self) -> None:
        for file in self._files:
            file.abort()


@contextlib.contextmanager
def compressed_output(
    ddbc: DuckDBPyConnection,
    filename: Union[str, "os.PathLike[str]"],
    block_size: Optional[int] = None,
    workers: Optional[int] = None,
) -> Iterator[str]:
    """For writing a file with DuckDB (`COPY ... TO`) which is compressed according to its extension (.gz, .bz2 or
    .xz, otherwise not at all), without the uncompressed file being written anywhere, and so that it's all there
    or it isn't there at all (see `atomic_output`).

    Gives the name for DuckDB to write to.  DuckDB must be told not to compress it too, as it would go by the name.
    The file is finished, or if anything went wrong the temporary file is removed, when the `with` finishes."""

    compress = compressor_for(filename)
    if compress is None:
        with atomic_output(filename) as temp_filename:  # pylint: disable=contextmanager-generator-missing-cleanup
            yield temp_filename
        return

    protocol = f"countess-{secrets.token_hex(4)}"  # (a different one each time, as they can be at once)
    filesystem = CompressingFilesystem(
        protocol, compress, block_size or DEFAULT_BLOCK_SIZE, workers or default_workers()
    )
    ddbc.register_filesystem(filesystem)
    try:
        with atomic_output(filename) as temp_filename:  # pylint: disable=contextmanager-generator-missing-cleanup
            try:
                yield f"{protocol}://{temp_filename}"
                filesystem.finish()
            except BaseException:
                # (the temporary file has to be closed before it's removed, which some platforms mind about)
                filesystem.abort()
                raise
    finally:
        ddbc.unregister_filesystem(protocol)
