import bz2
import gzip
import lzma
import os
import threading
import time

import duckdb
import pytest

from countess.utils import compress
from countess.utils.compress import (
    COMPRESSORS,
    BlockCompressedFile,
    CompressingFilesystem,
    compressed_output,
    compressor_for,
    default_workers,
)

OPENERS = {".gz": gzip.open, ".bz2": bz2.open, ".xz": lzma.open}

# something which compresses (so is not the same bytes in and out) and has some noise in it
DATA = b"".join(b"%d,text number %d,%s\n" % (n, n * 7919 % 1000, b"ACGT" * (n % 7)) for n in range(20000))


def threads_of_ours():
    return [t for t in threading.enumerate() if t.name.startswith("countess-compress")]


def run_in_thread(function, timeout=20):
    """Run something which might hang, so that a hang is a failed test and not a stuck one"""
    outcome = {}

    def target():
        try:
            outcome["value"] = function()
        except BaseException as exc:  # pylint: disable=broad-exception-caught
            outcome["error"] = exc

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    thread.join(timeout)
    assert not thread.is_alive(), "it hung"
    if "error" in outcome:
        raise outcome["error"]
    return outcome.get("value")


def write_in_pieces(file, data, piece):
    for start in range(0, len(data), piece):
        file.write(data[start : start + piece])
    file.close()


# ---- the compressors, by file name


@pytest.mark.parametrize(
    "name,extension",
    [
        ("out.csv.gz", ".gz"),
        ("OUT.CSV.GZ", ".gz"),
        ("out.csv.bz2", ".bz2"),
        ("out.tsv.XZ", ".xz"),
        ("/some/where/x.txt.gz", ".gz"),
    ],
)
def test_the_compressor_is_chosen_by_the_extension(name, extension):
    assert compressor_for(name) is COMPRESSORS[extension]


@pytest.mark.parametrize("name", ["out.csv", "out", "out.gzip", "out.gz.csv", "out.zip", "xz"])
def test_other_names_are_not_compressed(name):
    assert compressor_for(name) is None


@pytest.mark.parametrize("extension", OPENERS)
def test_what_each_compressor_makes_is_a_whole_file_even_for_nothing(extension):
    for data in (b"", b"x", DATA[:1000]):
        assert OPENERS[extension](__import__("io").BytesIO(COMPRESSORS[extension](data))).read() == data


def test_there_are_never_more_workers_than_can_be_used(monkeypatch):
    assert 1 <= default_workers() <= compress.MAX_WORKERS
    monkeypatch.setattr(os, "sched_getaffinity", lambda pid: set(range(3)), raising=False)
    assert default_workers() == 3  # (as many as this process is allowed to use, on a shared server)
    monkeypatch.setattr(os, "sched_getaffinity", lambda pid: set(range(500)), raising=False)
    assert default_workers() == compress.MAX_WORKERS
    monkeypatch.setattr(os, "sched_getaffinity", lambda pid: set(), raising=False)
    assert default_workers() == 1


# ---- BlockCompressedFile


@pytest.mark.parametrize("extension", OPENERS)
@pytest.mark.parametrize("piece", [1, 7, 999, 1000, 4096, 10**9])
def test_what_is_written_in_pieces_comes_back_as_it_was_whatever_the_blocks(tmp_path, extension, piece):
    path = tmp_path / f"out{extension}"
    # (slow ones like bz2 and xz get fewer, bigger, blocks, so as not to take all day)
    file = BlockCompressedFile(str(path), COMPRESSORS[extension], block_size=40_000, workers=3)
    write_in_pieces(file, DATA if piece > 10 else DATA[:5000], piece)
    with OPENERS[extension](path) as fh:
        assert fh.read() == (DATA if piece > 10 else DATA[:5000])


def test_the_blocks_are_the_size_asked_for_and_the_last_has_what_is_left(tmp_path):
    sizes = []

    def compress_noting_size(block):
        sizes.append(len(block))
        return block

    file = BlockCompressedFile(str(tmp_path / "out"), compress_noting_size, block_size=1000, workers=1)
    write_in_pieces(file, bytes(range(256)) * 10, 333)  # 2560 bytes
    assert sizes == [1000, 1000, 560]
    assert (tmp_path / "out").read_bytes() == bytes(range(256)) * 10


def test_a_last_block_is_not_left_empty_when_the_size_is_an_exact_multiple(tmp_path):
    sizes = []
    file = BlockCompressedFile(str(tmp_path / "out"), lambda b: sizes.append(len(b)) or b, block_size=500, workers=2)
    write_in_pieces(file, b"x" * 1500, 500)
    assert sizes == [500, 500, 500]


@pytest.mark.parametrize("extension", OPENERS)
def test_a_file_with_nothing_in_it_is_still_a_whole_file(tmp_path, extension):
    path = tmp_path / f"empty{extension}"
    BlockCompressedFile(str(path), COMPRESSORS[extension], block_size=1000, workers=2).close()
    assert path.stat().st_size > 0
    with OPENERS[extension](path) as fh:
        assert fh.read() == b""


def test_the_blocks_are_written_in_order_even_if_a_later_one_is_ready_first(tmp_path):
    def slow_at_first(block):
        time.sleep(0.2 if block[:1] == b"0" else 0.0)  # (the first block takes the longest)
        return block

    file = BlockCompressedFile(str(tmp_path / "out"), slow_at_first, block_size=100, workers=4)
    data = b"".join(bytes([48 + n]) * 100 for n in range(8))  # blocks of 0s, 1s, 2s ...
    write_in_pieces(file, data, 100)
    assert (tmp_path / "out").read_bytes() == data


def test_a_writer_which_is_faster_than_the_compression_has_to_wait_for_it(tmp_path):
    release = threading.Event()
    started = []

    def held_up(block):
        started.append(len(block))
        release.wait(20)
        return block

    file = BlockCompressedFile(str(tmp_path / "out"), held_up, block_size=100, workers=2)  # (room for 4)
    writer = threading.Thread(target=write_in_pieces, args=(file, b"x" * 2000, 100), daemon=True)
    writer.start()
    time.sleep(0.5)
    assert writer.is_alive()  # (it's waiting, with 20 blocks to write and room for 4)
    assert len(started) <= 2  # (as there are two workers)
    release.set()
    writer.join(20)
    assert not writer.is_alive()
    assert (tmp_path / "out").read_bytes() == b"x" * 2000


def test_it_cannot_be_written_to_after_it_is_closed(tmp_path):
    file = BlockCompressedFile(str(tmp_path / "out"), COMPRESSORS[".gz"], block_size=100, workers=1)
    file.write(b"abc")
    file.close()
    file.close()  # (as many times as you like)
    with pytest.raises(ValueError):
        file.write(b"def")
    assert gzip.open(tmp_path / "out").read() == b"abc"


def test_it_knows_how_much_has_been_written(tmp_path):
    file = BlockCompressedFile(str(tmp_path / "out"), COMPRESSORS[".gz"], block_size=100, workers=1)
    file.write(b"x" * 250)
    file.write(b"y" * 10)
    assert file.tell() == 260
    file.close()


def test_what_is_wrong_with_the_compression_is_raised_and_nothing_hangs(tmp_path):
    class Broken(Exception):
        pass

    def compress_badly(block):
        if block[:1] == b"2":
            raise Broken("this block can't be compressed")
        return block

    file = BlockCompressedFile(str(tmp_path / "out"), compress_badly, block_size=100, workers=2)

    def write_everything():
        for n in range(200):  # (far more than will fit at once)
            file.write(bytes([48 + n % 10]) * 100)

    with pytest.raises(Broken):
        run_in_thread(write_everything)
    with pytest.raises(Broken):
        run_in_thread(file.close)
    assert file.closed
    assert not threads_of_ours()


def test_an_error_at_the_last_block_is_raised_by_close(tmp_path):
    def compress_the_last_one_badly(block):
        if len(block) < 100:
            raise OSError("no")
        return block

    file = BlockCompressedFile(str(tmp_path / "out"), compress_the_last_one_badly, block_size=100, workers=1)
    file.write(b"x" * 150)
    with pytest.raises(OSError):
        run_in_thread(file.close)
    assert not threads_of_ours()


def test_a_file_which_cannot_be_written_to_is_raised_and_nothing_hangs(tmp_path):
    class Full:
        def write(self, data):
            raise OSError("no space left on device")

        def close(self):
            pass

    file = BlockCompressedFile(str(tmp_path / "out"), lambda b: b, block_size=100, workers=2)
    file._out.close()  # pylint: disable=protected-access
    file._out = Full()  # pylint: disable=protected-access

    def write_everything():
        for _ in range(500):
            file.write(b"x" * 100)

    with pytest.raises(OSError, match="no space"):
        run_in_thread(write_everything)
    with pytest.raises(OSError, match="no space"):
        run_in_thread(file.close)
    assert not threads_of_ours()


def test_aborting_stops_everything_and_raises_nothing(tmp_path):
    file = BlockCompressedFile(str(tmp_path / "out"), COMPRESSORS[".gz"], block_size=100, workers=2)
    file.write(b"x" * 1000)
    file.abort()
    file.abort()
    assert file.closed
    assert not threads_of_ours()


# ---- compressed_output


def test_a_file_which_is_not_compressed_is_just_written_atomically(tmp_path):
    connection = duckdb.connect()
    target = tmp_path / "out.csv"
    with compressed_output(connection, target) as name:
        assert not name.startswith("countess-")
        assert not target.exists()
        open(name, "w", encoding="utf-8").write("all of it")
    assert target.read_text() == "all of it"
    assert os.listdir(tmp_path) == ["out.csv"]


@pytest.mark.parametrize("extension", OPENERS)
def test_duckdb_can_write_to_what_it_gives(tmp_path, extension):
    connection = duckdb.connect()
    connection.sql("create table t as select range as n, 'text ' || range as s from range(30000)")
    target = tmp_path / f"out.csv{extension}"
    with compressed_output(connection, target, block_size=50_000, workers=2) as name:
        assert "://" in name
        protocol = name.split("://")[0]
        assert connection.filesystem_is_registered(protocol)
        connection.execute(f"COPY t TO '{name}' (FORMAT CSV, COMPRESSION none, USE_TMP_FILE false)")
        assert not target.exists()  # (not until it's finished)
    assert not connection.filesystem_is_registered(protocol)
    assert os.listdir(tmp_path) == [target.name]
    with OPENERS[extension](target, "rt", encoding="utf-8") as fh:
        lines = fh.read().splitlines()
    assert len(lines) == 30001 and lines[:2] == ["n,s", "0,text 0"] and lines[-1] == "29999,text 29999"


def test_a_failure_leaves_nothing_where_there_was_nothing_and_cleans_up(tmp_path):
    connection = duckdb.connect()

    def explode(n: int) -> int:
        raise RuntimeError("boom")

    connection.create_function("explode", explode)
    connection.sql("create table t as select range as n from range(1000)")
    target = tmp_path / "out.csv.gz"
    with pytest.raises(duckdb.Error):
        with compressed_output(connection, target) as name:
            protocol = name.split("://")[0]
            connection.execute(f"COPY (SELECT explode(n) FROM t) TO '{name}' (FORMAT CSV, COMPRESSION none)")
    assert os.listdir(tmp_path) == []
    assert not connection.filesystem_is_registered(protocol)
    assert not threads_of_ours()


def test_a_failure_leaves_an_old_file_as_it_was(tmp_path):
    connection = duckdb.connect()
    target = tmp_path / "out.csv.xz"
    target.write_bytes(b"the old file")
    with pytest.raises(RuntimeError, match="stop"):
        with compressed_output(connection, target) as name:
            connection.execute(f"COPY (SELECT 1) TO '{name}' (FORMAT CSV, COMPRESSION none)")
            raise RuntimeError("stop")
    assert target.read_bytes() == b"the old file"
    assert os.listdir(tmp_path) == [target.name]
    assert not threads_of_ours()


def test_several_can_be_written_at_once_each_with_a_filesystem_of_its_own(tmp_path):
    connection = duckdb.connect()
    connection.sql("create table t as select range as n, 'text ' || range as s from range(20000)")
    errors = []

    def write(extension):
        try:
            cursor = connection.cursor()
            with compressed_output(cursor, tmp_path / f"out.csv{extension}", block_size=20_000, workers=2) as name:
                cursor.execute(f"COPY t TO '{name}' (FORMAT CSV, COMPRESSION none)")
        except BaseException as exc:  # pylint: disable=broad-exception-caught
            errors.append(exc)

    threads = [threading.Thread(target=write, args=(extension,)) for extension in list(OPENERS) * 2]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(60)
    assert not errors
    texts = {ext: OPENERS[ext](tmp_path / f"out.csv{ext}", "rt").read() for ext in OPENERS}
    assert len(set(texts.values())) == 1 and texts[".gz"].count("\n") == 20001
    assert sorted(os.listdir(tmp_path)) == sorted(f"out.csv{ext}" for ext in OPENERS)


def test_the_filesystem_is_only_for_writing_and_nothing_is_there_already():
    fs = CompressingFilesystem("countess-test", COMPRESSORS[".gz"], 100, 1)
    assert not fs.exists("countess-test:///anywhere")
    with pytest.raises(FileNotFoundError):
        fs.info("countess-test:///anywhere")
    with pytest.raises(NotImplementedError):
        fs.open("countess-test:///anywhere", "rb")
    assert (
        CompressingFilesystem._strip_protocol("countess-test:///a/b.gz") == "/a/b.gz"
    )  # pylint: disable=protected-access
    assert CompressingFilesystem._strip_protocol("relative/b.gz") == "relative/b.gz"  # pylint: disable=protected-access
