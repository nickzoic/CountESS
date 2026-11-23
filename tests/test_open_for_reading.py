import bz2
import gzip
import io
import lzma
import threading
import time
import types

import pytest

from countess.utils import compress
from countess.utils.compress import default_workers, open_for_reading

# about 2.7MB of what a FASTQ file looks like
PAYLOAD = b"".join(b"@read%d\nACGTACGTTTGACCA\n+\nIIIIIIIIIIIIIII\n" % n for n in range(60000))


def bz2_in_blocks(data):
    return bz2.compress(data, 1)  # (a block is 100kB: this is a lot of blocks)


def xz_in_blocks(data):
    if compress.lzma_mt is not None:
        return compress.lzma_mt.compress(data, preset=0, threads=4)  # (the small blocks of preset 0, so several)
    return lzma.compress(data, preset=0)  # pragma: no cover


FORMATS = {
    "bz2": types.SimpleNamespace(
        extension=".bz2", compress=bz2_in_blocks, standard=bz2.BZ2File, library="indexed_bzip2", empty=bz2.compress(b"")
    ),
    "xz": types.SimpleNamespace(
        extension=".xz", compress=xz_in_blocks, standard=lzma.LZMAFile, library="lzma_mt", empty=lzma.compress(b"")
    ),
}
EVERY_FORMAT = pytest.mark.parametrize("fmt", FORMATS.values(), ids=FORMATS.keys())

# what's raised for a file which isn't all there or isn't right depends on what's reading it
DAMAGE_ERRORS = (EOFError, OSError, RuntimeError, ValueError, lzma.LZMAError)


def threads_now():
    return {t.ident for t in threading.enumerate()}


def read_all(path):
    with open_for_reading(path) as fh:
        return fh.read()


@EVERY_FORMAT
def test_what_is_read_is_what_was_written(each_decoder, tmp_path, fmt):
    cases = {
        "one stream": fmt.compress(PAYLOAD),
        "two streams one after the other": fmt.compress(PAYLOAD[:1000000]) + fmt.compress(PAYLOAD[1000000:]),
        "trailing garbage after the end": fmt.compress(PAYLOAD) + b"GARBAGE-AFTER-THE-END",
    }
    for name, compressed in cases.items():
        path = tmp_path / f"{name}{fmt.extension}"
        path.write_bytes(compressed)
        assert read_all(path) == PAYLOAD, name


@EVERY_FORMAT
def test_a_stream_with_nothing_in_it_is_read_as_nothing(each_decoder, tmp_path, fmt):
    path = tmp_path / f"nothing{fmt.extension}"
    path.write_bytes(fmt.empty)
    assert read_all(path) == b""


@EVERY_FORMAT
def test_it_can_be_read_a_bit_at_a_time(each_decoder, tmp_path, fmt):
    path = tmp_path / f"in{fmt.extension}"
    path.write_bytes(fmt.compress(PAYLOAD))
    pieces = []
    with open_for_reading(path) as fh:
        while piece := fh.read(65537):
            pieces.append(piece)
    assert b"".join(pieces) == PAYLOAD
    assert len(pieces) > 10


@EVERY_FORMAT
def test_an_empty_file_is_an_error_not_a_file_with_nothing_in_it(each_decoder, tmp_path, fmt):
    """(It's what bz2 and lzma say, and it's what a copy which went wrong leaves.)"""
    path = tmp_path / f"empty{fmt.extension}"
    path.write_bytes(b"")
    with pytest.raises(EOFError):
        read_all(path)


@EVERY_FORMAT
@pytest.mark.parametrize("damage", ["truncated in the middle", "end chopped off", "one byte wrong", "not compressed"])
def test_a_damaged_file_is_an_error_not_a_short_file(each_decoder, tmp_path, fmt, damage):
    whole = fmt.compress(PAYLOAD)
    wrong = bytearray(whole)
    wrong[len(whole) // 2] ^= 0xFF
    path = tmp_path / f"damaged{fmt.extension}"
    path.write_bytes(
        {
            "truncated in the middle": whole[: len(whole) // 2],
            "end chopped off": whole[:-5],
            "one byte wrong": bytes(wrong),
            "not compressed": b"just some text, not compressed at all\n" * 50,
        }[damage]
    )
    with pytest.raises(DAMAGE_ERRORS):
        read_all(path)


@EVERY_FORMAT
def test_the_standard_library_is_what_is_used_if_the_others_are_not_installed(monkeypatch, tmp_path, fmt):
    monkeypatch.setattr(compress, "indexed_bzip2", None)
    monkeypatch.setattr(compress, "lzma_mt", None)
    path = tmp_path / f"in{fmt.extension}"
    path.write_bytes(fmt.compress(PAYLOAD))
    with open_for_reading(path) as fh:
        assert isinstance(fh, fmt.standard)
        assert fh.read() == PAYLOAD


@EVERY_FORMAT
def test_the_fast_decoder_is_what_is_used_if_it_is_installed(tmp_path, fmt):
    if getattr(compress, fmt.library) is None:
        pytest.skip(f"{fmt.library} is not installed")
    path = tmp_path / f"in{fmt.extension}"
    path.write_bytes(fmt.compress(PAYLOAD))
    with open_for_reading(path) as fh:
        assert not isinstance(fh, fmt.standard)
        assert fh.read() == PAYLOAD


def test_the_number_of_workers_is_passed_to_the_fast_decoders(monkeypatch, tmp_path):
    """(With stand-ins for them, so this doesn't depend on them being installed.)"""
    calls = []

    def fake_bzip2_open(path, parallelization=1):
        calls.append(("bz2", path, parallelization))
        return io.BytesIO(b"")

    def fake_xz_open(path, mode="rb", threads=1):
        calls.append(("xz", str(path), mode, threads))
        return io.BytesIO(b"")

    monkeypatch.setattr(compress, "indexed_bzip2", types.SimpleNamespace(open=fake_bzip2_open))
    monkeypatch.setattr(compress, "lzma_mt", types.SimpleNamespace(open=fake_xz_open))
    (tmp_path / "a.bz2").write_bytes(b"not empty")
    (tmp_path / "a.xz").write_bytes(b"not empty")

    open_for_reading(tmp_path / "a.bz2", workers=3)
    open_for_reading(tmp_path / "a.xz", workers=5)
    open_for_reading(tmp_path / "a.bz2")
    open_for_reading(tmp_path / "a.xz")
    assert calls == [
        ("bz2", str(tmp_path / "a.bz2"), 3),
        ("xz", str(tmp_path / "a.xz"), "rb", 5),
        ("bz2", str(tmp_path / "a.bz2"), default_workers()),
        ("xz", str(tmp_path / "a.xz"), "rb", default_workers()),
    ]


def test_the_extension_can_be_in_any_case(each_decoder, tmp_path):
    for name, compressed in (
        ("IN.BZ2", bz2.compress(PAYLOAD)),
        ("IN.XZ", lzma.compress(PAYLOAD)),
        ("IN.GZ", gzip.compress(PAYLOAD)),
        ("in.Bz2", bz2.compress(PAYLOAD)),
    ):
        (tmp_path / name).write_bytes(compressed)
        assert read_all(tmp_path / name) == PAYLOAD, name


def test_gzipped_and_plain_files_are_read_as_bytes(tmp_path):
    (tmp_path / "in.gz").write_bytes(gzip.compress(PAYLOAD))
    (tmp_path / "in.fastq").write_bytes(PAYLOAD)
    assert read_all(tmp_path / "in.gz") == PAYLOAD
    assert read_all(tmp_path / "in.fastq") == PAYLOAD
    assert read_all(str(tmp_path / "in.fastq")) == PAYLOAD  # (a name, as well as a path)


@EVERY_FORMAT
def test_no_threads_are_left_running(each_decoder, tmp_path, fmt):
    path = tmp_path / f"in{fmt.extension}"
    path.write_bytes(fmt.compress(PAYLOAD))
    before = threads_now()
    with open_for_reading(path, workers=3) as fh:
        fh.read(100000)  # (and not all of it, which is what a limit on the rows does)
    time.sleep(0.2)
    assert threads_now() <= before
