import bz2
import gzip
import lzma

import duckdb
import pytest

from countess.plugins.fastq import LoadFastaPlugin, LoadFastqPlugin
from countess.utils import compress

# name, sequence, qualities:  'I' is a quality of 40, '#' is 2, '5' is 20, '+' is 10
READS = [
    ("r1", "ACGT", "IIII"),  # average 40, lowest 40
    ("r2", "ACGT", "IIII"),
    ("r3", "TTTT", "####"),  # average 2, lowest 2
    ("r4", "GGGG", "II##"),  # average 21, lowest 2
    ("r5", "CCCC", "5555"),  # average 20, lowest 20
    ("r6", "AAAA", "I+II"),  # average 31.75, lowest 10
]


def fastq_text(reads=READS):
    return "".join(f"@{name}\n{seq}\n+\n{qual}\n" for name, seq, qual in reads)


@pytest.fixture(name="fastq")
def fixture_fastq(tmp_path):
    path = tmp_path / "reads.fastq"
    path.write_text(fastq_text(), encoding="utf-8")
    return path


def load(ddbc, *filenames, row_limit=None, **params):
    plugin = LoadFastqPlugin()
    for number, filename in enumerate(filenames):
        plugin.set_parameter(f"files.{number}.filename", str(filename))
    for key, value in params.items():
        plugin.set_parameter(key, value)
    return plugin.execute(ddbc, None, row_limit)


def test_counts_of_each_sequence(ddbc, fastq, assert_rows):
    output = load(ddbc, fastq, min_avg_quality=0)  # (every read)
    assert output.columns == ["sequence", "count"]
    assert_rows(output, [("ACGT", 2), ("TTTT", 1), ("GGGG", 1), ("CCCC", 1), ("AAAA", 1)])


def test_reads_of_poor_average_quality_are_left_out(ddbc, fastq, assert_rows):
    # by default the average must be at least 10: that's only r3, and the rest have 20 and more
    assert_rows(load(ddbc, fastq), [("ACGT", 2), ("GGGG", 1), ("CCCC", 1), ("AAAA", 1)])
    assert_rows(load(ddbc, fastq, min_avg_quality=21), [("ACGT", 2), ("GGGG", 1), ("AAAA", 1)])
    assert_rows(load(ddbc, fastq, min_avg_quality=30), [("ACGT", 2), ("AAAA", 1)])
    assert_rows(load(ddbc, fastq, min_avg_quality=41), [])


def test_reads_with_any_base_of_poor_quality_can_be_left_out(ddbc, fastq, assert_rows):
    assert_rows(load(ddbc, fastq, min_low_quality=10), [("ACGT", 2), ("CCCC", 1), ("AAAA", 1)])  # not r4's 2
    assert_rows(load(ddbc, fastq, min_low_quality=11), [("ACGT", 2), ("CCCC", 1)])  # r6's 10 isn't enough
    assert_rows(load(ddbc, fastq, min_low_quality=21), [("ACGT", 2)])


def test_quality_limits_can_be_used_on_their_own(ddbc, fastq, assert_rows):
    assert_rows(load(ddbc, fastq, min_avg_quality=0, min_low_quality=5), [("ACGT", 2), ("CCCC", 1), ("AAAA", 1)])


def test_both_quality_limits_apply(ddbc, fastq, assert_rows):
    assert_rows(load(ddbc, fastq, min_avg_quality=25, min_low_quality=10), [("ACGT", 2), ("AAAA", 1)])


def test_without_grouping_there_is_a_row_for_each_read(ddbc, fastq, assert_rows):
    output = load(ddbc, fastq, group=False, min_avg_quality=0)
    assert output.columns == ["sequence"]
    assert sorted(r[0] for r in output.fetchall()) == ["AAAA", "ACGT", "ACGT", "CCCC", "GGGG", "TTTT"]


def test_headers_are_there_if_the_reads_are_not_grouped(ddbc, fastq, assert_rows):
    output = load(ddbc, fastq, group=False, header_column=True, min_avg_quality=0)
    assert output.columns == ["sequence", "header"]
    assert_rows(
        output,
        [("ACGT", "r1"), ("ACGT", "r2"), ("TTTT", "r3"), ("GGGG", "r4"), ("CCCC", "r5"), ("AAAA", "r6")],
    )


def test_headers_are_lost_if_the_reads_are_grouped(ddbc, fastq):
    """Documents what happens now: counting by sequence leaves nothing for a header to be attached to, so
    "Header Column?" has no effect unless the sequences are not grouped."""
    output = load(ddbc, fastq, header_column=True)
    assert output.columns == ["sequence", "count"]


def test_several_files_have_their_counts_added_up(ddbc, tmp_path, assert_rows):
    first, second = tmp_path / "one.fastq", tmp_path / "two.fastq"
    first.write_text(
        fastq_text([("a", "ACGT", "IIII"), ("b", "ACGT", "IIII"), ("c", "TTTT", "IIII")]), encoding="utf-8"
    )
    second.write_text(fastq_text([("d", "ACGT", "IIII"), ("e", "GGGG", "IIII")]), encoding="utf-8")
    assert_rows(load(ddbc, first, second), [("ACGT", 3), ("TTTT", 1), ("GGGG", 1)])


def test_or_kept_apart_with_a_column_of_filenames(ddbc, tmp_path, monkeypatch, assert_rows):
    monkeypatch.chdir(tmp_path)
    for name, reads in (
        ("one.fastq", [("a", "ACGT", "IIII"), ("b", "ACGT", "IIII")]),
        ("two.fastq", [("c", "ACGT", "IIII")]),
    ):
        (tmp_path / name).write_text(fastq_text(reads), encoding="utf-8")
    output = load(ddbc, "one.fastq", "two.fastq", filename_column=True)
    assert output.columns == ["sequence", "count", "filename"]
    assert_rows(output, [("ACGT", 2, "one.fastq"), ("ACGT", 1, "two.fastq")])


def test_files_matching_a_wildcard(ddbc, tmp_path, assert_rows):
    for n in range(3):
        (tmp_path / f"sample{n}.fq").write_text(fastq_text([("a", "ACGT", "IIII")]), encoding="utf-8")
    assert_rows(load(ddbc, tmp_path / "sample*.fq"), [("ACGT", 3)])


def test_row_limit_is_for_previews(ddbc, fastq, assert_rows):
    # (it limits the reads read in, not what's left after throwing out the poor ones)
    assert_rows(load(ddbc, fastq, row_limit=2), [("ACGT", 2)])
    assert_rows(load(ddbc, fastq, row_limit=3), [("ACGT", 2)])  # r3 is poor quality
    assert_rows(load(ddbc, fastq, row_limit=100), [("ACGT", 2), ("GGGG", 1), ("CCCC", 1), ("AAAA", 1)])


def test_an_empty_file(ddbc, tmp_path):
    path = tmp_path / "empty.fastq"
    path.write_text("", encoding="utf-8")
    assert load(ddbc, path).fetchall() == []


def test_no_files_gives_nothing(ddbc):
    assert LoadFastqPlugin().execute(ddbc, None) is None


@pytest.mark.parametrize(
    "extension,opener",
    [
        (".fastq", open),
        (".fq", open),
        (".fastq.gz", lambda p, m: gzip.open(p, m)),
        pytest.param(".fastq.bz2", lambda p, m: bz2.open(p, m), marks=pytest.mark.filterwarnings("ignore:Compression")),
        (".fastq.xz", lambda p, m: lzma.open(p, m)),
    ],
)
def test_compressed_files(ddbc, tmp_path, assert_rows, extension, opener, each_decoder):
    path = tmp_path / f"reads{extension}"
    with opener(path, "wt") if extension != ".fastq" and extension != ".fq" else open(path, "w", encoding="utf-8") as f:
        f.write(fastq_text())
    assert_rows(load(ddbc, path), [("ACGT", 2), ("GGGG", 1), ("CCCC", 1), ("AAAA", 1)])


def test_file_types():
    extensions = LoadFastqPlugin.file_types[0][1]
    assert {".fastq", ".fq", ".fastq.gz", ".fq.xz", ".fastq.bz2"} <= set(extensions)
    assert {".fasta", ".fa", ".fasta.gz", ".fa.xz", ".fa.bz2"} <= set(LoadFastaPlugin.file_types[0][1])


# ---- FASTA

FASTA = ">s1 first sequence\nACGT\nACGT\n>s2\nTTTT\n>s3 third\nGGGGCC\n"
FASTA_ROWS = [("s1", "first sequence", "ACGTACGT"), ("s2", None, "TTTT"), ("s3", "third", "GGGGCC")]


def load_fasta(ddbc, *filenames, row_limit=None, **params):
    plugin = LoadFastaPlugin()
    for number, filename in enumerate(filenames):
        plugin.set_parameter(f"files.{number}.filename", str(filename))
    for key, value in params.items():
        plugin.set_parameter(key, value)
    return plugin.execute(ddbc, None, row_limit)


def test_fasta(ddbc, tmp_path, assert_rows):
    path = tmp_path / "seqs.fasta"
    path.write_text(FASTA, encoding="utf-8")
    output = load_fasta(ddbc, path)
    assert output.columns == ["name", "description", "sequence"]
    assert_rows(output, FASTA_ROWS)  # (a sequence over several lines is joined up)


def test_fasta_limit_and_filenames(ddbc, tmp_path, monkeypatch, assert_rows):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "seqs.fa").write_text(FASTA, encoding="utf-8")
    assert len(load_fasta(ddbc, "seqs.fa", row_limit=2).fetchall()) == 2
    output = load_fasta(ddbc, "seqs.fa", filename_column=True)
    assert output.columns == ["name", "description", "sequence", "filename"]
    assert_rows(output, [(*row, "seqs.fa") for row in FASTA_ROWS])


def many_reads_text(n=30000):
    sequences = ["ACGTAC", "TTGACC", "GGGGGG", "ACACAC", "CATCAT", "TGCATG", "AAAAAA"]
    return "".join(f"@r{i}\n{sequences[i * i % 7]}\n+\nIIIIII\n" for i in range(n))


def compress_in_many_blocks(extension, text):
    """.bz2 and .xz files which have a lot of blocks in them, so there's something for the fast decoders to do"""
    if extension == ".fastq.bz2":
        return bz2.compress(text.encode(), 1)
    if compress.lzma_mt is not None:
        return compress.lzma_mt.compress(text.encode(), preset=0, threads=4)
    return lzma.compress(text.encode(), preset=0)  # pragma: no cover


@pytest.mark.parametrize("extension", [".fastq.bz2", ".fastq.xz"])
@pytest.mark.filterwarnings("ignore:Compression")
def test_a_file_with_a_lot_of_blocks_loads_the_same_as_a_plain_one(ddbc, tmp_path, each_decoder, extension):
    text = many_reads_text()
    plain = tmp_path / "reads.fastq"
    plain.write_text(text, encoding="utf-8")
    compressed = tmp_path / f"reads{extension}"
    compressed.write_bytes(compress_in_many_blocks(extension, text))
    expected = sorted(load(ddbc, plain).fetchall())
    assert sum(count for _, count in expected) == 30000
    assert sorted(load(ddbc, compressed).fetchall()) == expected


@pytest.mark.parametrize("extension", [".fastq.bz2", ".fastq.xz"])
@pytest.mark.filterwarnings("ignore:Compression")
def test_a_file_which_is_not_all_there_is_an_error_not_fewer_reads(ddbc, tmp_path, each_decoder, extension):
    data = compress_in_many_blocks(extension, many_reads_text())
    path = tmp_path / f"reads{extension}"
    path.write_bytes(data[: len(data) * 6 // 10])
    with pytest.raises(duckdb.Error):
        load(ddbc, path).fetchall()


@pytest.mark.parametrize("extension", [".fastq.bz2", ".fastq.xz"])
@pytest.mark.filterwarnings("ignore:Compression")
def test_an_empty_file_is_an_error_not_no_reads(ddbc, tmp_path, each_decoder, extension):
    path = tmp_path / f"reads{extension}"
    path.write_bytes(b"")
    with pytest.raises((duckdb.Error, EOFError)):
        load(ddbc, path).fetchall()


@pytest.mark.parametrize(
    "extension,opener",
    [
        (".fasta.gz", lambda p, m: gzip.open(p, m)),
        pytest.param(".fa.bz2", lambda p, m: bz2.open(p, m), marks=pytest.mark.filterwarnings("ignore:Compression")),
        (".fasta.xz", lambda p, m: lzma.open(p, m)),
    ],
)
def test_compressed_fasta(ddbc, tmp_path, assert_rows, extension, opener, each_decoder):
    path = tmp_path / f"seqs{extension}"
    with opener(path, "wt") as f:
        f.write(FASTA)
    assert_rows(load_fasta(ddbc, path), FASTA_ROWS)


def test_combining_nothing_gives_nothing(ddbc):
    assert LoadFastqPlugin().combine(ddbc, []) is None


def quality_filters(ddbc, fastq, **params):
    plugin = LoadFastqPlugin()
    plugin.set_parameter("files.0.filename", str(fastq))
    for key, value in params.items():
        plugin.set_parameter(key, value)
    return plugin.load_file(ddbc.cursor(), str(fastq), plugin.files.params[0]).sql_query()


def test_quality_limits_which_are_set_filter_the_reads(ddbc, fastq):
    assert "list_avg" in quality_filters(ddbc, fastq, min_avg_quality=10)
    assert "list_min" in quality_filters(ddbc, fastq, min_avg_quality=0, min_low_quality=5)


def test_quality_limits_of_zero_mean_no_filtering(ddbc, fastq):
    sql = quality_filters(ddbc, fastq, min_avg_quality=0, min_low_quality=0)
    assert "list_avg" not in sql and "list_min" not in sql
