import os
from pathlib import Path

import pytest

from countess.utils.files import atomic_output


def test_what_is_written_is_there_when_it_is_finished(tmp_path):
    target = tmp_path / "out.txt"
    with atomic_output(target) as temp:
        Path(temp).write_text("all of it")
    assert target.read_text() == "all of it"
    assert os.listdir(tmp_path) == ["out.txt"]


def test_it_works_with_a_string_too(tmp_path):
    with atomic_output(str(tmp_path / "out.txt")) as temp:
        assert isinstance(temp, str)
        Path(temp).write_text("x")
    assert (tmp_path / "out.txt").read_text() == "x"


def test_the_file_replaces_one_which_was_there(tmp_path):
    target = tmp_path / "out.txt"
    target.write_text("old")
    with atomic_output(target) as temp:
        Path(temp).write_text("new")
    assert target.read_text() == "new"
    assert os.listdir(tmp_path) == ["out.txt"]


def test_the_old_file_is_still_there_until_the_new_one_is_finished(tmp_path):
    target = tmp_path / "out.txt"
    target.write_text("old")
    with atomic_output(target) as temp:
        Path(temp).write_text("half of the new one")
        assert target.read_text() == "old"  # (so a reader in the meantime gets all of one or the other)
    assert target.read_text() == "half of the new one"


def test_a_failure_leaves_the_old_file_and_no_other(tmp_path):
    target = tmp_path / "out.txt"
    target.write_text("old")
    with pytest.raises(ValueError, match="went wrong"):
        with atomic_output(target) as temp:
            Path(temp).write_text("part of the new one")
            raise ValueError("went wrong")
    assert target.read_text() == "old"
    assert os.listdir(tmp_path) == ["out.txt"]


def test_a_failure_with_no_old_file_leaves_nothing(tmp_path):
    with pytest.raises(ValueError):
        with atomic_output(tmp_path / "out.txt") as temp:
            Path(temp).write_text("part of it")
            raise ValueError("went wrong")
    assert os.listdir(tmp_path) == []


def test_a_failure_before_anything_is_written_is_fine(tmp_path):
    with pytest.raises(ValueError):
        with atomic_output(tmp_path / "out.txt"):
            raise ValueError("went wrong before it started")
    assert os.listdir(tmp_path) == []


def test_an_interrupt_is_a_failure_too(tmp_path):
    with pytest.raises(KeyboardInterrupt):
        with atomic_output(tmp_path / "out.txt") as temp:
            Path(temp).write_text("part of it")
            raise KeyboardInterrupt()
    assert os.listdir(tmp_path) == []


def test_the_temporary_file_is_beside_the_file_hidden_and_has_the_same_ending(tmp_path):
    """(Beside it so it can be moved to it, and the same ending as what a writer goes by to know how to write it.)"""
    with atomic_output(tmp_path / "out.csv.gz") as temp:
        assert Path(temp).parent == tmp_path
        assert Path(temp).name.startswith(".") and Path(temp).name.endswith(".out.csv.gz")
        assert Path(temp) != tmp_path / "out.csv.gz"
        Path(temp).write_text("x")


def test_each_use_has_a_temporary_file_of_its_own(tmp_path):
    """(So that two writing the same file at the same time don't write to the same one.)"""
    with atomic_output(tmp_path / "out.txt") as first, atomic_output(tmp_path / "out.txt") as second:
        assert first != second
        Path(first).write_text("1")
        Path(second).write_text("2")


def test_it_can_not_replace_a_directory_and_says_so_and_tidies_up(tmp_path):
    (tmp_path / "out.txt").mkdir()
    with pytest.raises(OSError):
        with atomic_output(tmp_path / "out.txt") as temp:
            Path(temp).write_text("x")
    assert os.listdir(tmp_path) == ["out.txt"]


def test_a_writer_which_made_no_file_is_an_error_and_not_a_success(tmp_path):
    (tmp_path / "out.txt").write_text("old")
    with pytest.raises(FileNotFoundError):
        with atomic_output(tmp_path / "out.txt"):
            pass  # (it wrote nothing)
    assert (tmp_path / "out.txt").read_text() == "old"
    assert os.listdir(tmp_path) == ["out.txt"]
