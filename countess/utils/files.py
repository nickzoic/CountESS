"""Helpers for files which are written."""

import contextlib
import logging
import os
import secrets
from pathlib import Path
from typing import Iterator, Union

logger = logging.getLogger(__name__)


@contextlib.contextmanager
def atomic_output(filename: Union[str, "os.PathLike[str]"]) -> Iterator[str]:
    """For writing a file so that it's all there or it isn't there at all.

    Gives the name of a temporary file, in the same directory so that it can be moved into place, and with the
    same ending as `filename`, as that's what tells some writers how to write it.  When the `with` finishes the
    temporary file replaces `filename`, which is then the whole of what was written; and if it doesn't finish, as
    the writing went wrong part way through, the temporary file is removed and so whatever was at `filename`
    before is left as it was, and nothing which isn't all of a file is left in its place.

    DuckDB's own way of doing this (it writes to a file called tmp_<name> if the file's already there) leaves
    that file behind when the write fails with more than one thread, and when the file wasn't there leaves
    a part of one at `filename`.  And the pandas writers open `filename` itself, so a failure part way through
    leaves the old file replaced with a part of the new one."""
    path = Path(filename)
    temp_path = path.with_name(f".{secrets.token_hex(4)}.{path.name}")
    try:
        yield str(temp_path)
        os.replace(temp_path, path)
    except BaseException:
        # (BaseException as it's just as much a failure if it's an interrupt: whatever it is, it's re-raised)
        try:
            temp_path.unlink(missing_ok=True)
        except OSError as exc:  # pragma: no cover  (as when something still has it open, which Windows minds)
            logger.warning("Couldn't remove the temporary file %s: %s", temp_path, exc)
        raise
