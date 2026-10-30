"""kai_slice.media — local filesystem path translation.

Re-homed verbatim from the source project (only imports and docstrings changed).
A capture manifest may record a path in its originating OS's drive form; the
local process can only open the mount this OS actually exposes, so every probe
or copy translates first.
"""

from __future__ import annotations

import re

_WIN_DRIVE = re.compile(r"^([A-Za-z]):[/\\](.*)$")


def local_path(src: str) -> str:
    """``src`` as the LOCAL OS can open it.

    A Windows drive path (``<drive>:/...`` or ``<drive>:\\...``) becomes its
    mount form (``/mnt/<drive>/...``); anything else is returned unchanged. A
    manifest may keep the drive form on purpose, so every local probe has to
    translate first or it hands the OS a path it cannot open.
    """
    m = _WIN_DRIVE.match(src)
    if not m:
        return src
    return f"/mnt/{m.group(1).lower()}/{m.group(2).replace(chr(92), '/')}"


__all__ = ["local_path"]
