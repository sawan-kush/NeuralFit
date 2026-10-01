from __future__ import annotations

import re
import uuid

_ID_RE = re.compile(r"^[0-9a-f]{32}$")
_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,100}$")


def new_id() -> str:
    return uuid.uuid4().hex


def is_valid_id(value: str) -> bool:
    return bool(_ID_RE.match(value))


def is_safe_name(value: str) -> bool:
    return bool(_NAME_RE.match(value)) and ".." not in value
