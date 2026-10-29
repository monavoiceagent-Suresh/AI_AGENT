"""Redaction for diagnostic logs of telephony payloads.

Credentials are never logged. A phone number is shown only by its last four digits, except for the keys a
caller names as the business side of the call (for example the number that was dialled), which are shown in full.
"""

import re
from typing import Any

_SENSITIVE_KEY = re.compile(r"authori[sz]ation|token|secret|passw(or)?d|api[_-]?key|credential|cookie|bearer", re.I)
_PHONE_LIKE = re.compile(r"^\+?[\d\s().-]{7,}$")
_URL_CREDENTIALS = re.compile(r"(://)[^/\s@]+@")
_PARTIAL_KEYS = frozenset({"account_sid", "accountsid"})
MAX_TEXT = 60
MAX_DEPTH = 3
MAX_ITEMS = 30


def mask_number(text: str) -> str:
    digits = re.sub(r"\D", "", text)
    return "*" * max(len(digits) - 4, 0) + digits[-4:]


def redact(value: Any, *, key: str = "", keep_full: frozenset[str] = frozenset(), depth: int = 0) -> Any:
    """A copy of `value` that is safe to log: structure kept, secrets removed, personal numbers masked."""
    if _SENSITIVE_KEY.search(key):
        return "[redacted]"
    if isinstance(value, dict):
        if depth >= MAX_DEPTH:
            return "{...}"
        items = list(value.items())[:MAX_ITEMS]
        return {str(k): redact(v, key=str(k), keep_full=keep_full, depth=depth + 1) for k, v in items}
    if isinstance(value, list | tuple):
        return [redact(v, key=key, keep_full=keep_full, depth=depth + 1) for v in list(value)[:MAX_ITEMS]]
    if isinstance(value, str):
        text = value.strip()
        if key.lower() in _PARTIAL_KEYS:
            return "..." + text[-4:]
        if key in keep_full:
            return text[:MAX_TEXT]
        if _PHONE_LIKE.match(text):
            return mask_number(text)
        text = _URL_CREDENTIALS.sub(r"\1[redacted]@", text)
        return text[:MAX_TEXT] + ("..." if len(text) > MAX_TEXT else "")
    return value
