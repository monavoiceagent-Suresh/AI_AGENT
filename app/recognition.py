"""Speech-recognition hints: a persona's business vocabulary, turned into ASR keyterms.

A persona's `recognition_hints` is free text from the dashboard ("RFID, UHF, EPC Gen2, GreenFuturz"). It is
normalised here (identically to Mona's preview, monaaivoiceagent-main/src/lib/recognition-hints.ts; both are
checked against tests/fixtures/recognition_cases.json) and then cut to each provider's own limits:

- ElevenLabs Scribe Realtime (`keyterms`, repeated query parameter). The official SDKs document "Maximum 50
  keyterms, each up to 20 characters" for realtime (@elevenlabs/client 1.30.0 scribe.d.ts; elevenlabs-python
  realtime/scribe.py). The 5-word cap and the forbidden `< > { } [ ] \\` come from the batch API reference, the
  only place ElevenLabs states them; they are applied to realtime too.
- Deepgram Nova-3 (`keyterm`, repeated): "limited to 500 tokens per request"; Deepgram's guidance is to "focus on
  the most important 20-50 terms". Deepgram states no per-term limit, so the per-term caps below are this
  engine's own conservative ones, and the total is held to 500 characters to stay clear of the token limit.

Terms that are invalid (unsupported characters) or over a per-term limit are dropped, never truncated or
cleaned up: a cut-off word would bias recognition towards something nobody says. Once a provider's term count or
total size is reached, later terms are left out (the dashboard tells the user to put important terms first).
These terms go to the ASR provider only; they are never added to the LLM prompt.
"""

import re
import unicodedata
from dataclasses import dataclass


@dataclass(frozen=True)
class Limits:
    max_terms: int
    max_term_chars: int  # Unicode code points, after normalisation
    max_term_words: int
    max_total_chars: int | None  # None: no total budget beyond max_terms * max_term_chars


SCRIBE_LIMITS = Limits(max_terms=50, max_term_chars=20, max_term_words=5, max_total_chars=None)
DEEPGRAM_LIMITS = Limits(max_terms=50, max_term_chars=50, max_term_words=5, max_total_chars=500)
PROVIDER_LIMITS: dict[str, Limits] = {"elevenlabs_scribe": SCRIBE_LIMITS, "deepgram_nova3": DEEPGRAM_LIMITS}
# A provider this module does not know gets the strictest of both.
STRICTEST_LIMITS = Limits(max_terms=50, max_term_chars=20, max_term_words=5, max_total_chars=500)

# What the dashboard may send at most (PersonaIn/AgentIn); parsing then keeps only what fits the limits above.
MAX_HINTS_CHARS = 4000

_SEPARATORS = re.compile(r"[,;\n\r]+")
# A term containing any of these is invalid and dropped whole: the characters Scribe rejects, C0/C1 control
# characters other than the whitespace ones (tab, VT, FF), and the byte-order mark.
_FORBIDDEN = re.compile(r"[<>{}\[\]\\\x00-\x08\x0e-\x1f\x7f-\x9f\ufeff]")
# Whitespace, spelled out so that Python and JavaScript (/\s/) agree exactly on what separates words.
_WHITESPACE = re.compile(r"[\t\x0b\x0c \u00a0\u1680\u2000-\u200a\u2028\u2029\u202f\u205f\u3000]+")


def limits_for(provider: str | None) -> Limits:
    return PROVIDER_LIMITS.get(provider or "", STRICTEST_LIMITS)


def normalise(raw: str) -> str | None:
    """One entry as a term: NFC, single spaces, trimmed. "" for an empty entry, None for an invalid one."""
    text = unicodedata.normalize("NFC", raw)
    if _FORBIDDEN.search(text):
        return None
    return _WHITESPACE.sub(" ", text).strip(" ")


def keyterms(hints: str | None, provider: str | None = "elevenlabs_scribe") -> tuple[str, ...]:
    """The persona's hints as keyterms `provider` accepts, in the order given. Empty hints give no terms."""
    if not hints:
        return ()
    limits = limits_for(provider)
    terms: list[str] = []
    seen: set[str] = set()
    total = 0
    for raw in _SEPARATORS.split(hints):
        term = normalise(raw)
        if not term:
            continue
        if len(term) > limits.max_term_chars or len(term.split(" ")) > limits.max_term_words:
            continue
        key = term.lower()
        if key in seen:
            continue
        if len(terms) >= limits.max_terms or (
            limits.max_total_chars is not None and total + len(term) > limits.max_total_chars
        ):
            break
        seen.add(key)
        terms.append(term)
        total += len(term)
    return tuple(terms)
