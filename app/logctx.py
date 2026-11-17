"""Per-call log correlation that reaches the provider adapters without threading arguments through them.

VoiceSession sets the call fields once (copied into every task it creates) and each response task sets its own
response id, so an LLM or TTS log line can name the call and response it belongs to. Values are identifiers
only: never numbers, transcripts or credentials.
"""

from contextvars import ContextVar

_call: ContextVar[str] = ContextVar("call_log_context", default="")
_response: ContextVar[str] = ContextVar("response_log_context", default="")


def set_call(*, call_id: int | None, call_sid: str, stream_sid: str) -> None:
    _call.set(f"call_id={call_id or '-'} call_sid={call_sid or '-'} stream_sid={stream_sid or '-'}")


def set_response(response_id: int | None) -> None:
    _response.set(f"response_id={response_id}" if response_id is not None else "")


def fields() -> str:
    """`call_id=… call_sid=… stream_sid=… response_id=…` for the current task, or "" outside a call."""
    return " ".join(part for part in (_call.get(), _response.get()) if part)
