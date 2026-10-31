"""connect_asr: the connect-time-only ASR fallback. Falls back only if the primary's connection attempt
itself fails; once a session is open, a later failure must never trigger a silent switch to the fallback,
since that could duplicate or drop turns mid-call."""

from contextlib import asynccontextmanager

import pytest

from app.calls import connect_asr
from tests.fakes import make_settings

pytestmark = pytest.mark.anyio


class FakeTranscriber:
    def __init__(self, name: str):
        self.name = name
        self.closed = False

    async def send_audio(self, pcm: bytes) -> None: ...

    async def events(self):
        return
        yield  # pragma: no cover - makes this an async generator function


class FakeASR:
    """An ASR provider: .connect() either yields a transcriber or raises, as scripted."""

    def __init__(self, name: str, *, fails_to_connect: bool = False, fails_mid_session: bool = False):
        self.name = name
        self.fails_to_connect = fails_to_connect
        self.fails_mid_session = fails_mid_session
        self.connect_attempts = 0
        self.closed_cleanly = False

    @asynccontextmanager
    async def connect(self, *, language: str, sample_rate: int):
        self.connect_attempts += 1
        if self.fails_to_connect:
            raise ConnectionError(f"{self.name} refused the connection")
        transcriber = FakeTranscriber(self.name)
        try:
            yield transcriber
            if self.fails_mid_session:
                raise RuntimeError(f"{self.name} dropped mid-session")
        finally:
            self.closed_cleanly = True
            transcriber.closed = True


async def test_primary_connects_successfully_the_fallback_is_never_touched():
    primary, fallback = FakeASR("primary"), FakeASR("fallback")

    async with connect_asr(primary, fallback, language="en", sample_rate=8000) as transcriber:
        assert transcriber.name == "primary"

    assert primary.connect_attempts == 1 and primary.closed_cleanly
    assert fallback.connect_attempts == 0


async def test_primary_connect_failure_falls_back():
    primary = FakeASR("primary", fails_to_connect=True)
    fallback = FakeASR("fallback")

    async with connect_asr(primary, fallback, language="en", sample_rate=8000) as transcriber:
        assert transcriber.name == "fallback"

    assert primary.connect_attempts == 1
    assert fallback.connect_attempts == 1 and fallback.closed_cleanly


async def test_primary_connect_failure_with_no_fallback_configured_raises():
    primary = FakeASR("primary", fails_to_connect=True)

    with pytest.raises(ConnectionError, match="primary refused"):
        async with connect_asr(primary, None, language="en", sample_rate=8000):
            pass


async def test_a_mid_session_failure_is_never_retried_on_the_fallback():
    """The critical safety property: once connected, a later failure must propagate, not silently switch
    providers (which could lose or duplicate turns already in progress)."""
    primary = FakeASR("primary", fails_mid_session=True)
    fallback = FakeASR("fallback")

    with pytest.raises(RuntimeError, match="primary dropped mid-session"):
        async with connect_asr(primary, fallback, language="en", sample_rate=8000):
            pass  # the body runs fine; the failure happens on the primary's own context-manager exit

    assert primary.connect_attempts == 1
    assert fallback.connect_attempts == 0  # never touched: the connection had already succeeded


async def test_a_failure_raised_inside_the_call_body_also_never_falls_back():
    primary = FakeASR("primary")
    fallback = FakeASR("fallback")

    with pytest.raises(ValueError, match="something else failed"):
        async with connect_asr(primary, fallback, language="en", sample_rate=8000):
            raise ValueError("something else failed")

    assert primary.connect_attempts == 1 and primary.closed_cleanly
    assert fallback.connect_attempts == 0


async def test_language_and_sample_rate_are_forwarded_to_whichever_provider_connects():
    seen = {}

    class RecordingASR(FakeASR):
        @asynccontextmanager
        async def connect(self, *, language: str, sample_rate: int):
            seen["language"], seen["sample_rate"] = language, sample_rate
            async with super().connect(language=language, sample_rate=sample_rate) as t:
                yield t

    primary = RecordingASR("primary", fails_to_connect=True)
    fallback = RecordingASR("fallback")

    async with connect_asr(primary, fallback, language="ta", sample_rate=8000):
        pass

    assert seen == {"language": "ta", "sample_rate": 8000}


def test_make_settings_fixture_still_has_an_asr_fallback_default_of_none():
    # Sanity check that this test file's expectations match the shared test fixture's defaults.
    assert make_settings().asr_fallback_provider is None
