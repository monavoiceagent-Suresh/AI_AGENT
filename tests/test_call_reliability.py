"""P0 call reliability, end to end through VoiceSession with a media leg that plays audio in real time (like Exotel):
greeting protection, barge-in only on meaningful speech during audible playback, replies after playback not mistaken
for interruptions, playback confirmation kept distinct from transmission, controlled failure paths and deterministic
terminal reasons."""

import asyncio
import logging

import pytest

from app.interfaces import CommittedTranscript, LLMServed, LLMUsage, PartialTranscript, TTSError, Voice
from app.providers.failover import LLMUnavailable
from app.providers.scribe import ASRError
from app.session import SessionConfig, VoiceSession, speech_chars
from tests.fakes import FakeASR, FakeLLM, FakeTTS, RealtimeMedia

pytestmark = pytest.mark.anyio

GREETING = "Hello, this is Mona calling from Acme about the enquiry you made last week."

CONFIG = SessionConfig(
    system_prompt="You are a test agent.",
    greeting=GREETING,
    fallback="Sorry, say that again?",
    language="en",
    voice=Voice(voice_id="voice", model_id="eleven_flash_v2_5"),
    fallback_voice=None,
    llm_model="primary-model",
    barge_in=True,
    idle_timeout_seconds=60,
    max_duration_seconds=60,
    log_transcripts=False,
    barge_in_min_chars=4,
    barge_in_min_speech_ms=300,
    greeting_protect_ms=2500,
)


def make(llm, *, tts=None, media=None, asr=None, **overrides):
    media = media or RealtimeMedia()
    asr = asr or FakeASR()
    config = SessionConfig(**{**CONFIG.__dict__, **overrides})
    session = VoiceSession(config, media=media, asr=asr, llm=llm, tts=tts or FakeTTS(), call_id=9, call_sid="CA9")
    return session, media, asr


async def wait_for(predicate, timeout: float = 5.0) -> None:
    async with asyncio.timeout(timeout):
        while not predicate():
            await asyncio.sleep(0.01)


async def speak_over(asr: FakeASR, text: str, *, seconds: float, steps: int = 4) -> None:
    """Caller speech arriving as a run of growing partials over `seconds`, like Scribe sends while someone talks."""
    words = text.split()
    for i in range(1, steps + 1):
        asr.queue.put_nowait(PartialTranscript(" ".join(words[: max(1, len(words) * i // steps)])))
        await asyncio.sleep(seconds / steps)


def reply_playing(session: VoiceSession, media: RealtimeMedia) -> bool:
    response = session._response
    return (
        response is not None
        and not response.is_greeting
        and response.audio_started_at is not None
        and media.playback_remaining() > 0.3
    )


def test_speech_chars_counts_indic_vowel_signs():
    assert speech_chars("रुको") == 4  # र ु क ो: a short Hindi "stop" is not reduced to its two consonants
    assert speech_chars("hm.") == 2


async def test_greeting_plays_whole_and_confirmed_when_nobody_speaks():
    session, media, asr = make(FakeLLM([]))
    task = asyncio.create_task(session.run())
    await wait_for(lambda: len(session.turns) == 1)
    media.hang_up()
    assert await task == "caller:callended"

    greeting = session.turns[0]
    assert greeting["agent_text"] == GREETING and not greeting["interrupted"]
    assert greeting["playback_confirmed"] is True
    assert greeting["sentences_confirmed"] == greeting["sentences_sent"] >= 1


async def test_caller_saying_hello_right_after_pickup_does_not_cancel_the_greeting():
    llm = FakeLLM(["Yes, I am calling about your enquiry."])
    session, media, asr = make(llm)
    task = asyncio.create_task(session.run())
    await wait_for(lambda: len(media.audio) > 0)  # the greeting has started playing

    await speak_over(asr, "Hello? Hello?", seconds=0.6)  # sustained, meaningful, but inside the protected window
    asr.queue.put_nowait(CommittedTranscript("Hello? Hello?"))
    await wait_for(lambda: len(session.turns) == 2, timeout=8)
    media.hang_up()
    await task

    greeting, answer = session.turns
    assert greeting["agent_text"] == GREETING and not greeting["interrupted"]
    assert media.clears == 0
    assert answer["user_text"] == "Hello? Hello?"
    # The LLM answered knowing the greeting had been said.
    assert llm.requests[0][1] == {"role": "assistant", "content": GREETING}


async def test_a_genuine_barge_in_during_playback_stops_the_agent():
    reply = "Our flats start at forty lakhs. They are ready to move in. We also have villas available now."
    llm = FakeLLM([reply, "Of course, go ahead."])
    session, media, asr = make(llm)
    task = asyncio.create_task(session.run())
    await wait_for(lambda: len(session.turns) == 1, timeout=8)
    asr.say("Tell me about prices")
    # Interrupt once the first sentence has been confirmed played and the rest is still playing.
    await wait_for(lambda: reply_playing(session, media) and session._response.played >= 1)

    await speak_over(asr, "wait wait I have a question", seconds=0.5)
    await wait_for(lambda: len(session.turns) == 2)
    asr.queue.put_nowait(CommittedTranscript("wait wait I have a question"))
    await wait_for(lambda: len(session.turns) == 3, timeout=8)
    media.hang_up()
    await task

    interrupted = session.turns[1]
    assert interrupted["interrupted"] and interrupted["interrupted_by"] == "barge_in"
    assert media.clears == 1
    assert interrupted["agent_text"].startswith("Our flats start at forty lakhs.")
    assert "villas" not in interrupted["agent_text"]  # only what was heard is kept
    assert session.history[-3]["content"].endswith("…")
    assert session.turns[2]["user_text"] == "wait wait I have a question"


async def test_short_fragments_and_noise_during_playback_do_not_interrupt():
    reply = "Our flats start at forty lakhs. They are ready to move in."
    llm = FakeLLM([reply, "Yes."])
    session, media, asr = make(llm)
    task = asyncio.create_task(session.run())
    await wait_for(lambda: len(session.turns) == 1, timeout=8)
    asr.say("Tell me about prices")
    await wait_for(lambda: reply_playing(session, media))

    asr.queue.put_nowait(PartialTranscript("hm"))  # a backchannel / noise fragment
    await asyncio.sleep(0.5)
    asr.queue.put_nowait(PartialTranscript("hm"))
    asr.queue.put_nowait(CommittedTranscript("hm"))
    await wait_for(lambda: len(session.turns) == 3, timeout=8)
    media.hang_up()
    await task

    assert media.clears == 0
    assert session.turns[1]["agent_text"] == reply and not session.turns[1]["interrupted"]
    assert session.turns[2]["user_text"] == "hm"  # answered after the reply, not by cutting it off


async def test_caller_speaking_after_playback_ended_is_a_reply_not_an_interruption():
    # Exotel's marks are late/missing; by the playback clock the agent has finished speaking.
    llm = FakeLLM(["Sure, we are open till nine.", "Great."])
    session, media, asr = make(llm)
    task = asyncio.create_task(session.run())
    await wait_for(lambda: len(session.turns) == 1, timeout=8)
    media.ack_marks = False
    asr.say("Are you open late")
    await wait_for(lambda: session._response is not None and session._response.draining)
    await wait_for(lambda: media.playback_remaining() == 0)

    await speak_over(asr, "Okay thanks a lot", seconds=0.5)
    asr.queue.put_nowait(CommittedTranscript("Okay thanks a lot"))
    await wait_for(lambda: len(session.turns) == 3, timeout=8)
    media.hang_up()
    await task

    answered = session.turns[1]
    assert not answered["interrupted"] and answered["interrupted_by"] is None
    assert answered["agent_text"] == "Sure, we are open till nine."  # nothing truncated from history
    assert answered["playback_confirmed"] is False and answered["sentences_confirmed"] == 0  # never faked
    assert media.clears == 0


async def test_a_missing_playback_mark_is_recorded_as_unconfirmed_not_played(caplog):
    caplog.set_level(logging.WARNING)
    session, media, asr = make(FakeLLM([]), media=RealtimeMedia(ack_marks=False))
    task = asyncio.create_task(session.run())
    await wait_for(lambda: len(session.turns) == 1, timeout=8)
    media.hang_up()
    await task

    greeting = session.turns[0]
    assert greeting["sentences_sent"] >= 1 and greeting["sentences_confirmed"] == 0
    assert greeting["playback_confirmed"] is False
    assert any("playback unconfirmed" in r.getMessage() for r in caplog.records)


async def test_interrupting_stops_synthesis_of_the_rest_of_the_reply():
    # A long first sentence (about 3 s of synthesis) so the interruption always lands while it is still being
    # synthesized, even on a loaded machine.
    reply = (
        "First sentence is quite long and takes a while to say because it keeps on going with many more words "
        "than usual before it finally reaches its end. Second sentence here. Third sentence too."
    )
    tts = FakeTTS(chunk_delay=0.1)  # synthesis at roughly real-time speed
    session, media, asr = make(FakeLLM([reply, "Okay."]), tts=tts)
    task = asyncio.create_task(session.run())
    await wait_for(lambda: len(session.turns) == 1, timeout=8)
    asr.say("Explain")
    await wait_for(lambda: session._response is not None and not session._response.is_greeting
                   and session._audible())

    await speak_over(asr, "stop please stop", seconds=0.4)
    await wait_for(lambda: media.clears == 1)  # the barge-in happened
    await asyncio.sleep(0.5)  # give any (wrongly) continuing synthesis time to show up
    media.hang_up()
    await task

    assert not any("Third sentence" in text for text in tts.texts)  # cancelled, not synthesized and wasted
    assert not any("Second sentence" in text for text in tts.texts)
    assert len(session.turns) == 1  # interrupted mid first sentence: nothing complete was heard, so no fake turn


class UnavailableThenOkLLM(FakeLLM):
    async def stream(self, messages, model):
        self.requests.append(messages)
        if len(self.requests) == 1:
            raise LLMUnavailable("no LLM", status=429, reason="primary=rate_limited fallback=timeout")
        yield LLMServed(provider="OpenRouter", model="fallback-model", role="fallback")
        yield "Back again. "
        yield LLMUsage(1, 1, 0.0)


async def test_both_llms_failing_speaks_the_fallback_phrase_and_records_why():
    session, media, asr = make(UnavailableThenOkLLM([]))
    task = asyncio.create_task(session.run())
    await wait_for(lambda: len(session.turns) == 1, timeout=8)
    asr.say("Hello there")
    await wait_for(lambda: len(session.turns) == 2, timeout=8)
    asr.say("Are you there")
    await wait_for(lambda: len(session.turns) == 3, timeout=8)
    media.hang_up()
    assert await task == "caller:callended"  # the call continued after the controlled failure

    failed, recovered = session.turns[1], session.turns[2]
    assert failed["agent_text"] == "Sorry, say that again?"
    assert failed["failure_reason"] == "llm_unavailable(primary=rate_limited fallback=timeout)"
    assert session.failures == [{"response_id": 2, "reason": "llm_unavailable(primary=rate_limited fallback=timeout)"}]
    assert recovered["agent_text"] == "Back again." and recovered["llm_model"] == "fallback-model"


class BrokenTTS(FakeTTS):
    async def stream(self, text, **kwargs):
        self.texts.append(text)
        raise TTSError("ElevenLabs TTS failed (HTTP 500, model eleven_flash_v2_5)")
        yield b""  # pragma: no cover


async def test_a_tts_api_failure_is_recorded_and_the_call_ends_deterministically():
    session, media, asr = make(FakeLLM([]), tts=BrokenTTS(), max_consecutive_llm_failures=1)
    reason = await asyncio.wait_for(session.run(), 5)

    assert reason == "response_failures"
    assert session.turns == []  # nothing was ever heard: no fake "successful" turn
    assert session.failures == [{"response_id": 1, "reason": "tts_failed"}]


class FailingSendMedia(RealtimeMedia):
    async def send_audio(self, pcm: bytes) -> None:
        raise RuntimeError("socket write failed")


async def test_an_audio_transmission_failure_is_distinct_from_a_synthesis_failure():
    tts = FakeTTS()
    session, media, asr = make(FakeLLM([]), tts=tts, media=FailingSendMedia(), max_consecutive_llm_failures=1)
    reason = await asyncio.wait_for(session.run(), 5)

    assert reason == "response_failures"
    assert tts.texts  # synthesis succeeded...
    assert session.failures[0]["reason"] == "audio_send_failed"  # ...transmission did not
    assert session.turns == []


class ScriptedServedLLM(FakeLLM):
    async def stream(self, messages, model):
        self.requests.append(messages)
        yield LLMServed(provider="OpenRouter", model="google/gemini-2.5-flash-lite", role="primary")
        for word in self.replies.pop(0).split(" "):
            yield word + " "
        yield LLMUsage(1, 1, 0.0)


async def test_the_model_that_served_each_reply_is_recorded():
    session, media, asr = make(ScriptedServedLLM(["We open at nine."]))
    task = asyncio.create_task(session.run())
    await wait_for(lambda: len(session.turns) == 1, timeout=8)
    asr.say("When do you open")
    await wait_for(lambda: len(session.turns) == 2, timeout=8)
    media.hang_up()
    await task
    assert session.turns[1]["llm_model"] == "google/gemini-2.5-flash-lite"


class DyingASR(FakeASR):
    async def events(self):
        yield PartialTranscript("")
        raise ASRError("scribe disconnected (ConnectionClosedError) and could not be reopened")


async def test_an_asr_failure_ends_the_call_with_a_recorded_reason():
    session, media, asr = make(FakeLLM([]), asr=DyingASR())
    reason = await asyncio.wait_for(session.run(), 5)
    assert reason == "asr_failed:ASRError"


class RefusingASR(FakeASR):
    async def send_audio(self, pcm: bytes) -> None:
        raise ConnectionError("asr socket gone")


async def test_an_asr_that_stops_accepting_audio_ends_the_call_with_a_recorded_reason():
    from app.interfaces import CallAudio

    session, media, asr = make(FakeLLM([]), asr=RefusingASR())
    task = asyncio.create_task(session.run())
    media.inbox.put_nowait(CallAudio(bytes(320)))
    assert await asyncio.wait_for(task, 5) == "asr_failed:ConnectionError"


async def test_a_websocket_disconnect_mid_reply_ends_with_its_reason_and_keeps_what_was_heard():
    from app.interfaces import CallEnded

    reply = "This reply is fairly long. It keeps going for a while. And then some more."
    session, media, asr = make(FakeLLM([reply]))
    task = asyncio.create_task(session.run())
    await wait_for(lambda: len(session.turns) == 1, timeout=8)
    asr.say("Tell me more")
    await wait_for(lambda: reply_playing(session, media))
    media.inbox.put_nowait(CallEnded("disconnected"))

    assert await asyncio.wait_for(task, 5) == "caller:disconnected"
    if len(session.turns) == 2:  # whatever had already played is kept, marked as cut off by the call ending
        assert session.turns[1]["interrupted_by"] == "call_ended"


async def test_the_greeting_can_still_be_interrupted_once_its_protection_has_passed():
    long_greeting = " ".join(["Hello there, this is a long introduction."] * 5)
    session, media, asr = make(FakeLLM(["Sure."]), greeting=long_greeting, greeting_protect_ms=400)
    task = asyncio.create_task(session.run())
    await wait_for(lambda: session._response is not None and session._response.audio_started_at is not None)
    await asyncio.sleep(0.6)

    await speak_over(asr, "sorry who is this calling", seconds=0.5)
    await wait_for(lambda: len(session.turns) >= 1)
    media.hang_up()
    await task

    assert session.turns[0]["interrupted"] and session.turns[0]["interrupted_by"] == "barge_in"
    assert session.turns[0]["agent_text"] != long_greeting
