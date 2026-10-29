import asyncio

import pytest

from app.interfaces import CallAudio, TTSError, Voice
from app.session import SentenceSplitter, SessionConfig, VoiceSession
from tests.fakes import FakeASR, FakeLLM, FakeMedia, FakeTTS

pytestmark = pytest.mark.anyio

CONFIG = SessionConfig(
    system_prompt="You are a test agent.",
    greeting="Hello, how can I help?",
    fallback="Sorry, say that again?",
    language="en",
    voice=Voice(voice_id="voice", model_id="m1"),
    fallback_voice=Voice(voice_id="backup", model_id="eleven_flash_v2_5"),
    llm_model="model",
    barge_in=True,
    idle_timeout_seconds=60,
    max_duration_seconds=60,
    log_transcripts=False,
)


def make_session(llm: FakeLLM, tts=None, fallback_tts=None, **overrides) -> tuple[VoiceSession, FakeMedia, FakeASR]:
    media, asr = FakeMedia(), FakeASR()
    config = SessionConfig(**{**CONFIG.__dict__, **overrides})
    session = VoiceSession(config, media=media, asr=asr, llm=llm, tts=tts or FakeTTS(), fallback_tts=fallback_tts)
    return session, media, asr


async def wait_for(predicate, timeout: float = 2.0) -> None:
    async with asyncio.timeout(timeout):
        while not predicate():
            await asyncio.sleep(0.01)


def test_splitter_emits_sentences_and_long_clauses():
    splitter = SentenceSplitter()
    assert splitter.push("Hello there. How") == ["Hello there."]
    assert splitter.push(" are you? I am") == ["How are you?"]
    assert splitter.push(" fine") == []
    assert splitter.finish() == ["I am fine"]
    # Hindi danda ends a sentence; decimals don't.
    assert splitter.push("कीमत 3.5 लाख है। ठीक") == ["कीमत 3.5 लाख है।"]
    splitter.finish()
    long_clause = "We have flats in Chennai, Coimbatore and Madurai, all ready to move in"
    assert splitter.push(long_clause) == ["We have flats in Chennai, Coimbatore and Madurai,"]


async def test_greeting_then_answer_records_turns_and_history():
    session, media, asr = make_session(FakeLLM(["Sure. We open at nine."]))
    task = asyncio.create_task(session.run())
    await wait_for(lambda: len(session.turns) == 1)
    media.inbox.put_nowait(CallAudio(bytes(1600)))
    asr.say("When do you open?")
    await wait_for(lambda: len(session.turns) == 2)
    media.hang_up()

    assert await task == "caller:callended"
    greeting, answer = session.turns
    assert greeting["user_text"] is None and greeting["agent_text"] == "Hello, how can I help?"
    assert answer["user_text"] == "When do you open?"
    assert answer["agent_text"] == "Sure. We open at nine."
    assert not answer["interrupted"]
    assert answer["llm_ttft_ms"] is not None and answer["response_latency_ms"] is not None
    assert [m["role"] for m in session.history] == ["system", "assistant", "user", "assistant"]
    assert session.asr_bytes == 1600
    assert session.llm_usage.completion_tokens == 20
    assert session.tts_characters == len("Hello, how can I help?") + len("Sure.") + len("We open at nine.")


async def test_barge_in_clears_playback_and_keeps_only_heard_text():
    llm = FakeLLM(["First sentence here. Second sentence that is long. Third one.", "Okay."])
    tts = FakeTTS(chunk_delay=0.05)
    session, media, asr = make_session(llm, tts)
    task = asyncio.create_task(session.run())
    await wait_for(lambda: len(session.turns) == 1)
    asr.say("Tell me everything")
    await wait_for(lambda: len(media.marks) >= 2)  # greeting + first sentence played
    asr.say("Stop, I have a question")
    await wait_for(lambda: len(session.turns) == 3)
    media.hang_up()
    await task

    interrupted = session.turns[1]
    assert interrupted["interrupted"]
    assert interrupted["agent_text"] == "First sentence here."
    assert media.clears >= 1
    assert session.turns[2]["user_text"] == "Stop, I have a question"
    assert session.history[-3]["content"] == "First sentence here.…"


async def test_interruption_before_any_audio_merges_caller_text():
    llm = FakeLLM(["This reply is never heard.", "Got it."], token_delay=0.2)
    session, media, asr = make_session(llm)
    task = asyncio.create_task(session.run())
    await wait_for(lambda: len(session.turns) == 1)
    asr.say("I want to book")
    await asyncio.sleep(0.05)
    asr.say("an appointment for Monday")
    await wait_for(lambda: len(session.turns) == 2, timeout=5)
    media.hang_up()
    await task

    assert session.turns[1]["user_text"] == "I want to book an appointment for Monday"
    assert session.turns[1]["agent_text"] == "Got it."
    assert llm.requests[1][-1]["content"] == "I want to book an appointment for Monday"


async def test_end_call_token_is_not_spoken_and_ends_the_call():
    tts = FakeTTS()
    session, media, asr = make_session(FakeLLM(["Thank you, goodbye! <end_call>"]), tts)
    task = asyncio.create_task(session.run())
    await wait_for(lambda: len(session.turns) == 1)
    asr.say("That's all, bye")

    assert await asyncio.wait_for(task, 2) == "agent_ended"
    assert tts.texts[-1] == "Thank you, goodbye!"
    assert session.turns[-1]["agent_text"] == "Thank you, goodbye!"


async def test_without_barge_in_caller_speech_is_answered_after_the_response():
    llm = FakeLLM(["One. Two. Three.", "Answer."])
    session, media, asr = make_session(llm, FakeTTS(chunk_delay=0.05), barge_in=False)
    task = asyncio.create_task(session.run())
    await wait_for(lambda: len(session.turns) == 1)
    asr.say("Count please")
    await wait_for(lambda: len(media.marks) >= 2)
    asr.say("And then?")
    await wait_for(lambda: len(session.turns) == 3, timeout=5)
    media.hang_up()
    await task

    assert media.clears == 0
    assert session.turns[1]["agent_text"] == "One. Two. Three." and not session.turns[1]["interrupted"]
    assert session.turns[2]["user_text"] == "And then?"


async def test_missing_marks_fall_back_to_playback_clock():
    session, media, asr = make_session(FakeLLM([]))
    media.ack_marks = False
    task = asyncio.create_task(session.run())
    await wait_for(lambda: len(session.turns) == 1, timeout=4)
    media.hang_up()
    await task
    assert session.turns[0]["agent_text"] == "Hello, how can I help?"


async def test_idle_caller_ends_the_call():
    session, media, asr = make_session(FakeLLM([]), idle_timeout_seconds=1)
    assert await asyncio.wait_for(session.run(), 5) == "idle_timeout"


class FailingLLM(FakeLLM):
    async def stream(self, messages, model):
        raise RuntimeError("upstream 429")
        yield  # pragma: no cover - makes this an async generator


async def test_llm_failure_apologises_to_the_caller():
    tts = FakeTTS()
    session, media, asr = make_session(FailingLLM([]), tts)
    task = asyncio.create_task(session.run())
    await wait_for(lambda: len(session.turns) == 1)
    asr.say("What are your timings?")
    await wait_for(lambda: "Sorry, say that again?" in tts.texts)
    await asyncio.sleep(0.1)
    media.hang_up()
    await task

    failed = session.turns[1]
    assert failed["user_text"] == "What are your timings?"
    assert failed["agent_text"] == "Sorry, say that again?" and not failed["interrupted"]


class FailingTTS:
    """Raises TTSError after yielding `chunks_before_error` chunks of silence."""

    def __init__(self, chunks_before_error: int = 0) -> None:
        self.chunks_before_error = chunks_before_error
        self.calls = 0

    async def stream(self, text, **_):
        self.calls += 1
        for _chunk in range(self.chunks_before_error):
            yield bytes(1600)
        raise TTSError("provider down")


async def test_tts_failure_before_audio_falls_back_to_the_second_provider():
    fallback = FakeTTS()
    primary = FailingTTS()
    session, media, asr = make_session(FakeLLM(["Sure."]), primary, fallback)
    task = asyncio.create_task(session.run())
    await wait_for(lambda: len(session.turns) == 1)
    asr.say("Hello")
    await wait_for(lambda: len(session.turns) == 2)
    media.hang_up()
    await task

    assert fallback.texts == ["Hello, how can I help?", "Sure."]
    assert session.tts_characters == 0
    assert session.fallback_tts_characters == len("Hello, how can I help?") + len("Sure.")
    assert session.turns[0]["tts_ttfb_ms"] is not None


async def test_tts_failure_after_audio_started_is_not_repeated_on_the_fallback():
    fallback = FakeTTS()
    session, media, asr = make_session(FakeLLM([]), FailingTTS(chunks_before_error=1), fallback)
    task = asyncio.create_task(session.run())
    await asyncio.sleep(0.3)
    media.hang_up()
    await task

    assert fallback.texts == []  # the caller already heard part of the sentence
    assert session.tts_characters > 0 and session.fallback_tts_characters == 0


async def test_tts_failure_without_a_fallback_does_not_end_the_call():
    session, media, asr = make_session(FakeLLM([]), FailingTTS())
    task = asyncio.create_task(session.run())
    await asyncio.sleep(0.3)
    assert not task.done()
    media.hang_up()
    assert await task == "caller:callended"
