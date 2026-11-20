"""Sarvam AI (Bulbul) text-to-speech over WebSocket.

Reference: https://docs.sarvam.ai/api-reference-docs/api-guides-tutorials/text-to-speech/streaming-api/web-socket
(fetched 2026-11-20). Auth is `Api-Subscription-Key: <key>` (not Bearer). `send_completion_event=true` on the
connect URL makes the server send an `event`/`final` message once the flushed text has been fully spoken,
which is what this adapter waits on instead of guessing when the stream has ended. `output_audio_codec=linear16`
is raw 16-bit PCM with no container, matching this engine's TTS.stream contract directly (same idea as
Deepgram's `encoding=linear16`) — but unlike Deepgram, Sarvam's audio arrives base64-encoded inside JSON text
frames, not as raw binary frames.

Sarvam speaks all four of this engine's languages (en/hi/ta/te; see app/config.py sarvam_tts_speakers), unlike
ElevenLabs (no Telugu on an allowed model) or Deepgram Aura (English only).
"""

import base64
import json
from collections.abc import AsyncIterator

from websockets.asyncio.client import connect
from websockets.exceptions import InvalidStatus

from app.interfaces import TTSError

SPEAK_URL = "wss://api.sarvam.ai/text-to-speech/ws"


class SarvamTTS:
    def __init__(self, *, api_key: str, url: str = SPEAK_URL):
        self._api_key = api_key
        self._url = url

    async def stream(
        self,
        text: str,
        *,
        voice_id: str,
        model_id: str,
        language: str,
        sample_rate: int,
        previous_text: str,  # Sarvam has no continuity context across text messages; kept for the shared interface
        voice_settings: dict | None = None,  # Sarvam has no tone controls; kept for the shared TTS interface
    ) -> AsyncIterator[bytes]:
        url = f"{self._url}?model={model_id}&send_completion_event=true"
        config = {
            "type": "config",
            "data": {
                "language_code": f"{language}-IN",
                "speaker": voice_id,
                "output_audio_codec": "linear16",
                "speech_sample_rate": sample_rate,
            },
        }
        try:
            async with connect(url, additional_headers={"Api-Subscription-Key": self._api_key}) as ws:
                await ws.send(json.dumps(config))
                await ws.send(json.dumps({"type": "text", "data": {"text": text}}))
                await ws.send(json.dumps({"type": "flush"}))
                produced_anything = False
                async for raw in ws:
                    message = json.loads(raw)
                    kind = message.get("type")
                    data = message.get("data") or {}
                    if kind == "audio":
                        produced_anything = True
                        yield base64.b64decode(data["audio"])
                        continue
                    if kind == "error":
                        raise TTSError(f"Sarvam TTS error: {data.get('message')}")
                    if kind == "event" and data.get("event_type") == "final":
                        break
                if not produced_anything:
                    raise TTSError("Sarvam TTS produced no audio")
        except InvalidStatus as exc:
            raise TTSError(f"Sarvam TTS failed ({exc.response.status_code})") from exc
