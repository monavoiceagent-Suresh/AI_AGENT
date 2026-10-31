# Voice Agent Platform

AI telecalling over Exotel. A caller's audio streams over Exotel's Voicebot WebSocket to this FastAPI service, which runs
**ElevenLabs Scribe Realtime** (ASR) → **OpenRouter or Groq** (LLM, chosen by `LLM_PROVIDER`, the other as the automatic fallback) → **Bakbak (Raya) streaming TTS** and plays the reply back,
with barge-in. Agents, calls, turns, outcomes, latency and cost estimates are stored in PostgreSQL.

There is no frontend. Agents and calls are managed through a bearer-protected JSON API; interactive docs are at `/docs`
(click *Authorize* and paste `APP_SECRET_KEY`).

```
Exotel ──wss──▶ /exotel/stream ─▶ VoiceSession ─▶ Scribe (ASR) ─▶ OpenRouter or Groq (LLM) ─▶ Bakbak (TTS) ─┐
   ▲                                  │ persist (PostgreSQL)                                              │
   └────────────── media / mark / clear ◀───────────────────────────────────────────────────────────────┘
```

## Run locally

Requires [uv](https://docs.astral.sh/uv/) and Docker.

```bash
cp .env.example .env          # then fill in the values (see comments in the file)
docker compose up -d --wait db
uv run uvicorn app.main:app --port 8000   # applies migrations on startup
```

`docker compose up -d --build` runs the app in a container as well (it points itself at the `db` container).
If another PostgreSQL already uses port 5432, set `POSTGRES_HOST_PORT` and the port in `DATABASE_URL`.

Development: `uv run pytest`, `uv run ruff check .`, `uv run ruff format .`. The API tests create a throwaway
database on the compose PostgreSQL and skip themselves if it is not running.

## Configure an agent

```bash
AUTH="Authorization: Bearer $APP_SECRET_KEY"
curl -X POST localhost:8000/agents -H "$AUTH" -H 'Content-Type: application/json' -d '{
  "name": "Sunrise Clinic", "language": "en", "workflow": "appointment", "is_inbound_default": true,
  "instructions": "You are the receptionist of Sunrise Clinic, open 9am-6pm Mon-Sat.",
  "greeting": "Hello, this is Sunrise Clinic. How can I help you today?",
  "collect_fields": ["name", "preferred_time", "service"]
}'
```

- `language`: `en`, `hi`, `ta`, `te`. `workflow`: `general`, `lead_qualification`, `faq`, `appointment`, `order`.
- `order` (shops, supermarkets): takes an order or answers an enquiry, reads the order back, and never promises
  price, stock or delivery time. Outcomes: `order_placed`, `enquiry_answered`, `callback_requested`.
- `knowledge` holds FAQ facts the agent may use; `collect_fields` lists details to gather.
- `instructions` may contain the placeholders `[[KNOWLEDGE]]`, `[[CALLER_CONTEXT]]` (inbound/outbound and, for
  outbound calls, the `contact_name` passed to `POST /calls`) and `[[CURRENT_DATETIME]]` (India time). They are
  filled in for every call; with `[[KNOWLEDGE]]` present the knowledge is inlined instead of appended.
- Optional per-agent `voice_id`, `tts_model_id`, `llm_model` override the `.env` defaults (`llm_model` replaces
  the primary provider's model, `OPENROUTER_MODEL` or `GROQ_MODEL`; the fallback uses its own).
- The voice comes from `BAKBAK_VOICES` by language (see *Text-to-speech* below). `voice_id` and `tts_model_id`
  override it, but must be set together because a Bakbak voice belongs to exactly one model.
- Exactly one agent can have `is_inbound_default: true`; it answers inbound calls.

## Language model (LLM)

`LLM_PROVIDER` chooses which LLM answers first: `LLM_PROVIDER=GROQ` (Groq primary, OpenRouter fallback) or
`LLM_PROVIDER=OPENROUTER` (OpenRouter primary, Groq fallback). Any letter case works; the default is `openrouter`.
The other provider is the automatic fallback when its key is set; without it the primary answers alone and its errors
are raised as they are. The primary's key must be set or the engine refuses to start. Both are called through the same
OpenAI-compatible streaming adapter (`app/providers/chat_completions.py`); `app/providers/failover.py` chooses.

- **OpenRouter:** `POST {OPENROUTER_BASE_URL}/chat/completions` (default `https://openrouter.ai/api/v1`) with
  `Authorization: Bearer $OPENROUTER_API_KEY`, model `OPENROUTER_MODEL`. `OPENROUTER_FALLBACK_MODEL`, the name this
  setting had when OpenRouter was always the fallback, is still read.
- **Groq:** `POST {GROQ_BASE_URL}/chat/completions` (default `https://api.groq.com/openai/v1`) with
  `Authorization: Bearer $GROQ_API_KEY`, model `GROQ_MODEL` (default `openai/gpt-oss-20b`).
- **Models:** each provider is asked for its own model. An agent's `llm_model` goes to the primary only.
- **Reasoning:** gpt-oss streams its thinking in a separate reasoning field before the answer in `delta.content`. Only
  the answer is spoken. Thinking tokens count against `GROQ_MAX_OUTPUT_TOKENS`, which is why it defaults to 1024.
- **When it falls back:** the primary times out, cannot connect, answers 429, or answers 5xx. Timeouts, connection
  errors and 5xx are retried first, waiting with exponential backoff and jitter between attempts
  (`LLM_RETRY_BACKOFF_BASE_MS`/`LLM_RETRY_BACKOFF_MAX_MS`, default 300–2000 ms); a 429 never retries inline — it goes
  straight to the fallback. By role, the primary defaults to a 4000 ms timeout and no retry (every retry adds a full
  timeout before the fallback starts) and the fallback to 8000 ms and one retry, since nothing follows it.
  `OPENROUTER_TIMEOUT_MS`, `OPENROUTER_MAX_RETRIES`, `GROQ_TIMEOUT_MS` and `GROQ_MAX_RETRIES` override this for that
  provider. Client errors (401 bad key, 404 unknown model) are **not** failed over, so misconfiguration stays visible.
- **Request-rate safety limits** (`app/providers/rate_limit.py`, `app/providers/budget.py`): each provider
  self-throttles to `OPENROUTER_MAX_REQUESTS_PER_MINUTE` / `GROQ_MAX_REQUESTS_PER_MINUTE` (default 18/60, shared by
  every concurrent call) — once reached, the next request fails fast with no HTTP call at all and the fallback
  answers instead. A 429 also opens a short cooldown (honouring `Retry-After` if the provider sends one; 1 s
  otherwise) so a just-limited provider is not asked again immediately. Separately, each phone call gets its own
  budget of at most `LLM_MAX_REQUESTS_PER_CALL` requests (default 40, counting both conversation turns and the
  post-call outcome extraction) so a stuck call can never make an unbounded number of requests. These are independent
  of each other and of the per-attempt retry above.
- **No replay:** the fallback is used only if the primary failed *before any text was produced*. If it fails after
  speaking has started, the call keeps what the caller already heard and the agent says its short apology instead.
- **Both fail:** the agent speaks the apology phrase and the caller's words are kept for the next turn.
  `MAX_CONSECUTIVE_LLM_FAILURES` (default 3) ends the call instead of apologising forever if responses keep failing
  one after another; any response that succeeds resets the count.
- **Duplicate utterances:** a committed transcript identical to the one just answered, delivered again within
  `LLM_DEDUP_WINDOW_MS` (default 700 ms — an ASR/websocket retransmit, not a caller repeating themselves), is dropped
  instead of being answered a second time. Partial transcripts, silence and the greeting never call the LLM at all.
- **Post-call outcome extraction** uses the same primary-then-fallback order and shares the call's request budget.
- **Logs** one line per request attempt (provider, model, a short request id, attempt number, status, latency) and
  name failures (for example `LLMError 503`) but never log the message content, a response body or credentials.
- **Cost:** OpenRouter reports a `cost` in each response and it is recorded as USD; Groq reports none, so a Groq-
  answered turn records 0.
- The engine does not execute tools, so no tool definitions are sent and a fallback can never repeat a tool call.

## Text-to-speech

**Bakbak** ([API docs](https://docs.litwizlabs.com)) is the primary TTS: `POST {BAKBAK_BASE_URL}/text-to-speech/stream`
with an `X-API-Key` header. It streams Server-Sent Events of base64 float32 PCM at the sample rate Exotel negotiated
(8/16/24 kHz), which the adapter converts to the 16-bit PCM Exotel expects. On barge-in the HTTP stream is closed,
so no further audio is received.

- **Voices are per language.** A Bakbak voice belongs to one model (`standard` or `m1`) and one language, and `m1`
  has only English and Hindi voices, so `BAKBAK_VOICES` maps each agent language (`en`, `hi`, `ta`, `te`) to a
  voice and model. Browse them with `GET {BAKBAK_BASE_URL}/voices` and pick by ear; the defaults were chosen by
  availability, not listening. English agents use Bakbak's `en-in` (Indian English).
- **Text:** Bakbak wants Hindi in Devanagari, and says Indian names inside English sound better written in
  Devanagari. The agent prompt already asks for native script, so this only matters for English agents.
- **Fallback (optional):** `TTS_FALLBACK_ENABLED=true` retries a sentence on the *other* provider if the primary fails (Bakbak primary: ElevenLabs takes over; ElevenLabs primary: Bakbak takes over with the voice for the agent's language), but only
  *before any audio was produced* (never mid-sentence, which would repeat words). Characters are metered per
  provider. Set `TTS_PROVIDER=elevenlabs` to switch fully; Telugu there needs `"tts_model_id": "eleven_v3"`.
- **Cost:** Bakbak publishes no price. Set `BAKBAK_INR_PER_1K_CHARS` to your contracted rate; until then
  `cost.tts` is `null` and `cost.complete` is `false` rather than guessed.
- **`GET /voices`** (admin) proxies whichever provider `TTS_PROVIDER` names and returns its real voices right
  now — never a hardcoded list. The dashboard's voice picker calls this instead of offering fixed labels.
- **A campaign's or test call's own voice**: `POST /campaigns/dispatch` and `POST /calls/test` accept an optional
  `voiceId`/`ttsModelId` (both or neither, like an agent's own `voice_id`/`tts_model_id`), checked against the
  configured provider exactly like an agent's. When given, it overrides the agent's current voice for that
  persona from then on; when omitted, the agent keeps whatever voice it already had.

## Exotel setup (manual, in the Exotel dashboard)

Verified against Exotel's [AgentStream docs](https://developer.exotel.com/docs/agentstream/websocket-protocol).

1. **Public URL.** Expose port 8000 over HTTPS (e.g. `ngrok http 8000`) and set `PUBLIC_BASE_URL`.
2. **Credentials.** *Settings → API Settings*: copy Account SID, API Key, API Token into `.env`. Use
   `https://api.exotel.com` or `https://api.in.exotel.com` as `EXOTEL_API_BASE_URL` depending on your account's region
   (an account only authenticates on its own region). `EXOTEL_CALLER_ID` is your ExoPhone.
3. **Flow.** *App Bazaar → create a flow* with a **Voicebot** applet whose URL is
   `wss://<API_KEY>:<API_TOKEN>@<your-host>/exotel/stream?sample-rate=16000`
   (URL-encode special characters in the key/token). Exotel sends them as HTTP Basic auth, which the server checks.
   The server logs this URL template at startup.
4. **Inbound.** Attach the flow to your ExoPhone. Inbound calls are answered by the `is_inbound_default` agent.
5. **Outbound.** Put the flow's ID (*App Bazaar → My Apps*) in `EXOTEL_APP_ID`, set `ENABLE_OUTBOUND_CALLS=true`, then:
   ```bash
   curl -X POST localhost:8000/calls -H "$AUTH" -H 'Content-Type: application/json' \
        -d '{"agent_id": 1, "to": "+919876543210"}'
   ```
   This calls Exotel's *Connect number to call flow* API (`POST /v1/Accounts/{sid}/Calls/connect.json`): Exotel dials the
   number, then runs the flow, which opens the WebSocket. Terminal status (busy, no-answer, failed) arrives at
   `/exotel/status?token=<EXOTEL_WEBHOOK_SECRET>` so unanswered calls are recorded.

Outbound calls to Indian numbers must comply with TRAI/DND rules; that is outside this service.

## Inspect results

```bash
curl localhost:8000/calls -H "$AUTH"        # recent calls
curl localhost:8000/calls/1 -H "$AUTH"      # call, turns, outcome, latency, cost
```

Each call stores: status and end reason, per-turn transcript with `interrupted` flag, workflow outcome and extracted
fields (`appointment_requested`, `qualified`, …) with a summary, usage (ASR seconds, TTS characters, LLM tokens), and:

- **Latency** per turn: `endpointing_ms` (last speech → commit), `llm_ttft_ms`, `tts_ttfb_ms`, and
  `response_latency_ms` (commit → first audio sent); summarised as avg/p50/p95 on the call.
- **Cost** in INR: ASR + TTS + LLM (from the provider's reported cost) + telephony, converted at `USD_TO_INR`, compared
  with `TARGET_COST_PER_CONNECTED_MINUTE_INR`. Telephony is included only when `EXOTEL_INR_PER_MINUTE` is set and
  `cost.complete` says whether every component was priced. ASR/TTS use ElevenLabs list prices unless you set `BAKBAK_INR_PER_1K_CHARS`; check them against your plan.

## How the conversation works

- ASR uses Scribe's VAD commit (`ENDPOINTING_MIN_SILENCE_MS` of silence ends the caller's turn).
- LLM output is split into sentences as it streams and each is synthesised immediately; audio is sent to Exotel in
  ≥100 ms chunks (multiples of 320 bytes, as Exotel requires) and at most `AUDIO_QUEUE_MAX_MS` ahead of playback.
- **Barge-in:** speech while the agent talks cancels generation, sends Exotel `clear`, and keeps in the history only the
  sentences Exotel confirmed as played (via `mark` events). If nothing was heard, the caller's words are merged into the
  next turn. If generation fails, the agent apologises in the agent's language.
- The agent ends the call by closing the socket (Exotel then moves to the flow's next applet) after saying goodbye;
  calls also end on caller hang-up, `SESSION_IDLE_TIMEOUT_SECONDS`, or `MAX_CALL_DURATION_SECONDS`.
- One concurrent call by default (`MAX_CONCURRENT_CALLS`); extra calls are refused.

## Troubleshooting

- **OpenRouter 404 "guardrail restrictions and data policy"**: the model's only provider trains on prompts and your
  account privacy settings exclude it. Change the setting at openrouter.ai/settings/privacy or pick another model.
- **Bakbak `404 unknown voice_id`**: the voice ID does not belong to the model you paired it with (or is not a
  voice at all). Re-check `GET /voices`; a voice and its `model_id` must match.
- **ElevenLabs 402 `paid_plan_required`**: library voices need a paid plan or credits on the API; use a default voice
  or upgrade. The API key needs Speech-to-Text and Text-to-Speech permissions.
- **TLS `CERTIFICATE_VERIFY_FAILED` behind a corporate proxy**: point `SSL_CERT_FILE` at a CA bundle that includes the
  proxy's root certificate (httpx uses it); for uv use `UV_NATIVE_TLS=1`. The same applies to `docker build`.
