# ElevenLabs API notes

What the scripts call, with limits, costs and failure modes. Checked against the API reference
and live calls in September 2026; `check_setup.py` shows the live account and model list.
Base URL `https://api.elevenlabs.io` (override with `ELEVENLABS_API_BASE`; https only, because the
key is sent to it), header `xi-api-key`. The key comes from the environment or a `.env`: only
`ELEVENLABS_API_KEY` is read from the file, and the key is not handed on to ffmpeg or the other
programs the scripts start.

## Endpoints used

| Script | Call | Notes |
| --- | --- | --- |
| synthesize | `POST /v1/text-to-speech/{voice_id}/with-timestamps?output_format=` | JSON `audio_base64` + `alignment` (characters with start/end seconds) + `normalized_alignment`; falls back to the plain endpoint if timestamps are refused |
| synthesize | body | `text, model_id, voice_settings{stability, similarity_boost, style, use_speaker_boost, speed}, seed (0-4294967295), previous_text, next_text, previous_request_ids (max 3), next_request_ids (max 3), apply_text_normalization, language_code` (not for multilingual_v2) |
| generate_music | `POST /v1/music?output_format=` | `prompt` + `music_length_ms` (3,000-600,000) + `force_instrumental` + `model_id` (`music_v2_5`, `music_v2`, `music_v1`), or `composition_plan`; `seed` only with a plan; response audio with a `song-id` header; paid plans only |
| generate_music | `POST /v1/music/plan` | composition plan from a prompt (`--plan-only`) |
| generate_sfx | `POST /v1/sound-generation?output_format=` | `text, duration_seconds (0.5-30), prompt_influence (0-1, default 0.3), loop (bool), model_id eleven_text_to_sound_v2` |
| synthesize `--convert` | `POST /v1/speech-to-speech/{voice_id}?output_format=` (multipart) | `audio`, `model_id` (`eleven_multilingual_sts_v2`, `eleven_english_sts_v2`), `voice_settings` (JSON string), `seed`, `remove_background_noise`; returns audio of the same duration as the input, so the guide's alignment still applies |
| voices `design` / `create` | `POST /v1/text-to-voice/design`, `POST /v1/text-to-voice` | `voice_description`, `text` (100-1000 chars), `model_id` (`eleven_ttv_v3`, `eleven_multilingual_ttv_v2`), `seed`, `guidance_scale`; returns three previews with `generated_voice_id`; `create` saves one (`voice_name`, `voice_description`, `generated_voice_id`) and takes a voice slot. Charged one credit per preview character, once, with no cost header |
| voices `similar` | `POST /v1/similar-voices` (multipart) | `audio_file`, `top_k`, `similarity_threshold`; returns library voices, most similar first; free |
| qa_report | `POST /v1/speech-to-text` (multipart) | `model_id scribe_v2, file, language_code, timestamps_granularity=word`; returns `text` and `words` |
| voices | `GET /v2/voices`, `GET /v1/voices/{id}`, `GET /v1/shared-voices`, `POST /v1/voices/add/{owner}/{voice}` | `fine_tuning.state` per model, `high_quality_base_model_ids`, saved `settings`, `verified_languages` |
| check_setup, budget guards | `GET /v1/user/subscription`, `GET /v1/models` | tier, status, `character_count`/`character_limit`, open invoices; per-model `character_cost_multiplier`, max characters |

Response headers worth keeping (the scripts store them per take): `request-id` (for stitching),
`history-item-id`, `character-cost` (credits actually charged; sent by TTS and sound effects),
`song-id` (music).

## Request stitching

`previous_request_ids`/`next_request_ids` condition a generation on neighbouring generations for
continuous prosody. Rules: at most 3 each; the ids must be under two hours old (the scripts use
110 minutes); `previous_text` is ignored when previous ids are sent. `eleven_v3` refuses both
the ids and `previous_text`/`next_text` (HTTP 400 `unsupported_model`, confirmed live), so v3
passages are generated without context. The scripts only use ids from takes with the same voice and model, and never use the id
of a take that the same run is about to replace.

Context can change the pace. One library voice read the same passages 40-60 percent faster with
context (ids or text) than without; three others were unaffected. Auditions are generated
without context, so `synthesize.py` stops sending it for the rest of a run once a reading with
context comes out faster than the accepted audition (`--no-context` from the start).

## Output formats and tiers

- `mp3_44100_128` all tiers; `mp3_44100_192` Creator and above; `pcm_44100` and `wav_44100` Pro
  and above. Raw PCM from TTS is 16-bit mono little-endian; `synthesize.py` wraps it in a WAV.
- The scripts pick `mp3_44100_192` on Creator+ and `mp3_44100_128` below, and fall back to
  128 kbps if a format is refused. Everything is decoded to 44.1 kHz float for processing.

## Limits

| Model | Characters per request | Credits per character (documented) |
| --- | ---: | ---: |
| eleven_v3 | 5,000 | 1.0 |
| eleven_v3_conversational | 5,000 | 0.5 |
| eleven_multilingual_v2 | 10,000 | 1.0 |
| eleven_flash_v2_5 / turbo_v2_5 | 40,000 | 0.5 |

Speed 0.7-1.2. eleven_v3 stability is one of 0.0, 0.5, 1.0. SSML `<break>` tags work only on v2
models and destabilize long generations; this skill never uses them.

Audio length (observed, September 2026): no eleven_multilingual_v2 request returned more than
23.684 s. Of 174 readings of up to 245 characters, 34 came back exactly that long: all words
present, the pace squeezed to fit, the last breath clipped. Longer texts were not tested. A
speech-to-speech conversion returns the length of its input, so it inherits the limit from its
guide reading. v3 readings ran to 25.8 s. `validate_script.py` warns about segments that may
reach the limit; `synthesize.py` flags takes that did (`at_length_limit` in the take's JSON).

## Costs

Estimates use documented rates; the charge is in the `character-cost` header. Observed on
Starter and Creator accounts in September 2026: multilingual_v2 charged 0.5 credits per character (half the
documented multiplier), sound effects 10 credits per second (documented: 40 per second when a
duration is given), speech to speech 10 credits per second of input (estimated at 1,000 per
minute), music about 700-900 credits per generated minute (no cost header). Plans
and discounts differ; trust the ledger's header totals over the estimates, and the account
balance (`check_setup.py`) after it catches up (the counter lags by minutes).

## Errors and what the scripts do

| Error | Meaning | Handling |
| --- | --- | --- |
| 401 `payment_required` / `payment_issue` | failed or incomplete subscription payment | stop, exit 2, tell the user to pay the latest invoice; `check_setup.py` flags a `past_due` status before any spend |
| 401 `quota_exceeded` | not enough credits | stop, exit 2 |
| 401 `invalid_api_key` | key rejected | stop, exit 2 |
| 400/404 `voice_not_found` | wrong id | check the id. Voice Library voices work in TTS by id without being added (confirmed live); `GET /v1/voices/{id}` answers `voice_not_found` for some of them, so the scripts look the name up in `/v1/shared-voices?search={id}` |
| 400/422 mentioning previous/next/request ids | stitching context refused | retried once without context (recorded in the take's meta) |
| 400/403 mentioning output_format | format not in the tier | retried with `mp3_44100_128` |
| 400 `bad_prompt` / `bad_composition_plan` (music) | prompt names an artist, band, song or lyrics | prints the suggested prompt; `--accept-suggestion` retries with it |
| 429, 5xx, network | rate limit, overload | retried with backoff (Retry-After honoured). A generation retried after a network timeout may also have been billed for the lost attempt, which the ledger cannot see; compare with the account balance |

## MCP

An ElevenLabs MCP server (tools like `search_voice_library`, `text_to_speech`, `play_audio`) is
handy for quick exploration in a live session. Build deliverables through the scripts anyway,
so every take has its recipe, request id, alignment and ledger entry.
