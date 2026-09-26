---
name: guided-meditation
description: Produce a finished guided meditation audio track with ElevenLabs. Writes a timed script (phrases, rests, closing rest), auditions and locks a narrator voice, synthesizes connected passages, assembles the narration with exact rests, and optionally adds generated music and sound effects (nature ambience, bells), looped, cued and loudness-matched. Always delivers a raw voice track, plus voice+music, voice+sfx and/or voice+music+sfx versions. Use whenever someone wants a guided meditation, relaxation or sleep narration, breathing exercise, body scan, yoga nidra or mindfulness audio written, voiced, scored or remixed, even if ElevenLabs is not named, and for retakes or re-mixes of an existing session.
argument-hint: "[length, setting, voice, versions]"
---

# Guided meditation

Turn a request ("ten minutes by a lake, calm British voice, with gentle music") into delivered
audio by moving through gated phases. The voice is the product: most failed meditation tracks
fail on delivery (rushed or word-by-word phrasing, accent drift, a whisper that turns strained),
and no meter, transcript or timing number can detect that. So the listener approves the voice on
the script's own opening before any full run, and every later check is screening that points at
things to listen to, never a substitute for listening.

"Done" means: the requested files exist in `output/`, `qa_report.py` shows no FAIL, the listener
accepted the voice audition and has the final listening checklist, and the report states the
measured length, levels, credits spent, and anything not yet approved by ear.

## Runtime

- Python 3.9+ with numpy, and ffmpeg/ffprobe on PATH. Run scripts as
  `python3 scripts/<name>.py`, or `uv run scripts/<name>.py` (installs numpy on the fly). Paths
  below are relative to this skill's folder; every script takes `--help`.
- `ELEVENLABS_API_KEY` from the environment or the nearest `.env` above the session folder or
  the working directory. Never print it.
- Start with `check_setup.py`: it reports tier, credits left, billing status (a past-due
  subscription makes ElevenLabs refuse generation with `payment_required`), the default output
  format (mp3_44100_192 needs Creator; 44.1 kHz PCM needs Pro) and the available models.
- One folder per session, default `meditations/<slug>/` in the working directory:

```text
brief.md  script.json  script.md  script.txt  timing.md  voice.json  ledger.jsonl
voice/audition/take-0001.mp3 (+.json, .alignment.json, .preview.mp3)   voice/passages/<seg>/take-01.mp3 ...
voice/selection.json  voice/voice-track.wav  voice/timeline.json
music/source-01.mp3 (+.json)  music/selection.json
sfx/ambience-01.mp3  sfx/one-shots/S01-01.mp3  sfx/selection.json
stems/voice.wav music.wav sfx.wav      output/<slug>.voice.wav|mp3 ... manifest.json    qa/report.md
```

| Script | Purpose |
| --- | --- |
| `scripts/check_setup.py` | tools, key, tier, credits, billing status, formats, models; `--probe` proves generation works |
| `scripts/voices.py` | `suggest`, `mine`, `library`, `similar` (voices that sound like a take), `show`, `preview`; `--measure` screens previews for whisper strength (free); `add` is optional (uses a voice slot: ask first) |
| `scripts/validate_script.py` | every script rule, estimated timing, credit estimate; `--write` fills the timing block |
| `scripts/render_script.py` | script.md (phrases, pauses, cue notes), script.txt, timing.md (estimated, then measured) |
| `scripts/synthesize.py` | `--audition` (with previews at the script's rests), `--accept`, passages as numbered takes with stitching; `--retake`, `--freeze`, `--select`, `--list`, `--preview`, `--rescreen`; `--convert` re-voices a take (speech to speech, auditions only) |
| `scripts/assemble_voice.py` | voice-track.wav + timeline.json: splits only at phrase boundaries, adds only missing rest |
| `scripts/generate_music.py` | Music API candidates from the script's prompt; `--select`, `--plan-only`, bad_prompt handling |
| `scripts/generate_sfx.py` | ambience loop and one-shots from the script; several ambience takes alternate |
| `scripts/mix.py` | calibrated stems and the requested versions; looping, cues, ducking, loudness, peak ceiling |
| `scripts/qa_report.py` | file, loudness, timing, cue and silence checks; `--transcribe` word diff; listening checklist |
| `scripts/pipeline.py` | runs the remaining phases; prints the plan and cost unless `--yes`; `--mock` rehearses offline |

Read `references/script-writing.md` and `references/script-schema.md` before Phase 1,
`references/voice-direction.md` before Phase 2, `references/audio-production.md` before
Phases 5-7, and `references/elevenlabs-api.md` when an API call fails or costs matter.

## Credits

Every paid script prints an estimate and supports `--dry-run` and `--max-credits`, refuses to
start above the account's remaining credits, and appends each call to `ledger.jsonl`. The
exact charge comes from ElevenLabs' `character-cost` response header where the API sends one
(TTS, sound effects); the account counter lags by minutes. Estimates use documented rates and
usually overstate the real charge. Rough scale for a 15-minute session: narration about 2,000
characters, music 3 minutes looped, a 30 s ambience loop and a bell: a few thousand credits.
Show the estimate and get a yes before the first paid step of a session unless the user already
authorized spending; retakes and extra candidates multiply their share.

## Gate protocol

At each gate, show the user the artifact (file paths to play, the rendered script, the numbers)
and what is still unapproved. Continue on approval. If the user said not to ask, continue on
the defaults, keep every take and candidate on disk, and say in the final report which gates
were passed by default rather than by ear. Never describe a voice as calm, British, whispered
or natural because a metric passed; say what was measured and what needs listening.

## Phase 0: brief

Write `brief.md` from `assets/brief_template.md`. Ask only for what changes the result and
cannot be inferred: duration, versions wanted (voice is always made; voice+music, voice+sfx,
voice+music+sfx), voice character (gender, accent), how whispered it should be (plain calm, a
little breath, soft whisper, heavy whisper or full whisper; "whispery" alone is ambiguous, so ask),
and a budget ceiling if spending matters. Offer these defaults and proceed with them when the user says "just go":
about 10 minutes (approximate), a calm voice from the account chosen with `voices.py suggest`,
`eleven_multilingual_v2`, scene ambience plus soft opening and closing bells, a quiet
instrumental bed, all four versions, 6 s lead-in, 20 s tail, a closing rest of about 15 percent
of the session, WAV + MP3 at -18 LUFS.

A user-supplied script keeps its words. The work is segmentation, pauses, closing rest and the
music/SFX plan; flag, don't silently fix, anything that breaks the writing rules.

## Phase 1: script

Write `script.json` (schema 5) following `references/script-writing.md`. Start from the shape of
`assets/example_script.json`; take its structure, not its words. The script is data: spoken
text in phrases, rests as `pause_after_ms`, music and SFX as cues anchored to segment boundaries,
never timing or production notes inside spoken text.

```bash
python3 scripts/validate_script.py meditations/lake --write     # fix every error; act on or justify warnings
python3 scripts/render_script.py meditations/lake               # script.md, script.txt, timing.md
```

Gate: the user reads `script.md` (phrases with pause lines and cue notes) and `timing.md`
(estimated segment times, closing rest, cue intervals). In approximate mode the estimate may
differ from the target; report it, never pad or speed up to hit it.

## Phase 2: voice

```bash
python3 scripts/check_setup.py --session meditations/lake
python3 scripts/voices.py suggest --accent british --gender female --library --measure \
  --out meditations/lake/voice/previews                               # free: previews + whisper strength
python3 scripts/voices.py library --search calm --accent british --voiced 70-80 --page-size 50 \
  --out meditations/lake/voice/previews                               # only previews in the brief's band
python3 scripts/synthesize.py meditations/lake --audition --voice-id ID1,ID2,ID3   # shortlist, same text
python3 scripts/synthesize.py meditations/lake --audition --voice-id VOICE_ID --takes 2
python3 scripts/synthesize.py meditations/lake --audition --speed 0.9 --stability 0.6    # vary one thing
python3 scripts/synthesize.py meditations/lake --list          # every audition with its numbers
python3 scripts/synthesize.py meditations/lake --preview 0003,0005   # re-render previews after re-phrasing (free)
python3 scripts/synthesize.py meditations/lake --accept take-0003
```

Shortlist three or four voices whose previews measure in the band for the brief's breath level
(generated takes run a little whispier than previews), audition them in one round, then refine
the chosen voice's settings.

The audition text is the script's own first segment, so an accepted take is reused as that
segment (frozen) and never regenerated. Library voices work directly by id (no voice slot).
Each take also gets `take-NNNN.preview.mp3`: the same audio with the script's phrase rests
inserted as assembly will insert them. Play the previews; the raw take has the model's own
pauses only. The phrase boundaries are the script's, so when the listener wants the pauses in
different places, re-split the phrases (same words and punctuation) and rerun `--preview`: no
retake needed. Compare takes by ear against the brief: accent held, calm without dragging,
phrases connected, whisper at the strength asked for, tags performed rather than spoken. Each
take prints three screening numbers: words per minute (flagged fast above 165 or slow below
90), `voiced` (share of speech with pitch: ~0% full whisper, 30-45% heavy whisper, 50-65% soft
and breathy, ~80% ordinary speech) and pauses inside phrases (dips of 0.2 s or more, at least
20 dB under the speech, between words with no punctuation; breath and room tone count as quiet:
the word-by-word delivery listeners reject). Whisper strength lives in the voice, not the settings: when a
whisper is too heavy or too light, change the voice. When the listener likes a voice whose delivery
breaks phrases whatever the settings, keep the sound and change the delivery:
`voices.py similar TAKE.mp3` finds library voices that sound like it, and `--convert TAKE
--voice-id ID` re-voices a connected take of another voice in it (speech to speech; timing from
the guide, timbre from the target, auditions only for now). See "Pauses inside phrases" in
`references/voice-direction.md`. After `--accept`, set
`timing.words_per_minute` to the measured pace it suggests and rerun `validate_script.py --write`
so the estimate reflects this voice. Model and settings guidance, and what each failure
usually means, are in `references/voice-direction.md`.

Gate: the listener approves the opening. Before the full run, audition the chosen recipe on a
longer passage too (`--audition --segments 02 --takes 2`): a short opening can come out right by
luck while the voice's ordinary delivery breaks phrases everywhere else. A model change or a new
voice needs a fresh audition. If the opening fails, keep working on the opening; do not expand
to the full script.

## Phase 3: narration

```bash
python3 scripts/synthesize.py meditations/lake --dry-run       # requests, stitching context, credits
python3 scripts/synthesize.py meditations/lake                 # every segment without a take
python3 scripts/synthesize.py meditations/lake --takes 3 --pick   # best of 3 per passage
python3 scripts/synthesize.py meditations/lake --list
```

Generation is not repeatable across texts: the same voice, settings and seed can read one
passage connected and the next with breaks, and some voices break phrases often. When the
audition showed breaks in any take, or the listener has struggled with consistency, generate
three or four takes per passage with `--pick`: it keeps the take with the fewest breaks inside
phrases, then the one closest to the accepted audition's pace and breathiness, and reports
passages where no take was clean or the pick drifted from the accepted voice. Every take stays
on disk; the listener hears the picks, and a pick is overruled with `--select`. It multiplies
the narration cost (still small next to music).

One request per segment keeps each passage connected. With `eleven_multilingual_v2` and the
flash/turbo models, request ids of neighbouring selected takes younger than two hours are sent
as `previous_request_ids`/`next_request_ids` (request stitching); otherwise surrounding text is
sent as context. `eleven_v3` accepts no context at all (no stitching, no previous/next text), so
each v3 passage stands alone; it is the only model that performs audio tags, and tags are
stripped for every other model.

Retakes replace complete passages: `--segments 03,06 --retake` adds a take and selects it; the
old take stays on disk (`--select 03=take-01` restores it). `--freeze 01,02,04` protects kept
passages; do not regenerate good passages because another one needs work.

## Phase 4: assembly

```bash
python3 scripts/assemble_voice.py meditations/lake
python3 scripts/render_script.py meditations/lake              # timing.md gains measured columns
```

Each passage is split only at its authored phrase boundaries (alignment corroborated by the
waveform). The natural gap counts once and only the missing rest is added; breaths and exhales
before the first word are kept. `voice/timeline.json` holds the measured `segment_start`,
`speech_end` and `pause_end` of every segment: the clock every music, ambience and SFX cue
resolves against. Screening notes (a phrase much faster than the rest, a silence inside a
phrase, clipping, missing alignment) are places to listen, not verdicts. The raw voice track
exists from this point.

## Phase 5: music (voice+music, voice+music+sfx)

```bash
python3 scripts/generate_music.py meditations/lake --dry-run
python3 scripts/generate_music.py meditations/lake --candidates 2
python3 scripts/generate_music.py meditations/lake --select source-02
```

Generate a few minutes (default 180 s); `mix.py` loops it with long crossfades to cover the
session. Prompts must not name artists, bands or songs (the API answers `bad_prompt` with a
suggested prompt; `--accept-suggestion` uses it). A user-supplied licensed recording goes in
`music.source_file` or `mix.py --music`. Gate: listen to the candidates for vocals, a beat,
swells or bright events before mixing.

## Phase 6: sound effects (voice+sfx, voice+music+sfx)

```bash
python3 scripts/generate_sfx.py meditations/lake --dry-run
python3 scripts/generate_sfx.py meditations/lake                          # ambience + one-shots
python3 scripts/generate_sfx.py meditations/lake --ambience --candidates 2
python3 scripts/generate_sfx.py meditations/lake --select ambience=ambience-01+ambience-02
```

The ambience is a seamless loop of up to 30 s, so it must be a steady texture: a distinctive
event (a bird call, a thunder clap) repeats audibly every loop. Put single events in
`one_shots`; a one-shot with `same_as` reuses another's audio for free. Several selected
ambience takes alternate in the mix. Gate: listen for repeating events and clipped edges.

## Phase 7: mix

```bash
python3 scripts/mix.py meditations/lake --outputs voice,voice+music,voice+sfx,voice+music+sfx
python3 scripts/mix.py meditations/lake --outputs all --music-db -19 --formats wav,mp3,m4a
```

The voice is normalised to `--lufs` (default -18, measured as dual mono) and sits at the same
level in every version; beds are set relative to it (`--music-db -16`, `--ambience-db -18`,
`--sfx-db -10`) with slow passage-level ducking. Every version shares one gain, lowered for all
if any would pass the `--peak` ceiling (-1.5 dBTP); `--limit` catches peaks with an oversampled
limiter instead. `--normalize integrated` levels each whole track instead (for platforms that
normalise whole tracks). Re-mixing is free: adjust levels after listening and rerun; nothing
is regenerated. `references/audio-production.md` explains the stems, loop joins, cue envelopes
and when to change which level.

## Phase 8: QA and handover

```bash
python3 scripts/qa_report.py meditations/lake                  # add --transcribe for a word diff (paid, small)
```

Report to the user in this order: the files per version (paths), measured length against the
target, voice loudness and true peak, what the automated checks found (FAIL/WARN), the
screening notes worth a listen, credits spent (from the ledger), and the listening checklist in
`qa/report.md`. Say which gates were approved by ear and which by default.

## Faster routes

- `pipeline.py SESSION --outputs all` prints what is missing and its cost; add `--yes` to run
  synthesize → assemble → music → SFX → mix → QA, skipping finished steps. It still needs an
  accepted `voice.json`; the audition gate is not skipped.
- `--mock` on synthesize, generate_music, generate_sfx and pipeline rehearses the whole chain
  offline (system TTS voice, synthetic drone, rain and bell) to check timing, cues and mix
  plumbing without credits. Mock output is marked and never delivered.

## Revisions

| Request | Do |
| --- | --- |
| "Segment 3 sounds rushed" | listen; `synthesize.py --segments 03 --retake --takes 2`; pick; assemble; mix |
| "Keep everything except 5 and 7" | `--freeze` the rest, retake 05 and 07, assemble, mix |
| "Music is too loud / too busy" | `mix.py --music-db -19` (free); busy: new candidate or a stiller prompt |
| "Add ocean sounds" | add or enable `sfx.ambience`, validate, `generate_sfx.py`, `mix.py --outputs ...+sfx` |
| "Make it longer" | lengthen rests or add segments (validate, render, synthesize only the new ones) |
| "Different voice" | new audition and accept; every passage is regenerated with the new recipe |

## Working rules

- Listening approves; numbers screen. Word timing, transcripts, loudness and speech rate never
  establish accent, calm or natural phrasing.
- Never synthesize single words, splice inside a phrase, edit word gaps, or time-stretch or
  speed-process speech. A broken phrase is regenerated as a whole passage.
- Pauses are data (`pause_after_ms`), never tags, SSML breaks or ellipses in the text.
- Audio tags only for eleven_v3, at most one per segment and four per script.
- A lower speed setting is not calm: below ~0.9 voices tend to separate words; very low
  stability (below ~0.3) tends to rush and wander. Prefer a naturally calm voice.
- Keep the voice consistent: same voice, model and settings for every passage; a model change
  needs a fresh audition even with the same voice id.
- Music is instrumental, steady and behind the voice; ambience is a separate layer; loop phase
  runs continuously through muted interludes; music begins and ends at gain 0.
- Report the measured duration and the actual credits; never label an estimate as measured.
