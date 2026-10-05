# AGENTS.md

Notes for agents and contributors working on this repository. `README.md` is for people using
the skill; `skills/guided-meditation/SKILL.md` is what Claude follows when it runs a session.

## Repository layout

```text
skills/guided-meditation/        the skill; installers (npx skills add) copy only this folder
  SKILL.md                       phases 0-8, gates, commands, credit rules, working rules
  scripts/                       one CLI per phase, shared code in gm_common.py
  references/                    loaded on demand by SKILL.md:
                                   script-writing.md, script-schema.md (script.json v5),
                                   voice-direction.md (voice choice, settings, listening lessons),
                                   audio-production.md (loops, cues, ducking, loudness),
                                   elevenlabs-api.md (endpoints, limits, observed costs, errors)
  assets/                        example_script.json, brief_template.md
  evals/evals.json               test prompts with expected outputs
README.md                        user-facing: plain English, keep it short
AGENTS.md                        this file
.env.example                     ELEVENLABS_API_KEY placeholder
.github/logo-{light,dark}.svg    README logo, one cut per GitHub theme (keep the two in step)
```

Ignored and local only: `.env` (the real key; never print or commit it), `samples/` (reference
material from an earlier project), `meditations/` (session folders with audio, often 100+ MB),
`.claude/` (a local link so Claude Code loads the skill from this checkout:
`mkdir -p .claude/skills && ln -s ../../skills/guided-meditation .claude/skills/guided-meditation`).

The README title is an HTML `<h1>` so the logo can sit above it. Keep its lines unindented: GitHub
turns every leading space inside a heading into a hyphen in the anchor (`#guided-meditation-skill`).

## Scripts

| Script | Purpose |
| --- | --- |
| `check_setup.py` | tools, key, tier, credits, billing status, output formats, models; `--probe` makes one tiny TTS request (~20 credits) |
| `voices.py` | `mine`, `suggest`, `library`, `similar` (library voices that sound like an audio file), `show`, `preview`, `add`, `design` / `create` (Voice Design previews; `create` uses a voice slot); `--measure` / `--voiced MIN-MAX` screen previews for breathiness |
| `validate_script.py` | every script rule, timing estimate, credit estimate (conversion included), segments that may pass the request length limit; `--write` fills the timing block |
| `render_script.py` | script.md, script.txt, timing.md (estimated, then measured after assembly) |
| `synthesize.py` | `--audition` (one or several voice ids; writes `.preview.mp3` with the script's rests), `--accept`, passage takes with request stitching (dropped for the rest of a run when a reading with it runs fast; `--no-context`), `--takes N --pick` (stops at the first reading within `--pace-tolerance`), `--retake`, `--select`, `--freeze`, `--list`, `--preview`, `--rescreen`, `--convert` (speech to speech). Accepting a conversion makes voice.json a guide + target recipe: guide readings in `voice/guides/`, only a good one converted (`--convert-guide SEG:TAKE`, `--force-convert`). Flags takes at the 23.7 s request limit |
| `assemble_voice.py` | voice-track.wav + timeline.json; splits only at phrase boundaries, counts the natural pause and inserts only the missing rest; `preview()` is reused for audition previews |
| `generate_music.py` | Eleven Music candidates from the prompt or `--composition-plan` (length from the plan), `--select`, `--keep-selection`, `--plan-only`, `bad_prompt` suggestions |
| `fit_music.py` | composed parts (`music/timed.json`: sources, anchors, stretchable sections) placed on the measured timeline with joins in rests -> `music/fitted.flac`, selected; rerun after retakes. A `loop` entry cuts a seamless loop from a source (`music/loop.flac`) and hands the track's music over to it before the end |
| `generate_sfx.py` | ambience loop takes and one-shots |
| `mix.py` | stems and the four versions: placement (`--music-offset`), looping or straight-through play, cue envelopes, ducking, loudness, peak ceiling, `--limit`; `--music FILE --tag NAME` for side-by-side music versions; with a fitted loop, `<slug>.music-loop.wav` (starts where the track stops) and a handover preview |
| `qa_report.py` | file, loudness, timing, cue and silence checks; `--transcribe` word diff (paid); listening checklist; `--tag NAME` for a tagged mix (run right after it: stems are shared) |
| `pipeline.py` | runs the remaining phases in order; plan and cost only unless `--yes`; `--mock` |

## Session folder

```text
meditations/<slug>/
  brief.md  script.json  script.md  script.txt  timing.md  voice.json  ledger.jsonl
  voice/    audition/take-NNNN.mp3 (+.json recipe and screening, .alignment.json, .preview.mp3)
            guides/<seg>/take-NN.mp3 (conversion recipe: the guide voice's readings)
            passages/<seg>/take-NN.mp3 (+.json, .alignment.json, .preview.mp3)
            selection.json  voice-track.wav  timeline.json
  music/    source-NN.mp3 (+.json)  selection.json  plan-*.json  timed.json  fitted.flac (+.json)  loop.flac
            sourced/ (licensed files and their licence notes)
  sfx/      ambience-NN.mp3  one-shots/SNN-NN.mp3  selection.json
  stems/    voice.wav music.wav sfx.wav (sample-aligned, at mix level)
  output/   <slug>.voice|.voice-music|.voice-sfx|.voice-music-sfx[.<tag>] .wav/.mp3, manifest[.<tag>].json
            <slug>.music-loop[.<tag>].wav (loop after the session), *-into-loop*.preview.mp3
  qa/       report[.<tag>].md, report[.<tag>].json, transcripts/
```

`voice.json` is the accepted recipe (voice, model, settings, seed, format, measured pace and
voiced share of the accepted audition); for a conversion it also holds the `guide` voice that
reads, and `method: speech_to_speech`. `timeline.json` is the measured clock every music and
SFX cue resolves against.

## Conventions

- The skill folder must stay self-contained: no imports or paths outside
  `skills/guided-meditation/`.
- Python 3.9+, standard library plus numpy; audio through ffmpeg/ffprobe subprocesses. Each script
  has inline `# /// script` metadata so `uv run` works without setup.
- Shared code lives in `scripts/gm_common.py`: API client (retries, error explanations,
  multipart), the launcher for other programs, audio I/O, level and voice measures
  (`voicing_ratio`, `inner_pauses`, `quiet_threshold`), envelopes, loop planning, script helpers.
  Extend it instead of duplicating.
- Other programs start in one place: `run_tool` and `open_tool` (ffmpeg, ffprobe, the mock voice)
  and `run_script` (the sibling scripts `pipeline.py` chains), all in `gm_common.py`. They take
  argument lists, never a shell command line, run only that fixed set of programs, and leave the
  API key out of the environment of everything but the sibling scripts. No other script imports
  `subprocess`.
- The API key is read from the environment or from `ELEVENLABS_API_KEY` in a `.env` (nothing else
  in the file is used) and is sent only to the API base, which must be https.
- Paid calls print an estimate, support `--dry-run` and `--max-credits`, check remaining credits,
  and append to the session's `ledger.jsonl` with the charge from the `character-cost` header.
  Estimates use documented rates and usually overstate the real charge.
- Takes are never overwritten (a retake adds a numbered take); frozen segments are never
  regenerated; every candidate stays on disk.
- Speech is never edited: no word-gap edits, splices inside a phrase, time-stretching or speed
  processing. Pauses are data (`pause_after_ms`), never tags, SSML or ellipses in the text.
- Screening numbers point at things to listen to; they never approve a take. The listener's ear
  decides, and reports say what was measured, not that a voice "sounds calm".
- When behavior changes, update SKILL.md and the relevant reference in the same change. Put
  listening lessons (what was tried, what the listener heard) in `references/voice-direction.md`.
  Keep SKILL.md about what to do; detail goes in references.

## Testing

Offline, no credits:

```bash
# whole pipeline with stand-in audio (macOS `say` or espeak for the voice, else tone bursts)
mkdir -p /tmp/gm-test && cp skills/guided-meditation/assets/example_script.json /tmp/gm-test/script.json
python3 skills/guided-meditation/scripts/pipeline.py /tmp/gm-test --mock --yes

# errors and undefined names
uvx ruff check --no-cache --select F,E9 skills/
```

On an existing session these are free: `synthesize.py SESSION --list | --preview TAKES |
--rescreen`, `validate_script.py SESSION`, `assemble_voice.py`, `mix.py`, `qa_report.py`
(without `--transcribe`). `--convert` has no mock mode.

Live tests cost credits: run `--dry-run` first and pass `--max-credits`. Observed charges and API
quirks (v3 refuses stitching context, library voices work by id without being added, and so on)
are in `references/elevenlabs-api.md`.

`evals/evals.json` holds prompts with expected outputs for evaluating the skill end to end.

## Security scan

The listing on skillsdirectory.com grades the skill with a static scanner: one finding per rule
per file, 25 points off for a critical finding, 15 for high, 8 for medium, and 75 is a B. Its
methodology says only SKILL.md counts, but at submission (October 2026) the listing also counted
the scripts, so keep them to the one finding that cannot go: process execution in `gm_common.py`
(critical; the skill runs ffmpeg). What it matched before, and what to avoid:

- `subprocess.run(` / `subprocess.call(` in five scripts: start programs through `gm_common.py`.
- `shutil.rmtree(` in `mix.py`: scratch folders come from `tempfile.TemporaryDirectory`.
- `pip install` in an error message, read as installing packages at run time: no package-manager
  commands in the scripts' text.

```bash
# gm_common.py should be the only file listed
grep -lE "subprocess|rmtree|(pip|npm) install" skills/guided-meditation/scripts/*.py
```

https://www.skillsdirectory.com/security/scan shows the grade and the findings per file: the
repo link scans what is published, a ZIP of the skill folder what a push would be graded as.
