# Guided Meditation Skill

An [Agent Skill](https://agentskills.io) for [Claude Code](https://code.claude.com/docs/en/skills)
that produces finished guided meditation tracks with ElevenLabs: a timed script, an auditioned
narrator voice, connected narration with exact rests, and optional generated music and sound
effects, looped, cued and loudness-matched.

Every session delivers a **raw voice track**, plus any of **voice+music**, **voice+sfx** and
**voice+music+sfx**.

| Skill | What it does |
| --- | --- |
| [`guided-meditation`](skills/guided-meditation/SKILL.md) | Brief → script (phrases, rests, closing rest, music and SFX cues) → voice audition (previews with the script's rests, screening for whisper strength, pace and breaks inside phrases) → passage synthesis with request stitching, optionally best of N takes per passage → assembly on a measured timeline → music (Eleven Music) and SFX (ambience loop, bells) → mix in four versions → QA with a listening checklist. Each phase has a gate, and the listener approves the voice on the script's own opening before any full run. |

## Requirements

- Python 3.9+ with numpy (or run the scripts with `uv run`, which installs numpy on the fly)
- ffmpeg and ffprobe on PATH (`brew install ffmpeg` / `apt install ffmpeg`)
- An ElevenLabs API key in `ELEVENLABS_API_KEY`, or in a `.env` file in the project (copy
  `.env.example` to `.env`; the scripts look for the nearest `.env` above the session folder and
  the working directory). The Music API
  needs a paid plan; 192 kbps MP3 needs Creator, and 44.1 kHz PCM needs Pro.

Check everything with:

```bash
python3 skills/guided-meditation/scripts/check_setup.py
```

## Installation

This repository keeps each skill in `skills/<name>/`.

```bash
# With the skills CLI (all projects / this project only):
npx skills add donth77/guided-meditation-skill --skill guided-meditation -g -a claude-code
npx skills add donth77/guided-meditation-skill --skill guided-meditation -a claude-code

# Manually:
mkdir -p ~/.claude/skills && cp -R skills/guided-meditation ~/.claude/skills/
```

To use the skill while working in a clone of this repository, link it into the project's
skills folder (`.claude/` is not committed; restart Claude Code if `.claude/skills` did not
exist when the session started):

```bash
mkdir -p .claude/skills && ln -s ../../skills/guided-meditation .claude/skills/guided-meditation
```

## Example prompts

```text
Make me a 10-minute guided meditation for winding down after work, set on a quiet beach.
Calm British female voice. I want the raw voice track and a version with gentle music and
ocean sounds.
```

```text
5-minute breathing meditation, voice only, with my voice Clara on multilingual v2.
Don't ask me questions.
```

```text
Here's my script (meditations/forest/script.txt). Turn it into a sleep meditation with a
5-minute rest before the last line, voice+music only. Keep my words exactly.
```

```text
Segments 03 and 06 of the waterfall meditation sound rushed. Redo just those and remix.
```

## How a session looks on disk

```text
meditations/<slug>/
  brief.md  script.json  script.md  timing.md  voice.json  ledger.jsonl
  voice/    auditions, passages (numbered takes with recipes, request ids, alignment),
            voice-track.wav, timeline.json
  music/    generated candidates        sfx/   ambience loops and one-shots
  stems/    voice.wav music.wav sfx.wav (sample-aligned, at mix level)
  output/   <slug>.voice.wav|mp3, .voice-music.*, .voice-sfx.*, .voice-music-sfx.*, manifest.json
  qa/       report.md (checks + listening checklist), report.json
```

## Costs

Every paid script prints an estimate, supports `--dry-run` and `--max-credits`, refuses to run
beyond the account's remaining credits, and logs each call with its actual charge (from
ElevenLabs' `character-cost` header) in `ledger.jsonl`. A 15-minute session with music and
ambience typically needs a few thousand credits. `--mock` rehearses the whole pipeline offline
with stand-in audio at no cost.

## Layout

```text
skills/guided-meditation/
  SKILL.md                 phases, gates, commands, working rules
  scripts/                 check_setup, voices, validate_script, render_script, synthesize,
                           assemble_voice, generate_music, generate_sfx, mix, qa_report, pipeline
  references/              script-writing, script-schema, voice-direction, audio-production,
                           elevenlabs-api
  assets/                  example_script.json, brief_template.md
  evals/evals.json         test prompts with expected outputs
```
