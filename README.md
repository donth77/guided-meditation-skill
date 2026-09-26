# Guided Meditation Skill

A Claude Code skill that makes guided meditation audio with ElevenLabs. Describe the meditation
you want. Claude writes the script, helps you pick a voice by ear, records the narration, and can
add music and nature sounds.

## Overview

- A voice-only track
- Optional versions with music, nature sounds, or both
- WAV and MP3 files, plus the script and a short listening report

## How to use

1. Open Claude Code in the folder where you want your meditations saved. Type
   `/guided-meditation` and describe what you want: length, setting, voice, and which versions.
   Or just ask for a guided meditation; Claude picks up the skill on its own.
2. You approve the script.
3. You pick the voice from short samples of the opening.
4. Claude records the rest, making extra takes where needed to keep it smooth.
5. Claude generates music and nature sounds with ElevenLabs from short descriptions, or uses
   your own recordings, and mixes each version.

You see a cost estimate before anything is charged, and every take is kept.

## What you need

- Claude Code
- An ElevenLabs API key (music needs a paid plan)
- Python 3.9 or newer with numpy, and ffmpeg (`brew install ffmpeg` on a Mac)

Copy `.env.example` to `.env` and add your key, or set `ELEVENLABS_API_KEY`.

## Install

```bash
npx skills add donth77/guided-meditation-skill --skill guided-meditation -g -a claude-code
```

Or copy `skills/guided-meditation` into `~/.claude/skills/`.

## Example requests

```text
A 10-minute meditation for winding down after work, set on a quiet beach. Calm British
female voice, a little breathy. Voice only, plus a version with gentle music and waves.
```

```text
Turn my script (meditations/forest/script.txt) into a sleep meditation. Keep my words exactly.
```

```text
Segment 3 sounds rushed and the music is too loud. Fix just those.
```

## Costs

A 15-minute meditation with music and nature sounds usually costs a few thousand ElevenLabs
credits. Claude logs what each step actually cost. For a free trial run, ask for a rehearsal:
it uses a stand-in voice and placeholder sounds.