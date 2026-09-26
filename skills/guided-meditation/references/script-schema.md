# script.json (schema 5)

The production script. Application-level data, not an ElevenLabs request: only phrase text is
ever sent to speech synthesis. `validate_script.py` enforces every rule here; `--write` fills the
computed fields. Schema 4 files (music only, ambience from an external scene) still validate,
without the sfx layer.

```jsonc
{
  "schema_version": 5,
  "title": "Rain on the roof",            // shown in reports and file tags
  "slug": "rain-on-the-roof",             // output file names; defaults from title
  "scene": "...",                          // where the listener already is
  "scene_conditions": "...",               // e.g. "time of day varies" (turns on time-of-day checks)
  "theme": "...",
  "voice": "...",                          // desired voice character (the recipe is voice.json)
  "language": "en",                        // optional; used for transcription
  "target_seconds": 600,
  "duration_mode": "approximate",          // or "exact" (explicit requirement only)
  "estimated_seconds": 612.4,              // computed by --write
  "session": { "lead_in_ms": 6000, "tail_ms": 20000 },
  "timing": { "words_per_minute": 90, ... },   // wpm is an input; the rest is computed by --write
  "phrase_pause_policy": { ... },          // written by --write if absent (documentation)
  "closing_rest": { ... },                 // optional
  "music": { ... },
  "sfx": { ... },
  "segments": [ ... ]
}
```

## segments

```jsonc
{
  "id": "04",                               // unique string; order in the list is play order
  "phrases": [
    { "text": "Breathing is already happening.", "pause_after_ms": 1200 },
    { "text": "In. And out.", "pause_after_ms": 0 }           // last phrase: always 0
  ],
  "pause_after_ms": 35000                   // rest after the segment; 0 for the final segment
}
```

- `text`: spoken words, plain punctuation, permitted audio tags only (`[whispers]`, `[sighs]`,
  `[exhales]`, `[inhales deeply]`; eleven_v3 only, max one per segment, four per script). At
  least one spoken word per phrase.
- Phrase `pause_after_ms`: the minimum rest after the complete phrase, measured word to word.
  Positive for every non-final phrase; 0 for the last phrase.
- Segment `pause_after_ms`: the rest from the segment's last word to the next segment's first
  word. The final segment has 0; the file's ending is `session.tail_ms`.

## closing_rest

```jsonc
{
  "segment_id": "11",                        // the segment BEFORE the final spoken line
  "duration_ms": 300000,                     // must equal that segment's pause_after_ms
  "minimum_duration_ms": 240000,             // duration_ms >= minimum
  "music": "Hold the bed through the rest, fade it in the last 20 s.",
  "final_spoken_segment_id": "12",
  "included_in_session_duration": true
}
```

## Anchors

Every cue and one-shot has an anchor that resolves on the measured timeline
(`voice/timeline.json`, after assembly) or, before synthesis, on the estimated one.

```jsonc
{ "segment_id": "06", "boundary": "speech_end", "offset_ms": 0 }
{ "boundary": "session_start", "offset_ms": 800 }      // no segment_id
```

| boundary | time |
| --- | --- |
| `segment_start` | the segment's first word |
| `speech_end` | the segment's last word ends |
| `pause_end` | the next segment's first word (= `speech_end` for the final segment) |
| `session_start` | 0 (start of the file) |
| `session_end` | end of the file (after `tail_ms`, or later if a one-shot runs past it) |

## music

```jsonc
{
  "enabled": true,
  "prompt": "Instrumental only. ...",        // timbre, density, pulse, harmony, production; no artist names
  "generation": { "model_id": "music_v2_5", "length_ms": 180000, "force_instrumental": true },
  "source_file": null,                      // a licensed recording instead of generation (path)
  "mix_notes": "...",
  "gain_reference": "1.0 is the calibrated quiet music bed",
  "looping": { "crossfade_ms": 12000 },     // long equal-power joins between compatible passages
  "initial_gain": 0,                        // always 0
  "music_free_pauses": ["08"],              // whole rest after these segments at gain 0
  "cues": [
    { "id": "M01", "anchor": { "segment_id": "02", "boundary": "segment_start", "offset_ms": 0 },
      "target_gain": 1, "fade_ms": 30000, "direction": "Enter slowly under the voice." }
  ]
}
```

Rules: cues chronological; each fade finishes before the next cue and the session end; gain
ends at 0; music-free rests stay at 0 throughout. `length_ms` 3000-600000 (the mix loops it).
When disabled: `enabled: false`, `initial_gain: 0`, `cues: []`, `music_free_pauses: []`.

## sfx

```jsonc
{
  "enabled": true,
  "ambience": {
    "enabled": true,
    "prompt": "Steady light rain on a wooden roof, heard from inside a quiet room ...",
    "generation": { "duration_seconds": 30, "prompt_influence": 0.35, "loop": true },
    "source_file": null,                    // a recording instead (looped with 3 s crossfades)
    "mix_notes": "...",
    "initial_gain": 0,                      // gain at t=0; the fade-in takes it to 1.0
    "fade_in_ms": 5000,
    "fade_out_ms": 15000,                   // ends at session_end; keep <= session.tail_ms
    "cues": []                              // optional, same format as music cues
  },
  "one_shots": [
    { "id": "S01", "prompt": "A single soft singing bowl strike ...", "duration_seconds": 7,
      "prompt_influence": 0.6, "anchor": { "boundary": "session_start", "offset_ms": 800 },
      "gain": 1.0, "direction": "Opening bell." },
    { "id": "S02", "same_as": "S01", "duration_seconds": 7,
      "anchor": { "segment_id": "12", "boundary": "speech_end", "offset_ms": 3000 }, "gain": 0.8 }
  ]
}
```

`duration_seconds` 0.5-30. `prompt_influence` 0-1 (higher is more literal). `same_as` reuses the
named one-shot's audio (no generation); `source_file` uses a recording. `gain` multiplies the
calibrated one-shot level (`mix.py --sfx-db`).

## voice.json (written by `synthesize.py --accept`)

```jsonc
{
  "voice_id": "...", "voice_name": "...", "model_id": "eleven_multilingual_v2",
  "voice_settings": { "stability": 0.6, "similarity_boost": 0.75, "style": 0.0,
                      "use_speaker_boost": true, "speed": 0.95 },
  "seed": 1234567, "output_format": "mp3_44100_192", "language_code": null,
  "accepted_take": "take-0003", "accepted_text": "...", "accepted_at": "...", "measured_wpm": 142.0
}
```

It can also be written by hand; flags on `synthesize.py` override it for one run.
