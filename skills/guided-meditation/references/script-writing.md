# Writing the script

How to write `script.json`: scene-specific narration in complete phrases with rests between
them, plus a music plan and an SFX plan. Narration, music, ambience and one-shot sounds are
separate layers. Timing and production directions live in data fields, never in spoken text.
Field definitions are in `script-schema.md`; a complete example is `../assets/example_script.json`.

## Inputs (from the brief)

| Input | Meaning |
| --- | --- |
| scene | the environment the listener is already in |
| scene_conditions | fixed or variable time of day and lighting |
| theme | the intention of the session |
| duration | target minutes; `duration_mode` approximate (default) or exact (only when required) |
| voice | the chosen voice's character; delivery qualities of any supplied reference |
| narration_style | desired phrasing; qualities of any reference script |
| perceptual_anchors | two or three recurring sensory motifs |
| music | enabled, character, presence, where it should be absent |
| sfx | ambience texture; one-shot events (bells) and where |
| closing_rest | planned and minimum final rest, and the closing line |

When a reference script is supplied, take its phrasing, restraint and variety as guidance. Do
not copy unsupported scenery, invented history, emotional claims, tag overuse or its duration.
When the user supplies their own script, keep their words exactly and only segment, pause and
plan around them.

## Duration math

Plan quiet narration at about 90 words per minute until the chosen voice has a measured pace
(after the audition, `synthesize.py --accept` prints it; most voices measure 120-170). Time
without narration is often 55-75 percent of the runtime; this is guidance, not a quota.

For a D-minute session, roughly D x 40 spoken words is a starting point, not a quota to fill.
Use 8-14 segments for sessions of 8 minutes or more (4-10 for shorter ones), with deliberately
different lengths: one brief invitation can be a whole segment; do not pad it into a paragraph
or give every segment the same number of sentences.

Rests generally grow: 5-10 s early, 20-45 s during sustained listening. A closing rest (a long
stretch of music and ambience before one short closing line) is part of the session; honour any
minimum, including rests of five minutes or more. The final segment has `pause_after_ms: 0`;
the file's ending comes from `session.tail_ms`.

`validate_script.py --write` computes every number (spoken words, speech seconds, both pause
budgets, padding, the estimate) and writes the `timing` block. Do not do this arithmetic by
hand. In approximate mode, keep comfortable speech, useful rests and the closing rest even when
the estimate exceeds the target, and report the honest difference. Rebalance to an exact length
only when that is an explicit requirement, and then by changing rests, never speech speed.

A lower voice speed is not evidence of calm and tends to produce hesitant, separated words. Let
meaningful rests provide space rather than stretching words. The measured timeline after
assembly, not the estimate, is the real length.

## Writing rules

- Speak to one listener, mostly in the present tense. The listener is already in the scene:
  never narrate their arrival or ask them to imagine it.
- Keep narration and the music brief true at every time of day the scene allows. A changing sky
  must not make the script false: avoid morning, sunlight, darkness, stars or light-dependent
  colours unless the session fixes them. Favour persistent features; keep visual invitations
  optional, with a listening alternative.
- Vary sentence openings: observations about the scene, direct invitations, occasional
  fragments, a natural "let's". Second person does not mean starting every sentence with "You".
- Vary phrase and sentence lengths; let a long flowing sentence sit beside a short one. Keep
  each thought clear. Use contractions and plain spoken language. Avoid a repeated
  instruction-then-reassurance template.
- Use permissions sparingly: repeated "You can", "There is no need", "Nothing needs to" sound as
  mechanical as repeated commands.
- Let scene observations stand without explaining their significance. Trust the listener and
  the rests. Do not narrate every change of attention.
- Sensory detail is restrained and specific: two or three concrete anchors for the whole script,
  returned to rather than accumulated. A light metaphor can grow from an existing anchor. Avoid
  elaborate imagery, invented facts about the place, or instructions to escape the scene.
- No plot, manufactured tension, climax or required resolution. An opening invitation and an
  open-ended close are welcome. A session the listener does not finish is a success.
- Do not instruct eye closure or specific gaze targets unless the scene supports them; offer any
  visual invitation as optional with an auditory or bodily alternative.
- No therapeutic, medical or outcome claims; no promises about how they will feel; do not assign
  the listener an emotion.
- Avoid generic meditation filler: "simply", "just allow yourself to", "gently", "as you",
  "notice how you begin to". The validator flags these.

## Phrases and pauses

- Phrases are the units of narration. Split a segment only where an intentional pause gives room
  for an experience, lets a thought land, or marks a meaningful shift. A phrase can hold several
  sentences; a segment with no deliberate internal pause is one phrase.
- Keep connected utterances intact ("Let's begin here"; a list like "spreading, meeting and
  slipping past each other"). A comma or an emphasized word is not a reason for a boundary.
  Never place a pause after "Let's" or between ordinary words.
- Each phrase's `pause_after_ms` is the minimum rest after the whole phrase. Every non-final
  phrase has a purposeful positive pause (typically 600-1500 ms); the last phrase of a segment
  has 0, because the segment's `pause_after_ms` owns the following rest.
- Write plain punctuation. No ellipses as pacing, no pause labels, durations or stage
  directions in spoken text; the voice would read them or perform them oddly.
- Assembly adds only the rest that the natural gap does not already provide, keeps breaths,
  and never touches the inside of a phrase.

## Audio tags

- Only `eleven_v3` performs tags. For every other model they are stripped before synthesis;
  never send bracketed directions to a model that would read them aloud.
- Permitted: `[whispers]`, `[sighs]`, `[exhales]`, `[inhales deeply]`. At most one per segment
  and four in the script, each one plausible for the chosen calm voice.
- Never `[short pause]` or `[long pause]` (pauses are data), never sound-effect or environment
  tags (those are the SFX layer), never emotional stage directions as prose ("she said softly").

## Music plan

- Write an instrumental prompt covering timbre, density, pulse, harmonic movement and production
  character (the Music API follows key, BPM and studio vocabulary well: "in D major", "no
  discernible pulse", "large soft plate reverb"). Match the scene and voice. Favour a consistent
  palette, few changes and long fades; no dramatic arc.
- Keep the voice intelligible and the scene's anchor recognizable: almost still pads or drones
  if melodic music feels too active; no recurring plucks, bright attacks, beats, builds or
  vocals. Never name artists, bands or songs (the API rejects them).
- Music has no speech, vocals or environmental effects; ambience is the SFX layer.
- Cue gain 1.0 means the calibrated quiet bed, not full scale. Each cue moves from the gain at
  its start to `target_gain` over `fade_ms` with a smoothstep curve.
- Anchor cues to a segment's `segment_start`, `speech_end` or `pause_end` plus `offset_ms`, or to
  `session_start`/`session_end`. These resolve on the measured timeline after assembly; never
  hard-code estimated timestamps.
- Keep cues chronological and fades non-overlapping; a fade may cross a segment boundary but
  must finish before the next cue and the session end.
- `music_free_pauses` lists segments whose entire following rest has music gain 0 (ambience-only
  interludes). Fade out before that rest starts; return no earlier than its end.
- Begin at gain 0 and end at gain 0. With a closing rest, keep the bed through it and fade in its
  final portion (anchor to the rest's `pause_end` with a negative offset) so the closing line is
  spoken over ambience alone.
- Loop needs are handled in the mix (long crossfades between compatible passages, phase
  continuous through muted interludes), so a 2-4 minute generation usually suffices.
- If music is disabled: `enabled: false`, `initial_gain: 0`, empty `cues` and
  `music_free_pauses`.

## SFX plan

- The ambience is the scene's own sound (rain, stream, surf, wind in leaves), present from the
  first second, recognizable under the voice, continuing after the last word and fading out
  with the tail. It loops every `duration_seconds` (max 30), so describe a steady texture;
  distinctive single events become audible repetitions.
- Describe what should be heard, concretely, including distance and space ("heard from inside a
  quiet room", "close and steady"). Negations ("no thunder") tend to put the thing in.
- One-shots are single events at anchors: a soft bell before the first word and after the last
  is a meditation convention; use `same_as` to reuse one sound. Keep them sparse and quiet.
- Ambience cues are optional (a gentle lift during a music-free interlude, for instance); the
  fade-in at the start and fade-out at the end come from `fade_in_ms` and `fade_out_ms`.
- The session envelope: `session.lead_in_ms` (ambience and bell before the first word, 4-8 s) and
  `session.tail_ms` (ambience after the last word, 15-30 s, at least `fade_out_ms`).

## Before validating

Review the narration for repeated openings, equal-length passages, filler, and time-of-day
assumptions. Check that every non-final phrase has a purposeful positive pause and every
segment's last phrase has 0, tags are within limits, the closing rest meets its minimum, every
cue anchor exists, fades fit, music-free interludes are silent, and music ends at 0. Then run
`validate_script.py --write` and fix every error; act on or justify each warning.
