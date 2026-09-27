# Audio production

What `assemble_voice.py`, `mix.py` and `qa_report.py` do, why, and which knob to turn after
listening. All processing is 44.1 kHz float until the final encode.

## The clock

`voice/timeline.json` is written by assembly from the real audio. For every segment it holds
`segment_start_ms` (first word), `speech_end_ms` (last word ends) and `pause_end_ms` (next
segment's first word), plus phrase times, natural gaps and inserted rests. Every music cue,
ambience cue and one-shot resolves against it, so cues follow the actual delivery, not the
estimate. `session_start` is 0; `session_end` is the file end (`tail_ms` after the last word,
extended if a one-shot needs room to decay).

## Assembly

- One take per segment (the selected one). The passage is split only at its authored phrase
  boundaries: the character alignment from ElevenLabs locates each boundary, and the waveform
  corroborates it (alignments often start the first character at 0.0 s before a breath, and can
  run a little early or late).
- At each boundary the natural gap (last word of phrase to first word of the next) counts once.
  Only the missing rest is inserted, at the quietest point of the gap, with 5 ms fades. Breaths
  stay. If the natural gap already exceeds the requested pause, nothing is added.
- Segment edges keep up to 1.5 s of audible material before the first word (breaths, performed
  exhales) and 1 s after the last word, trimming only silence. Segment rests are measured word to
  word and include that kept material.
- Speech threshold: 40 dB below the loudest 30 ms frame (whispered onsets and decays sit 35-45 dB
  down); silence: 52 dB below. Screening flags phrases faster than ~185 or slower than ~70 words
  per minute, silences over 1.2 s inside a phrase, clipped samples, and missing alignment.
- The voice track is unnormalised float; loudness is set in the mix.

## Music

1. **Preprocess** (skip with `--no-music-eq`): high-pass 40 Hz, a 3 dB dip around 3 kHz (the
   presence range a quiet voice needs), a 1.5 dB high shelf cut above 9 kHz, and a gentle 2:1
   compressor above -22 dBFS with slow attack and release to smooth swells.
2. **Loop**: choose a restart point after the intro and a crossfade point before the outro fade
   (both from the loudness contour), then search nearby positions for the pair with the closest
   level and spectrum. Join with an equal-power crossfade of `looping.crossfade_ms` (default
   12 s). Terminal silence, MP3 padding and the outro fade never enter the loop. The first pass
   plays from the source's start; later passes restart at the chosen point.
3. **Cover the session** continuously: the bed runs from t=0 to the end whether or not it is
   audible, so muted interludes keep the loop phase and returning music continues the same
   texture. **Placement** (`--music-offset auto`): the source's activity (loudness, note onsets,
   brightness) is mapped, and the starting offset is chosen so its calmest stretches sit under
   speech and its busier moments and loop joins fall in rests; the manifest lists each passage's
   music as calm, moderate or busier, and any strong musical event under speech. A source at
   least as long as the session plays straight through with no join. `--music-offset SECONDS`
   fixes the start (and keeps the file's leading silence, for audio already on the session clock).
4. **Calibrate**: the loop's loudness is measured and set so cue gain 1.0 sits `--music-db`
   (default -16) below the voice's loudness.
5. **Envelope**: the script's cues (smoothstep ramps from the current gain), times slow ducking
   (`--duck-db`, default -3 dB, 1.5 s before each spoken passage, 3 s release after, held through
   gaps under ~8 s so it never pumps word by word or swells in short pauses), times enforcement
   of music-free rests (if the cue plan leaves music in one on the measured clock, it is muted
   there with 2.5 s fades and reported).

Turn it down with `--music-db -19` (or -22 for sleep); less ducking with `--duck-db -1.5`; a
busy source needs a new candidate or a stiller prompt, not more EQ.

### Prompts

A prompt that mostly says what to avoid ("steady, no melody, no events, behind the voice") comes
back as a drone the listener called "a single steady tone, not really music". Name what should be
there: instruments (soft felt piano, warm pads, slow-attack strings), a key and a slow tempo
("in D major, 60 BPM"), density ("very sparse, few notes, occasional single piano notes"). Keep
the avoid-list (drums, vocals, choir, nature sounds, swells, crescendo, busy melody) in the
negative styles of a composition plan.

### Composed to the session

For music that should follow the cues (enter under the first words, recede, withdraw before a
silent interlude, return, hold through the closing rest, end), compose it after the narration
is final:

1. Plan parts on the measured timeline: one Music API request per part, up to 10 minutes each,
   split where the music is silent anyway (a music-free interlude). Each part is a composition
   plan of sections of up to 2 minutes (`chunks` with `text`, `duration_ms`, `positive_styles`,
   `negative_styles`), named and timed to the cues: "[Opening]" until the first fade-in, "[Under
   the first words]", "[Settling]", "[Receding]" at a partial fade, "[Withdrawal]" at the fade to
   silence, "[Return under the voice]", "[Closing rest]" sections, "[Ending]". Repeat the same
   global styles in every section so the parts sound like one piece.
2. Generate each part: `generate_music.py SESSION --composition-plan music/plan-a.json`.
3. List the parts in `music/timed.json`: each source, the seconds used, a gain to match the parts'
   loudness, anchors (a section start in the source and the timeline boundary it belongs on) and
   the steady sections that may be lengthened or shortened (`stretch_s`). See `fit_music.py
   --help` for the format.
4. `fit_music.py SESSION` places the parts: the first anchor of each part sets its start; between
   two anchors it repeats (or cuts) a stretch of a steady section so the next anchor lands too,
   with one 8 s crossfade at the pair of points whose rhythm, harmony, level and spectrum match
   best, always in a rest. It writes `music/fitted.flac` and selects it; `mix.py` plays it from
   0 s. In testing, joins matched harmony at 0.98 and level within 0.1 dB; listen at each join
   (the report gives the times).

After a retake moves the timeline, run `assemble_voice.py`, then `fit_music.py` and `mix.py`
again. Nothing is regenerated. A plan composed before the narration was final drifts from its
cues by however much the takes change (37 s by the withdrawal in one session).

### Comparing music

`mix.py --music FILE --tag NAME` writes `<slug>.<version>.NAME.<fmt>` and `manifest.NAME.json`
beside the other versions, so the same narration can be heard with generated, composed and
licensed music (Suno, royalty-free libraries). Keep licensed files and their licence notes in
`music/sourced/`. `qa_report.py --tag NAME` checks a tagged mix; run it right after that mix,
because the stems are shared.

## SFX

- **Ambience**: high-pass 30 Hz, edges trimmed of silence and codec padding, then repeated.
  Loops generated with `loop: true` join with a 150 ms equal-power crossfade (they are designed
  to be seamless); recordings and non-loop generations join with 3 s crossfades. Several selected
  takes alternate (A, B, A, B), which hides repetition of a short loop. Calibrated to
  `--ambience-db` (default -18) below the voice. Envelope: from `initial_gain` up to 1.0 over
  `fade_in_ms`, optional cues, down to 0 over `fade_out_ms` ending at the file end, times light
  ducking (`--ambience-duck-db`, default -1.5).
- **One-shots**: high-pass 40 Hz, leading silence trimmed, a 20 ms fade at the end, loudness
  set to `--sfx-db` (default -10) below the voice times the cue's `gain`, placed at the anchor.
- Raise the ambience if the scene disappears under the voice (`--ambience-db -15`); lower bells
  with `--sfx-db -14` or a smaller `gain`.

## Levels and loudness

- The voice stem is normalised to `--lufs` (default -18 LUFS integrated, measured as dual mono
  because mono files play on both speakers). Beds are relative to it, so the voice sits at the
  same level in every version.
- Every version shares one output gain. If any version's true peak would pass `--peak`
  (default -1.5 dBTP), all are lowered together (reported), or with `--limit` a 4x-oversampled
  limiter holds the peaks instead (transparent for occasional plosives; check by ear on
  whispered voices).
- Integrated loudness of the bed versions reads several LU lower than the voice-only file
  because quiet beds fill the long rests and count toward the measurement. That is expected.
  For services that normalise whole tracks, `--normalize integrated` brings each version's
  integrated loudness to `--lufs` instead (the voice level then differs between versions).
- Common targets: -16 LUFS for podcast platforms, -18 to -20 for meditation apps and headphone
  listening, -23 for broadcast-style quiet delivery; -1 to -2 dBTP ceiling for lossy formats.

## Versions and files

| Version | Stems | Channels |
| --- | --- | --- |
| voice | voice | mono |
| voice+music | voice + music | stereo |
| voice+sfx | voice + sfx | stereo |
| voice+music+sfx | voice + music + sfx | stereo |

Files: `output/<slug>.<version>.<fmt>` with `+` written as `-` (`rain.voice-music-sfx.mp3`), or
`<slug>.<version>.<tag>.<fmt>` with `--tag`.
Formats: `wav` (24-bit PCM), `mp3` (192 kbps stereo, 128 kbps mono), `m4a` (AAC 160 kbps),
`flac`. Stems in `stems/` are float32, sample-aligned and at mix level, so any version can be
rebuilt or remixed in a DAW by summing them. `output/manifest.json` records levels, gains,
sources and their hashes, loop joins, resolved cue times and warnings.

## QA

`qa_report.py` checks what numbers can check: files exist and decode, equal durations, loudness,
true peak, clipping, the closing rest meets its minimum, music starts and ends in silence,
music-free rests are silent, the ambience is still present after the last word, each one-shot
is audible at its time, and (with `--transcribe`, ElevenLabs Scribe) that each passage says
the script's words. It cannot hear accent, calm, phrasing, loop seams or balance; its listening
checklist covers those, and the listener's verdict wins over any number.

A useful listening pass: the first two minutes (voice, entrances), one loop join (the manifest
gives the music crossfade time), every music-free interlude, the closing rest and final line,
and the last 30 seconds. Then the whole track at the intended playback level.
