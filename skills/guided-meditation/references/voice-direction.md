# Voice direction

The voice decides whether a meditation works. This is how to choose one, set it, audition it,
and read what goes wrong. Everything here serves the listener's ear; none of it replaces it.

## Choosing a voice

- Start with the account's voices: `voices.py suggest --accent X --gender Y` ranks them by
  meditation-related labels and shows which models each professional voice is fine-tuned for.
  `--library` adds Voice Library results. Library voices work in text to speech directly by id,
  without being added to the account (`voices.py add` is optional and uses a voice slot).
- `--measure` downloads each preview (free) and prints how much of its speech is voiced, which
  screens for whisper strength before any credits are spent (see below).
- A naturally calm, close voice beats settings that try to make a neutral voice calm.
  Whispered or ASMR voices are a specific character: confirm the user wants a whisper rather
  than plain calm.
- A library preview is the owner's showcase recorded with unknown settings and model. Matching
  its voice id, saved settings, or total duration does not reproduce its performance. Use it as
  a reference for character, then judge your own auditions.
- Professional voices are fine-tuned per model. Prefer a model the voice is fine-tuned for when
  accent and likeness matter; `synthesize.py` warns otherwise.

## Whisper strength

"Whispery" covers a wide range, from a breathy but fully voiced tone to a true whisper with no
pitch at all, and listeners are specific about it. It is mostly a property of the voice's
source recordings, not of the settings:

- The screening number is the share of speech frames with pitch (`voiced` in `voices.py
  --measure` and in every audition and passage line). Pin the brief to a level before choosing:

  | Brief says | Level | Generated takes measure | Shortlist previews at |
  | --- | --- | --- | --- |
  | "plain calm", "no whisper" | plain calm | ~80% | 80%+ |
  | "a bit breathy", "whispery but not too much" | a little breath | 65-80% | 70-80% |
  | "soft whisper" | soft whisper | 50-65% | ~55-70% (few data points) |
  | "whispered", "ASMR" | heavy whisper | 30-50% | 45-55% |
  | "full whisper" | full whisper | ~0-10% | whispered previews, by ear |

  Measured in September 2026: voiced voices generated within about 5 points of their preview
  (71 to 66, 76 to 73, 83 to 81, 81 to 80; one ran 74 to 87), while a whispery voice fell from
  50 to 27-41. Treat the preview band as a shortlist filter, not a prediction.
- Settings barely move it. One whispery British voice measured 27-41 percent voiced across
  similarity 0.3-0.52, speaker boost on and off, stability 0.5-0.7, and three different models;
  the listener still heard "whispering too hard".
- Generated takes of whispery voices come out whispier than their previews; previews at 45-55
  percent generate heavy whispers.
- For a mostly plain voice with the occasional whispered line, use `eleven_v3` with a calm,
  voiced narrator and `[whispers]` on the one or two lines that want it.
- Numbers shortlist; the listener chooses. Always audition the shortlist on the script's opening.

## Choosing a model

| Model | Use when | Notes |
| --- | --- | --- |
| `eleven_multilingual_v2` (default) | long-form consistency matters most | most stable on long passages; request stitching; 10,000 chars/request; tags are stripped |
| `eleven_v3` | tags ([whispers], [exhales]) or more expressive delivery are wanted, and the voice holds up on it | no context between passages (request ids and previous/next text are both refused), so passages can vary more; stability is 0.0 Creative, 0.5 Natural, 1.0 Robust; 5,000 chars; can drift in accent on voices without a v3 fine-tune; very short inputs are less stable |
| `eleven_flash_v2_5` / `eleven_turbo_v2_5` | budget matters more than nuance | half the credits per character; less nuanced delivery; stitching works |

A model change needs a fresh audition even with the same voice id.

## Settings

| Setting | Start | Effect and cautions |
| --- | --- | --- |
| stability | 0.5 (v2), 0.5 Natural (v3) | lower = more expressive and more variable. Below ~0.3, long passages tend to rush, wander in pace or accent, and gap unpredictably. Raise it when takes vary too much |
| similarity_boost | voice's saved value or 0.75 | likeness to the source; too high can add artifacts from a noisy source |
| style | 0 | exaggerates the speaker's style; costs stability; leave at 0 for meditation |
| speed | voice's saved value or 1.0 (0.7-1.2) | below ~0.9 many voices separate words into a stop-start delivery. Rests, not speed, make a session spacious. Pace, like breath, lives mostly in the voice: at identical settings (speed 1.0), five meditation voices read the same opening line at 56 to 157 words per minute |
| use_speaker_boost | on | slightly more likeness; turn off if it adds harshness |
| seed | recorded per take | same seed + same settings + same text gives a similar (not identical) take |

`synthesize.py` starts from the voice's own saved settings when voice.json has none.

## Audition protocol

1. Audition the script's first segment. To choose a voice, give the shortlist in one round
   (`--audition --voice-id ID1,ID2,ID3`, one take each, the same text and seed); to refine a
   chosen voice, 2 takes with one recipe. The accepted take is reused as segment 01.
   `--list` shows every audition with its voice, settings and numbers.
2. Listen for: accent held through the take; calm without drag; phrases connected (words
   linked naturally, no gaps inside phrases); the whisper, if wanted, natural rather than
   strained or breathy noise; tags performed, never read aloud.
3. Change one thing per round (voice, model, stability, speed) and audition again; keep every
   take for comparison. The printed words-per-minute helps compare pace between takes (most
   calm narration sits around 110-150 over the voiced span); it is not a verdict.
4. Accept the take the listener prefers. Then set `timing.words_per_minute` to the measured pace
   and revalidate.
5. If the user wants extra certainty, audition one later, longer segment with the accepted
   recipe before the full run (`--audition --segments 07`). Success on the opening does not
   guarantee the same delivery on new text.
6. If the opening keeps failing, keep working on the opening. Do not expand to the full script,
   do not start a broad automatic search over settings, and do not stitch together words.

## Reading failures

| What the listener hears | Usual causes | Try |
| --- | --- | --- |
| Rushed, clipped, "reading aloud" | naturally fast voice; low stability; long dense passage | a calmer voice; stability 0.5-0.7; shorter sentences in the script; speed 0.9-0.95 at most |
| Word-by-word, stop-start | speed setting too low; very short passages; some voices simply read this way (the audition's "pauses inside phrases" lists dips of 0.2 s or more between words with no punctuation) | speed back to 1.0; a voice that sounds alike (`voices.py similar`); a conversion (`--convert`); see "Pauses inside phrases" |
| Whispering too hard | the voice's source recordings are whispered | a different voice: shortlist previews at 65-80 percent voiced with `voices.py ... --measure`; settings and model changes move it only a few points |
| Too plain, wants a little breath | a fully voiced narrator | a voice whose preview measures 60-75 percent voiced; or eleven_v3 with `[whispers]` on chosen lines |
| Accent drifts (e.g. British to American) | model not fine-tuned for the voice; v3 on a voice without a v3 fine-tune; low stability | a fine-tuned model (often multilingual_v2); stability up |
| Strained or noisy whisper | whisper forced from a voice not recorded whispering; style > 0 | a voice whose source is whispered; style 0; speaker boost off |
| Misplaced gaps inside a phrase | nondeterministic generation; odd punctuation | retake the whole passage (2 takes); simplify punctuation |
| Tags read aloud | model is not v3 | tags are stripped for non-v3 models by synthesize.py; check the model |
| Pace or tone jumps between passages | stitching context missing (ids older than 2 h, or v3) | regenerate neighbours in one run; keep the same recipe; for v3 accept some variation |
| A passage sounds unlike the audition | new text, longer passage, nondeterminism | retake that passage; do not change the recipe for everyone because of one passage |

## Pauses inside phrases

A listener hears "let's pause here ... for a minute ... beside ... the stream" when a voice
leaves 0.3-0.5 s of breath or room tone between words that belong together. These are not
silences: the waveform stays 15-40 dB under the speech, and the alignment spreads the pause into
the neighbouring words, so neither a silence threshold nor the alignment gap finds them. The
screening number is the longest dip (at least 20 dB under the take's speech level, or near its
noise floor) between the middles of two words with no punctuation between them; 0.2 s or more
is listed. It is calibrated on one listener: a 0.3 s dip that never fell 25 dB still sounded
like "pause here ... for a minute". Keep listening anyway.

What the September 2026 tests showed, for one ASMR-trained British voice (AImee), seventeen
takes of one opening line:

- Every take broke after "here" and "beside": stability 0.05-0.7, similarity 0.6-0.75, speed
  0.9-1.2, multilingual v2 and turbo v2.5, lower-cased text, full stops instead of a comma,
  three seeds. Settings only moved the dips between 0.2 and 0.8 s, never removed them; the
  listener heard the breaks in the takes whose voice they liked best. The cadence is the voice's
  reading of those words. (The earlier production, below, found v3 and gap editing worse still.)
- "Nothing ... to fix" kept its pause in every take: an emphasis the voice reads into that
  wording. Wording is the listener's call; ask before changing it.
- Voices that sound alike (`voices.py similar` on a take) read the same line without breaks
  (three of three British matches), but the listener heard none of them as calm and British: a
  similar timbre is not a similar character.
- Conversion (`synthesize.py --convert TAKE --voice-id ID`, speech to speech) keeps the guide's
  timing and gives it the target's timbre: no breaks, less breathy than the target's own takes
  (90-94 percent voiced against 77-80), and to this listener neither calm nor British, even from
  British guides (an earlier conversion from a v3 guide failed the same way). Converted takes
  cannot be accepted for a full production.
- When the voice is right and only its reading of certain words breaks, what remains is the
  wording or accepting the cadence. Both are the listener's decision; propose, audition, never
  rewrite silently. With the listener's approval, three rewordings were tried: "Let's take a
  minute, here by the stream" read without breaks on one seed and broke after "take" and "by" on
  another. Rewording moves the breaks; it does not end them. The listener finally preferred a
  take with a 0.3 s dip before "here" ("let's pause for a minute ... here, by the stream"): a
  break before a word that carries emphasis can sound intended; after "here" or "beside" it
  sounded broken. The number cannot tell these apart; the listener can.
- Picking (`--takes 4 --pick`) could not rescue the rest of the script: twelve takes of three
  longer passages, none without breaks (eight or nine in a 21-word passage), and the passages
  came out less breathy than the chosen opening (87-91 percent voiced against 67). The opening
  line had been luck. Picking handles the occasional misfire of a voice that usually reads
  connected; it cannot turn a voice that usually breaks into one that does not.
- So screen voices on a longer passage, not only the opening: before accepting, audition the
  shortlist on the script's longest early segment (`--audition --segments 02 --takes 2`). A
  voice whose takes break there will break across the script.

Phrase boundaries are a separate matter: pauses the listener wants ("for a minute [pause] beside
the stream") are rests in the script, inserted by assembly and heard in the audition previews.

## Lessons from earlier productions

These come from real attempts at a softly whispered, British, comforting narration:

- Short reference sentences that sounded right did not transfer reliably to a full script. The
  script's own opening is the audition that matters, then a later passage.
- Very low stability (0.05) produced erratic, fast passages: one passage ran near 90 words per
  minute and the next near 150, with gaps already inside it.
- Speed 0.75 produced stop-start, word-separated delivery; assembling phrases with rests did not
  fix a strained source performance.
- Switching to v3 for tag-driven phrasing lost the accent and the whispered calm on a voice
  fine-tuned only for v2 models.
- Converting an AI-generated guide with voice changer did not transfer a British accent; voice
  conversion carries the source's accent and cadence.
- Editing gaps between words to repair phrasing always sounded unnatural. Regenerate the passage.
- Automated timing, transcripts and loudness passed on takes the listener rejected. Keep the
  listener's verdict above every score.
