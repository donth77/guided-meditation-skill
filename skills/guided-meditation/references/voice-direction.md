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
| Word-by-word, stop-start | speed setting too low; very short passages; some voices simply read this way (the audition's "pauses inside phrases" lists dips of 0.2 s or more between words with no punctuation) | speed back to 1.0; a voice that sounds alike (`voices.py similar`); a conversion from a guide voice (see "Conversion") |
| Passages faster than the audition | stitching context: some voices read continuous narration faster | `--no-context` (a run switches by itself after one fast reading with context) |
| Hurried passage, last breath clipped | the reading hit the 23.7 s request limit and was squeezed to fit (flagged in `--list`) | split the segment at a sentence boundary; more takes read the same way |
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
  timing and gives it the target's timbre: no breaks, but the first guides tried made it plainer
  than the target's own takes (90-94 percent voiced against 77-80), and to this listener neither
  calm nor British. A calmer, breathier British guide later made it work: see "Conversion".
- In a conversion, breathiness and calm come from the guide reading: similarity, stability,
  style and speaker boost moved it by a point or two, and a whispering guide stays too whispery
  (a full-whisper guide converted to 61-66 percent voiced; breathy ASMR guides converted plain,
  90-93, and kept their fast, suspenseful delivery). The one setting that added breath was the
  English-only model (`--sts-model eleven_english_sts_v2`): 83 percent against 90 for the same
  guide reading, 75-78 when combined with a softly read guide or the target voice's whisper
  edition. Choose the guide for calm, unhurried delivery and let the target and model set the breath.
- Reading each phrase as its own request (four takes each, the cleanest kept) did not help this
  voice: 36 of 36 takes of nine short phrases broke between words, including "let's begin ...
  here" in three words. Request length is not the cause; the breaks belong to the voice. The
  joined phrases also sounded "like multiple phrases were stitched together", so the mode was
  removed. Sending the text lower-cased changed nothing either.
- The same voice on eleven_v3, which has no fine-tune for it, broke less (4 breaks in a 38-word
  passage against about 12 on multilingual v2) but still broke, and read plainer (92-93 percent
  voiced against 79-84); an earlier production also heard its accent drift on v3.
- Shortening the voice's own breaks (an edit, keeping 80 ms or a share of each; since removed)
  worked on one passage with few breaks and failed on the next: with 13-17 breaks in 34 words, full
  shortening sounded "artificially stitched together and unnaturally sped up" (the voice speaks
  its words fast and fills the time with breaks), and keeping 45 percent of each still sounded like
  "pausing after every word". Editing does not rescue a voice whose breaks come every other word.
- Converting a calm reader into the voice gave the right pace and no breaks; converting into the
  same creator's whisper edition changed the character completely ("a completely different
  voice"). Keep the target voice itself.
- When passages sound rushed, look at the pauses between sentences before the speed. Lowering
  the guide's speed below its saved value (1.1 to 1.0 or 0.92) did not slow the words but put
  breaks inside phrases ("that broad sound ... has"); splitting sentences into phrases with
  content-planned rests gave the calm rhythm without touching the delivery.
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

## Conversion

When the listener wants one voice's sound and another voice's delivery, a guide voice reads and
speech to speech re-voices the reading in the chosen voice. The conversion keeps the guide's
timing exactly (the same length and pauses, so the guide's alignment still applies) and takes the
target's timbre and accent. What the September 2026 production learned, with a British ASMR
voice (AImee) whose own readings broke phrases every few words as the target:

- The guide sets pace, pauses, intonation and most of the breath. A guide measuring 83 percent
  voiced converted to 96 ("the pacing is fine, but it should sound as calm and breathy as AImee");
  a calm British guide at 72 percent converted to 85 and was accepted. Expect a conversion about
  10-15 points more voiced than its guide, so pick a guide a little breathier than the target
  level.
- The accepted recipe: guide Rainbird on multilingual v2 (stability 0.75, similarity 0.9, style
  0.15, speed 1.0, read without stitching context, about 94 words per minute), target AImee on
  `eleven_multilingual_sts_v2` (stability 0.5, similarity 0.75, style 0, speaker boost on). A
  conversion costs about 10 credits per second of audio; guide readings cost what text to
  speech costs, so read several and convert one.
- Converting into the target's whisper edition gave "a completely different voice"; v3 readings of
  the target lost the British accent; the English-only STS model, an EQ and blended breath made
  no difference the listener could hear.
- Breaks in the guide reading carry into the conversion. `synthesize.py` screens every guide
  reading and converts only one near the accepted pace with at most one short break.
- Approving passages one at a time before the full run caught pace problems early; a full run
  whose readings came out rushed had wasted most of its credits.

## Pace and request length

- Stitching context changed one voice's pace more than any setting: Rainbird read the same
  passages at 121-159 words per minute with previous/next context and 84-108 without, same
  recipe. AImee, Paula and Clara read at their audition pace with context. The listener heard the
  fast readings as "way too fast"; the audition, generated without context, had been right. So
  `synthesize.py` drops context for the rest of a run once a reading with it comes out faster
  than the accepted pace, and `--no-context` drops it from the start.
- The listener asked for about 100 words per minute on this script; the accepted audition
  measured 94 (`speech_wpm` includes the voice's own pauses between sentences).
- One multilingual v2 request never returned more than 23.684 s: 34 of 174 readings, of 32-42
  words, came back exactly that long. Every word was there, but the reading was squeezed to fit:
  the same guide read a 33-word passage at 84 words per minute and a 42-word one at 108, both
  23.684 s, and the last word ended 0.2 s before the end of the file instead of about 1 s. Keep a
  segment's reading under about 20 s (roughly 30 words at a meditation pace).
  `validate_script.py` warns before synthesis and `synthesize.py --list` flags takes at the limit.

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
