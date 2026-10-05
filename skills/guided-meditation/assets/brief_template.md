# Brief: <title>

Fill every line before writing the script. Mark anything assumed rather than asked as
`(default)` so the final report can say which choices were never confirmed.

```text
TITLE        working title and slug (lowercase-hyphenated; used for file names)
SCENE        where the listener already is; the features that persist
CONDITIONS   fixed or variable time of day / light / weather; what the script must not assume
THEME        the intention of the session (settling, sleep, a work break, body scan...)
LISTENER     context and posture: lying down to sleep, seated at a desk, eyes open or closed,
             headphones or speaker, first-time or experienced
DURATION     target minutes; approximate (default) or exact (only if required); any hard limit
VOICE        character: gender, accent, age range, pace; any reference voice or preview;
             candidate voice ids from voices.py suggest
BREATH       how whispered, in the listener's words, mapped to one level:
             plain calm (voiced ~80%) / a little breath (65-80%) / soft whisper (50-65%) /
             heavy whisper (30-50%) / full whisper (~0%). "Whispery" alone is ambiguous: ask
MODEL        eleven_multilingual_v2 (default) / eleven_v4 (tags, most expressive; new, audition
             first) / eleven_v3 (tags, expressive) / flash (budget)
STYLE        phrasing and register; reference script qualities to borrow (not its content)
ANCHORS      two or three perceptual anchors (a sound of the scene, the breath, body contact)
OUTPUTS      voice (always) + voice+music / voice+sfx / voice+music+sfx
MUSIC        on/off; character (drone, pads, strings, piano?); where it enters, recedes, is
             absent; source: generated or a licensed file
SFX          ambience texture (steady, loopable); one-shots (opening/closing bell?) and where
CLOSING      closing rest length and minimum; the closing line's intent
LEVELS       loudness target (-18 LUFS default; -16 podcast; -20 sleep/headphones), formats
             (wav, mp3, m4a, flac)
BUDGET       credit ceiling for the session; whether to confirm before each paid phase
GATES        which approvals the user wants to give by ear (voice audition always recommended)
```
