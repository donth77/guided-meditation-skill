#!/usr/bin/env python3
# /// script
# requires-python = ">=3.9"
# dependencies = ["numpy"]
# ///
"""
VALIDATE SCRIPT
Check a production script (script.json, schema 5) before any credits are spent, and compute its
timing so the arithmetic is never done by hand.

  python3 scripts/validate_script.py SESSION             # errors, warnings, estimate, credits
  python3 scripts/validate_script.py SESSION --write     # also write the computed timing back
  python3 scripts/validate_script.py SESSION --json      # machine-readable report

Exit 0 when there are no errors, 1 otherwise. Warnings are advice to act on or to justify.
Checks structure, phrase and segment pause rules, audio-tag limits for the chosen model, the
closing rest, every music/ambience/SFX cue anchor, chronology and fade overlap on the estimated
clock, music-free interludes, the final music state, filler and permission phrases, repeated
openings, uniform segment lengths, segments too long for one multilingual v2 request, and estimated
duration and credits.
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import sys

from gm_common import (
    ANCHOR_BOUNDARIES, MUSIC_CREDITS_PER_MINUTE, PERMITTED_TAGS, SESSION_BOUNDARIES, SFX_CREDITS_PER_SECOND,
    STS_CREDITS_PER_MINUTE, V2_REQUEST_MAX_S, count_words, cue_envelope, estimated_timeline, find_tags, fmt_time,
    load_script, performs_tags, read_json, resolve_anchor, segment_request, session_pads, session_root, strip_tags,
    timeline_segment, tts_rate, words_per_minute, write_json,
)

FILLER = ["simply", "just allow yourself to", "gently", "as you", "notice how you begin to"]
PERMISSIONS = ["you can", "there is no need", "there's no need", "nothing needs to", "no need to"]
IMAGINE = ["imagine", "picture yourself", "visualize", "visualise", "you arrive", "you find yourself"]
CLAIMS = ["you will feel", "you'll feel", "heal", "cure", "anxiety will", "stress will", "melts away",
          "you will be", "you'll be completely", "guaranteed"]
TIME_OF_DAY = ["morning", "sunlight", "sunrise", "sunset", "dawn", "dusk", "night", "tonight", "stars",
               "moonlight", "moon", "evening", "noon"]
LOOP_EVENTS = ["thunder", "bird call", "birdsong", "bird song", "dog", "bell", "voice", "voices", "footstep",
               "car", "horn", "splash", "knock", "chime", "owl", "laugh"]
MUSIC_MODELS = ("music_v1", "music_v2", "music_v2_5")


class Report:
    def __init__(self):
        self.errors, self.warnings, self.info = [], [], []

    def e(self, msg):
        self.errors.append(msg)

    def w(self, msg):
        self.warnings.append(msg)

    def i(self, msg):
        self.info.append(msg)


def _num(v, default=None):
    return v if isinstance(v, (int, float)) and not isinstance(v, bool) else default


def _has(text, phrase):
    return re.search(r"(?<![\w'])" + re.escape(phrase) + r"(?![\w'])", text, re.IGNORECASE) is not None


def sentences(text):
    t = strip_tags(text)
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", t) if s.strip()]


def check_anchor(anchor, where, seg_ids, r):
    if not isinstance(anchor, dict):
        r.e(f"{where}: anchor must be an object")
        return False
    b = anchor.get("boundary")
    if b in SESSION_BOUNDARIES:
        pass
    elif b in ANCHOR_BOUNDARIES:
        if str(anchor.get("segment_id")) not in seg_ids:
            r.e(f"{where}: anchor segment_id '{anchor.get('segment_id')}' does not exist")
            return False
    else:
        r.e(f"{where}: boundary must be one of {', '.join(ANCHOR_BOUNDARIES + SESSION_BOUNDARIES)}")
        return False
    if _num(anchor.get("offset_ms", 0)) is None:
        r.e(f"{where}: offset_ms must be a number")
        return False
    return True


def check_cues(cues, label, seg_ids, timeline, r, initial):
    """Validate a cue list; returns the Envelope on the estimated clock (or None)."""
    ok = True
    if not isinstance(cues, list):
        r.e(f"{label}.cues must be a list")
        return None
    ids = set()
    for k, c in enumerate(cues):
        where = f"{label} cue {c.get('id', k + 1) if isinstance(c, dict) else k + 1}"
        if not isinstance(c, dict):
            r.e(f"{where}: must be an object")
            ok = False
            continue
        cid = c.get("id")
        if not cid:
            r.w(f"{where}: give the cue an id (M01, A01...)")
        elif cid in ids:
            r.e(f"{where}: duplicate cue id")
        ids.add(cid)
        g = _num(c.get("target_gain"))
        if g is None or not 0 <= g <= 1:
            r.e(f"{where}: target_gain must be between 0 and 1")
            ok = False
        if _num(c.get("fade_ms"), -1) < 0:
            r.e(f"{where}: fade_ms must be a number >= 0")
            ok = False
        ok &= check_anchor(c.get("anchor"), where, seg_ids, r)
    if not ok:
        return None
    times = [resolve_anchor(c["anchor"], timeline) for c in cues]
    if times != sorted(times):
        order = ", ".join("{}@{}".format(c.get("id"), fmt_time(t / 1000)) for c, t in zip(cues, times))
        r.e(f"{label}: cues are not in chronological order on the estimated clock ({order})")
    end = timeline["duration_ms"]
    for k, (c, t0) in enumerate(zip(cues, times)):
        t1 = t0 + float(c.get("fade_ms", 0))
        if t0 < 0:
            r.e(f"{label} cue {c.get('id')}: starts before the session ({fmt_time(t0 / 1000)})")
        if t1 > end + 1:
            r.e(f"{label} cue {c.get('id')}: fade ends at {fmt_time(t1 / 1000)}, after the session end "
                f"({fmt_time(end / 1000)})")
        if k + 1 < len(cues) and t1 > times[k + 1] + 1:
            r.e(f"{label} cue {c.get('id')}: fade runs until {fmt_time(t1 / 1000)}, past the next cue "
                f"{cues[k + 1].get('id')} at {fmt_time(times[k + 1] / 1000)} (estimated clock)")
    return cue_envelope(initial, cues, timeline)


def check(script, model, r):
    seg_ids = []
    version = script.get("schema_version")
    if version not in (4, 5):
        r.e("schema_version must be 5 (4 is read without the sfx layer)")
    elif version == 4:
        r.i("schema_version 4: no generated sfx layer; upgrade to 5 for ambience and one-shots")
    for k in ("scene", "theme", "voice"):
        if not isinstance(script.get(k), str) or not script.get(k).strip():
            r.w(f"'{k}' should describe the session ({k} is missing)")
    target = _num(script.get("target_seconds"))
    if target is None or target <= 0:
        r.w("target_seconds is missing; duration can only be reported, not compared")
    mode = script.get("duration_mode", "approximate")
    if mode not in ("approximate", "exact"):
        r.e("duration_mode must be 'approximate' or 'exact'")
    lead, tail = session_pads(script)
    if lead < 0 or tail < 0:
        r.e("session.lead_in_ms and session.tail_ms must be >= 0")

    segments = script.get("segments")
    if not isinstance(segments, list) or not segments:
        r.e("segments must be a non-empty list")
        return None
    total_tags = 0
    all_text = []
    fatal = False   # structure the timeline cannot be computed from; other errors keep checking
    for si, seg in enumerate(segments):
        sid = str(seg.get("id", "")) if isinstance(seg, dict) else ""
        where = f"[{sid or si + 1}]"
        if not isinstance(seg, dict) or not sid:
            r.e(f"{where} every segment needs an id")
            fatal = True
            continue
        if sid in seg_ids:
            r.e(f"{where} duplicate segment id")
        seg_ids.append(sid)
        phrases = seg.get("phrases")
        if not isinstance(phrases, list) or not phrases:
            r.e(f"{where} needs at least one phrase")
            fatal = True
            continue
        seg_tags = 0
        for pi, ph in enumerate(phrases):
            pw = f"{where} phrase {pi + 1}"
            text = ph.get("text") if isinstance(ph, dict) else None
            if not isinstance(text, str) or not text.strip():
                r.e(f"{pw}: text is empty")
                fatal = True
                continue
            all_text.append((pw, text))
            if count_words(text) == 0:
                r.e(f"{pw}: no spoken words (a phrase cannot be only an audio tag)")
            pause = _num(ph.get("pause_after_ms"))
            last = pi == len(phrases) - 1
            if pause is None or pause < 0:
                r.e(f"{pw}: pause_after_ms must be a number >= 0")
                fatal |= pause is None
            elif last and pause != 0:
                r.e(f"{pw}: the last phrase has pause_after_ms 0 (the segment owns the following rest)")
            elif not last and pause == 0:
                r.e(f"{pw}: a phrase boundary needs a purposeful positive pause; otherwise merge the phrases")
            elif not last and pause < 300:
                r.w(f"{pw}: {pause} ms is shorter than a natural sentence gap; is this boundary needed?")
            elif not last and pause > 6000:
                r.w(f"{pw}: {pause} ms inside a segment; a rest this long usually marks a new segment")
            if re.search(r"<\s*break", text, re.IGNORECASE):
                r.e(f"{pw}: SSML break tags are not allowed; pauses are data (pause_after_ms)")
            if "..." in text or "…" in text:
                r.w(f"{pw}: ellipses used as pacing; write plain punctuation and put rests in pause_after_ms")
            if re.search(r"\((softly|quietly|whisper|slowly|pause|breathe)[^)]*\)", text, re.IGNORECASE):
                r.e(f"{pw}: parenthetical stage direction would be spoken aloud")
            for tag in find_tags(text):
                total_tags += 1
                seg_tags += 1
                if tag not in PERMITTED_TAGS:
                    r.e(f"{pw}: audio tag [{tag}] is not permitted (allowed: {', '.join(PERMITTED_TAGS)}; "
                        f"pauses and sound effects are never tags)")
        if seg_tags > 1:
            r.e(f"{where} has {seg_tags} audio tags; maximum one per segment")
        sp = _num(seg.get("pause_after_ms"))
        if sp is None or sp < 0:
            r.e(f"{where} pause_after_ms must be a number >= 0")
            fatal |= sp is None
        elif si == len(segments) - 1 and sp != 0:
            r.e(f"{where} the final segment has pause_after_ms 0 (use session.tail_ms for the ending)")
    if total_tags > 4:
        r.e(f"{total_tags} audio tags in the script; maximum four")
    if total_tags and not performs_tags(model):
        r.i(f"audio tags will be removed before synthesis with {model} (only the v3 and v4 models perform them)")
    if fatal:
        return None

    # Language checks (warnings only).
    joined = " ".join(t for _, t in all_text)
    for pw, text in all_text:
        for f in FILLER:
            if _has(text, f):
                r.w(f"{pw}: \"{f}\" reads as generic meditation filler")
        for f in IMAGINE:
            if _has(text, f):
                r.w(f"{pw}: \"{f}\": the listener is already in the scene; don't ask them to imagine or arrive")
        for f in CLAIMS:
            if _has(text, f):
                r.w(f"{pw}: \"{f}\": no therapeutic or outcome claims, no promises about feelings")
        if _has(text, "close your eyes") and not _has(text, "if you like") and not _has(text, "if you'd like"):
            r.w(f"{pw}: eye closure as an instruction; offer it as optional with a listening alternative")
    for p in PERMISSIONS:
        n = len(re.findall(r"(?<![\w'])" + re.escape(p) + r"(?![\w'])", joined, re.IGNORECASE))
        if n >= 3:
            r.w(f"\"{p}\" appears {n} times; repeated permissions sound mechanical")
    cond = (script.get("scene_conditions") or "").lower()
    if any(k in cond for k in ("any time", "vari", "chang", "cycle")):
        for pw, text in all_text:
            for word in TIME_OF_DAY:
                if _has(text, word):
                    r.w(f"{pw}: \"{word}\" assumes a time of day the scene does not fix")
    sents = [s for _, t in all_text for s in sentences(t)]
    firsts = [re.findall(r"[\w']+", s.lower())[:2] for s in sents]
    firsts = [f for f in firsts if f]
    if len(firsts) >= 8:
        counts = {}
        for f in firsts:
            counts[f[0]] = counts.get(f[0], 0) + 1
        word, n = max(counts.items(), key=lambda kv: kv[1])
        if n >= 4 and n / len(firsts) > 0.3:
            r.w(f"{n} of {len(firsts)} sentences open with \"{word}\"; vary the openings")
        pairs = {}
        for f in firsts:
            if len(f) == 2:
                pairs[" ".join(f)] = pairs.get(" ".join(f), 0) + 1
        for pair, n in pairs.items():
            if n >= 3 and pair not in ("it s", "there s"):
                r.w(f"{n} sentences open with \"{pair}\"")

    seg_words = [sum(count_words(p["text"]) for p in s["phrases"]) for s in segments]
    n = len(segments)
    lo, hi = (8, 14) if (target or 0) >= 480 else (4, 10)
    if not lo <= n <= hi:
        r.w(f"{n} segments; {lo}-{hi} usually suits a session of this length")
    body = [w for w in seg_words if w > 5]
    if len(body) >= 4:
        cv = statistics.pstdev(body) / max(statistics.mean(body), 1)
        if cv < 0.25:
            r.w(f"segment lengths are similar ({min(body)}-{max(body)} words); vary them deliberately")

    timeline = estimated_timeline(script)
    seg_pauses = [float(s.get("pause_after_ms", 0)) for s in segments[:-1]]
    if seg_pauses and seg_pauses[0] > 15000:
        r.w(f"the first rest is {seg_pauses[0] / 1000:.0f}s; early rests are usually 5-10s")

    # Closing rest.
    cr = script.get("closing_rest")
    if cr:
        cid = str(cr.get("segment_id"))
        seg = next((s for s in segments if str(s["id"]) == cid), None)
        if seg is None:
            r.e(f"closing_rest.segment_id '{cid}' does not exist")
        else:
            dur = _num(cr.get("duration_ms"), -1)
            mn = _num(cr.get("minimum_duration_ms"), 0)
            if dur != seg.get("pause_after_ms"):
                r.e(f"closing_rest.duration_ms ({dur}) must equal segment {cid}'s pause_after_ms "
                    f"({seg.get('pause_after_ms')})")
            if dur < mn:
                r.e(f"closing rest {dur} ms is below its minimum {mn} ms")
            if str(segments[-1]["id"]) == cid:
                r.e("the closing rest belongs to the segment before the final spoken line")
        fs = cr.get("final_spoken_segment_id")
        if fs is not None and str(fs) != str(segments[-1]["id"]):
            r.w(f"closing_rest.final_spoken_segment_id '{fs}' is not the last segment")

    seg_set = set(seg_ids)
    # Music.
    music = script.get("music") or {}
    if music.get("enabled"):
        gen = music.get("generation") or {}
        prompt = music.get("prompt") or ""
        if not prompt.strip() and not music.get("source_file"):
            r.e("music.prompt is empty (or set music.source_file to a licensed recording)")
        if re.search(r"[\"“”]", prompt) or re.search(r"\bby [A-Z][a-z]+", prompt):
            r.w("music.prompt seems to name a work or artist; the Music API rejects those (bad_prompt)")
        if _num(music.get("initial_gain"), 0) != 0:
            r.e("music.initial_gain must be 0 (music enters with a cue)")
        if gen.get("model_id") and gen["model_id"] not in MUSIC_MODELS:
            r.w(f"music.generation.model_id '{gen['model_id']}' is not one of {', '.join(MUSIC_MODELS)}")
        length = _num(gen.get("length_ms"), 180000)
        if not 3000 <= length <= 600000:
            r.e("music.generation.length_ms must be between 3000 and 600000")
        if gen.get("force_instrumental") is False:
            r.w("music.generation.force_instrumental is false; meditation beds must have no vocals")
        xf = _num((music.get("looping") or {}).get("crossfade_ms"), 12000)
        if not 1000 <= xf <= 30000:
            r.w(f"music.looping.crossfade_ms {xf} is outside 1000-30000")
        env = check_cues(music.get("cues") or [], "music", seg_set, timeline, r, 0.0)
        if not music.get("cues"):
            r.e("music is enabled but has no cues (it would stay silent at gain 0)")
        loop_after = bool(music.get("loop_after_session"))
        if env is not None:
            end = timeline["duration_ms"]
            if loop_after:
                # The music hands over to a loop that plays on after the session: it must still be playing.
                if env.value_at(end) < 0.05:
                    r.e("music.loop_after_session: the music must still be playing at the end, where the loop "
                        "takes over (add a cue that brings it back after the final line)")
                elif env.value_at(end - 1500) != env.value_at(end):
                    r.w("music.loop_after_session: a cue is still moving in the last 1.5 s; let it settle before the "
                        "end so the loop continues at a steady level")
                else:
                    r.i(f"music continues into the loop after the session at gain {env.value_at(end):.2f}")
            elif env.value_at(end) > 1e-6:
                r.e(f"music does not finish at gain 0 (ends at {env.value_at(end):.2f}); add a closing fade")
            for sid in music.get("music_free_pauses") or []:
                seg = timeline_segment(timeline, sid)
                if seg is None:
                    r.e(f"music_free_pauses: segment '{sid}' does not exist")
                    continue
                peak = env.max_in(seg["speech_end_ms"], seg["pause_end_ms"])
                if peak > 1e-6:
                    r.e(f"music_free_pauses: the pause after {sid} has music (gain up to {peak:.2f}); fade out "
                        f"before it starts and return no earlier than its end")
            final = timeline["segments"][-1]
            if cr and env.value_at(final["segment_start_ms"]) > (0.5 if loop_after else 1e-6):
                r.w("music is still playing when the final spoken line starts; fade it during the closing rest"
                    + (" (with a loop after the session, a dip to about 0.3 is enough)" if loop_after else ""))
    else:
        if music.get("cues") or music.get("music_free_pauses") or _num(music.get("initial_gain"), 0):
            r.e("music is disabled: set cues and music_free_pauses to [] and initial_gain to 0")

    # SFX (schema 5).
    sfx = script.get("sfx") or {}
    if sfx.get("enabled"):
        amb = sfx.get("ambience") or {}
        if amb.get("enabled"):
            gen = amb.get("generation") or {}
            prompt = amb.get("prompt") or ""
            if not prompt.strip() and not amb.get("source_file"):
                r.e("sfx.ambience.prompt is empty (or set sfx.ambience.source_file)")
            d = _num(gen.get("duration_seconds"), 30)
            if not 0.5 <= d <= 30:
                r.e("sfx.ambience.generation.duration_seconds must be 0.5-30")
            pi_ = _num(gen.get("prompt_influence"), 0.3)
            if not 0 <= pi_ <= 1:
                r.e("sfx.ambience.generation.prompt_influence must be 0-1")
            for ev in LOOP_EVENTS:
                if _has(prompt, ev):
                    r.w(f"sfx.ambience.prompt mentions \"{ev}\": a {d:.0f}s loop repeats distinctive events "
                        f"audibly; prefer steady textures and put single events in one_shots")
                    break
            if re.search(r"\b(no|without|never)\b", prompt, re.IGNORECASE):
                r.w("sfx.ambience.prompt uses negation; describe what should be heard instead")
            g0 = _num(amb.get("initial_gain"), 0)
            if not 0 <= g0 <= 1:
                r.e("sfx.ambience.initial_gain must be 0-1")
            for k in ("fade_in_ms", "fade_out_ms"):
                if _num(amb.get(k), 0) < 0:
                    r.e(f"sfx.ambience.{k} must be >= 0")
            if _num(amb.get("fade_out_ms"), 0) > tail:
                r.w("sfx.ambience.fade_out_ms is longer than session.tail_ms; the ambience starts fading "
                    "before the last words end")
            if amb.get("cues"):
                check_cues(amb["cues"], "sfx.ambience", seg_set, timeline, r, g0)
        shots = sfx.get("one_shots") or []
        ids = {}
        for k, s in enumerate(shots):
            where = f"sfx one-shot {s.get('id', k + 1)}"
            sid = s.get("id")
            if not sid:
                r.e(f"{where}: needs an id")
                continue
            if sid in ids:
                r.e(f"{where}: duplicate id")
            ids[sid] = s
            if not (s.get("prompt") or s.get("same_as") or s.get("source_file")):
                r.e(f"{where}: needs a prompt, same_as or source_file")
            d = _num(s.get("duration_seconds"), 5)
            if not 0.5 <= d <= 30:
                r.e(f"{where}: duration_seconds must be 0.5-30")
            if _num(s.get("gain"), 1) < 0:
                r.e(f"{where}: gain must be >= 0")
            if check_anchor(s.get("anchor"), where, seg_set, r):
                t = resolve_anchor(s["anchor"], timeline)
                if t < 0 or t + d * 1000 > timeline["duration_ms"]:
                    r.w(f"{where}: at {fmt_time(t / 1000)} it runs past the estimated session edges; the track "
                        f"is extended to fit")
        for sid, s in ids.items():
            ref = s.get("same_as")
            if ref and (ref not in ids or ids[ref].get("same_as")):
                r.e(f"sfx one-shot {sid}: same_as '{ref}' must name a one-shot that has its own prompt")
        if not amb.get("enabled") and not shots:
            r.w("sfx is enabled but has neither ambience nor one-shots")
    elif sfx.get("one_shots") or (sfx.get("ambience") or {}).get("enabled"):
        r.i("sfx.enabled is false; its ambience and one-shots are ignored")
    return timeline


def request_length(script, model, r):
    """Segments whose reading may pass what one multilingual v2 request returns (it squeezes the
    reading to fit rather than running longer)."""
    if model != "eleven_multilingual_v2":
        return
    wpm = words_per_minute(script)
    for seg in script.get("segments") or []:
        words = sum(count_words(p.get("text") or "") for p in seg.get("phrases") or [])
        est = words / wpm * 60.0
        if est > 0.8 * V2_REQUEST_MAX_S:   # slow voices and many short sentences read longer than the estimate
            r.w(f"[{seg['id']}] {words} words read in about {est:.0f}s at {wpm:g} words/min; one multilingual v2 "
                f"request returns at most {V2_REQUEST_MAX_S:.1f}s, squeezing a longer reading to fit. Consider "
                "splitting the segment at a sentence boundary")


def credits(script, model, models=None, conversion=False):
    keep = performs_tags(model)
    chars = sum(len(segment_request(s, keep)[0]) for s in script["segments"])
    out = {"model": model, "narration_chars": chars, "narration": chars * tts_rate(model, models)}
    if conversion:      # each passage is read by the guide, then converted: billed per minute of audio
        words = sum(count_words(p["text"]) for s in script["segments"] for p in s["phrases"])
        out["narration"] += words / words_per_minute(script) * STS_CREDITS_PER_MINUTE
    music = script.get("music") or {}
    out["music"] = 0.0
    if music.get("enabled") and not music.get("source_file"):
        length = (music.get("generation") or {}).get("length_ms", 180000)
        out["music"] = length / 60000.0 * MUSIC_CREDITS_PER_MINUTE
        out["music_seconds"] = length / 1000.0
    sfx = script.get("sfx") or {}
    out["ambience"] = out["one_shots"] = 0.0
    if sfx.get("enabled"):
        amb = sfx.get("ambience") or {}
        if amb.get("enabled") and not amb.get("source_file"):
            out["ambience"] = (amb.get("generation") or {}).get("duration_seconds", 30) * SFX_CREDITS_PER_SECOND
        for s in sfx.get("one_shots") or []:
            if s.get("prompt") and not s.get("same_as") and not s.get("source_file"):
                out["one_shots"] += s.get("duration_seconds", 5) * SFX_CREDITS_PER_SECOND
    out["total"] = out["narration"] + out["music"] + out["ambience"] + out["one_shots"]
    return out


def timing_block(script, timeline):
    wpm = words_per_minute(script)
    spoken = sum(count_words(p["text"]) for s in script["segments"] for p in s["phrases"])
    phrase_pause = sum(float(p.get("pause_after_ms", 0)) for s in script["segments"] for p in s["phrases"]) / 1000
    seg_pause = sum(float(s.get("pause_after_ms", 0)) for s in script["segments"]) / 1000
    lead, tail = session_pads(script)
    speech = spoken / wpm * 60.0
    return {
        "words_per_minute": wpm,
        "spoken_words": spoken,
        "estimated_speech_seconds": round(speech, 2),
        "phrase_pause_seconds": round(phrase_pause, 2),
        "between_segments_seconds": round(seg_pause, 2),
        "narration_free_seconds": round(phrase_pause + seg_pause, 2),
        "session_padding_seconds": round((lead + tail) / 1000, 2),
        "cue_clock": "segment_boundaries_after_phrase_assembly",
        "estimate_basis": (f"Planning estimate at {wpm:g} words/minute, excluding deliberate rests. Actual delivery "
                           "sets the length: the measured timeline after assembly is authoritative, and speech is "
                           "never sped up, slowed or trimmed to reach the target."),
        "estimated_seconds": round(timeline["duration_ms"] / 1000, 2),
    }


POLICY = {
    "unit": "milliseconds",
    "semantics": "minimum pause after a complete phrase, before the next phrase in the same segment",
    "assembly": ("Each segment is synthesized as one connected passage. At authored phrase boundaries the "
                 "existing natural gap counts once and only the missing rest is added; breaths are kept. Nothing "
                 "inside a phrase is cut, moved or re-timed. The last phrase has pause_after_ms 0; the segment "
                 "owns the following rest."),
    "placement": ("Divide a segment only at an intentional, meaningful pause. A phrase may contain several "
                  "sentences. A broken or stretched phrase is regenerated as a complete passage, never repaired "
                  "with word-gap edits."),
}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("session", help="session folder (or its script.json)")
    ap.add_argument("--write", action="store_true", help="write timing, estimated_seconds and pause policy back")
    ap.add_argument("--json", action="store_true", help="print the report as JSON")
    ap.add_argument("--model", help="TTS model for tag and credit checks (default: voice.json, else eleven_multilingual_v2)")
    args = ap.parse_args()

    root = session_root(args.session)
    script = load_script(root)
    voice = read_json(root / "voice.json", {}) or {}
    conversion = voice.get("method") == "speech_to_speech"
    if conversion:      # a conversion recipe: its guide voice reads the text
        voice = {**voice, "model_id": (voice.get("guide") or {}).get("model_id")}
    model = args.model or voice.get("model_id") or "eleven_multilingual_v2"
    r = Report()
    timeline = check(script, model, r)
    if timeline is not None:
        request_length(script, model, r)
    result = {"errors": r.errors, "warnings": r.warnings, "info": r.info}
    if timeline is not None:
        tb = timing_block(script, timeline)
        cr = credits(script, model, conversion=conversion)
        result.update({"timing": tb, "credits": cr, "estimated_timeline": timeline})
        target = script.get("target_seconds")
        est = tb["estimated_seconds"]
        if target and script.get("duration_mode") == "exact" and abs(est - target) > max(5, 0.03 * target):
            r.w(f"exact duration requested: estimate {fmt_time(est)} vs target {fmt_time(target)}; rebalance rests "
                "(never speech speed), then confirm against the measured timeline")
        if args.write and not r.errors:
            script["timing"] = {k: v for k, v in tb.items() if k != "estimated_seconds"}
            script["estimated_seconds"] = est
            script.setdefault("phrase_pause_policy", POLICY)
            if script.get("duration_mode") is None:
                script["duration_mode"] = "approximate"
            write_json(root / "script.json", script)
    if args.json:
        print(json.dumps(result, indent=2))
        sys.exit(1 if r.errors else 0)

    title = script.get("title") or script.get("scene") or root.name
    segs = script.get("segments") or []
    nphr = sum(len(s.get("phrases") or []) for s in segs if isinstance(s, dict))
    print(f"{title}  (script.json, schema {script.get('schema_version')}): {len(segs)} segments, {nphr} phrases")
    if timeline is not None:
        tb, cr = result["timing"], result["credits"]
        print(f"Model for tag/credit checks: {model}" + (" (guide reading; narration credits include the conversion)"
                                                           if conversion else ""))
        print(f"Estimate at {tb['words_per_minute']:g} wpm: {tb['spoken_words']} words, speech "
              f"{fmt_time(tb['estimated_speech_seconds'])} + phrase rests {fmt_time(tb['phrase_pause_seconds'])} + "
              f"segment rests {fmt_time(tb['between_segments_seconds'])} + lead-in/tail "
              f"{fmt_time(tb['session_padding_seconds'])} = {fmt_time(tb['estimated_seconds'])}")
        target = script.get("target_seconds")
        if target:
            diff = tb["estimated_seconds"] - target
            mode = script.get("duration_mode", "approximate")
            print(f"Target ~{fmt_time(target)} ({mode}); estimate is {fmt_time(abs(diff))} "
                  f"{'longer' if diff >= 0 else 'shorter'}"
                  + (" -- approximate mode reports the difference; it does not pad or cut" if mode == "approximate" else ""))
        print(f"Credits (estimate): narration {cr['narration_chars']:,} chars = {cr['narration']:,.0f}"
              + (f" | music {cr.get('music_seconds', 0):.0f}s ~ {cr['music']:,.0f}" if cr["music"] else "")
              + (f" | ambience {cr['ambience']:,.0f}" if cr["ambience"] else "")
              + (f" | one-shots {cr['one_shots']:,.0f}" if cr["one_shots"] else "")
              + f" | total ~ {cr['total']:,.0f} (each retake or extra candidate adds its share)")
        print()
        print(f"{'seg':<5}{'words':>6}{'start':>10}{'speech':>9}{'phrase rests':>14}{'rest after':>12}")
        for seg, ts in zip(segs, timeline["segments"]):
            w = sum(count_words(p["text"]) for p in seg["phrases"])
            pr = sum(float(p.get("pause_after_ms", 0)) for p in seg["phrases"]) / 1000
            print(f"{ts['id']:<5}{w:>6}{fmt_time(ts['segment_start_ms'] / 1000):>10}"
                  f"{(ts['speech_end_ms'] - ts['segment_start_ms']) / 1000 - pr:>8.1f}s{pr:>13.1f}s"
                  f"{float(seg.get('pause_after_ms', 0)) / 1000:>11.1f}s")
    print()
    print(f"Errors ({len(r.errors)})")
    for m in r.errors:
        print(f"  - {m}")
    print(f"Warnings ({len(r.warnings)})")
    for m in r.warnings:
        print(f"  - {m}")
    if r.info:
        print("Info")
        for m in r.info:
            print(f"  - {m}")
    if args.write:
        print("\nwrote timing to script.json" if not r.errors else "\nnot written: fix the errors first")
    sys.exit(1 if r.errors else 0)


if __name__ == "__main__":
    main()
