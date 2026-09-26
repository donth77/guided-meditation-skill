#!/usr/bin/env python3
# /// script
# requires-python = ">=3.9"
# dependencies = ["numpy"]
# ///
"""
RENDER SCRIPT
Write the review documents for a session from script.json:

  script.md   narration in phrase blocks, with pauses, rests and cue notes on their own lines
  script.txt  the spoken words only, one paragraph per segment
  timing.md   estimated timing, closing rest, music cues and SFX times; measured timing and drift
              as well once voice/timeline.json exists

  python3 scripts/render_script.py SESSION

Nothing in these files is sent to speech synthesis; they are for the listener's review.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from gm_common import (
    count_words, estimated_timeline, fmt_time, load_script, read_json, resolve_anchor, session_root, strip_tags,
    timeline_segment,
)


def secs_words(ms):
    s = ms / 1000.0
    if s < 60:
        return f"{s:g} seconds" if s != 1 else "1 second"
    m, rem = divmod(round(s), 60)
    out = f"{m} minute{'s' if m != 1 else ''}"
    if rem:
        out += f" {rem} second{'s' if rem != 1 else ''}"
    return out


def cue_notes(script):
    """Map (segment_id, boundary) or session boundary -> list of note strings."""
    notes = {}

    def add(anchor, text):
        b = anchor.get("boundary")
        key = (b,) if b in ("session_start", "session_end") else (str(anchor.get("segment_id")), b)
        off = float(anchor.get("offset_ms", 0) or 0)
        if off:
            text += f" ({'+' if off > 0 else '-'}{secs_words(abs(off))} from this point)"
        notes.setdefault(key, []).append(text)

    music = script.get("music") or {}
    if music.get("enabled"):
        for c in music.get("cues") or []:
            add(c["anchor"], f"Music {c.get('id', '')}: to gain {c.get('target_gain')} over "
                             f"{secs_words(c.get('fade_ms', 0))}. {c.get('direction', '')}".strip())
    sfx = script.get("sfx") or {}
    if sfx.get("enabled"):
        amb = sfx.get("ambience") or {}
        for c in (amb.get("cues") or []) if amb.get("enabled") else []:
            add(c["anchor"], f"Ambience {c.get('id', '')}: to gain {c.get('target_gain')} over "
                             f"{secs_words(c.get('fade_ms', 0))}. {c.get('direction', '')}".strip())
        for s in sfx.get("one_shots") or []:
            what = s.get("direction") or s.get("prompt") or f"same sound as {s.get('same_as')}"
            add(s["anchor"], f"SFX {s.get('id')}: {what}")
    return notes


def render_md(script, voice, timeline):
    title = script.get("title") or script.get("scene") or "Meditation"
    notes = cue_notes(script)
    free = {str(x) for x in (script.get("music") or {}).get("music_free_pauses") or []} \
        if (script.get("music") or {}).get("enabled") else set()
    cr = script.get("closing_rest") or {}
    out = [f"# {title}: narration in phrases", "",
           "**For review and audio assembly. Pause lines and notes are never sent to speech synthesis.**", ""]
    out.append(f"Voice: {script.get('voice', '(not described)')}")
    if voice:
        s = voice.get("voice_settings") or {}
        out.append(f"Recipe: {voice.get('voice_name') or voice.get('voice_id')} ({voice.get('voice_id')}), "
                   f"{voice.get('model_id')}, stability {s.get('stability')}, similarity {s.get('similarity_boost')}, "
                   f"style {s.get('style')}, speed {s.get('speed')}")
    out += ["", "Read each block as connected speech; a block can hold several sentences. Deliberate pauses sit on "
                "their own lines between blocks. Nothing is timed between words. A broken block is regenerated "
                "as a whole passage, never repaired by editing word gaps.", ""]
    est = timeline["duration_ms"] / 1000
    target = script.get("target_seconds")
    line = f"Estimated length about {fmt_time(est)}"
    if target:
        line += f" (target ~{fmt_time(target)}, {script.get('duration_mode', 'approximate')})"
    out += [line + ". Actual delivery sets the final length.", ""]
    for n in notes.get(("session_start",), []):
        out += [f"> {n}", ""]
    segs = script["segments"]
    for si, seg in enumerate(segs):
        sid = str(seg["id"])
        out += [f"## Segment {sid}", ""]
        for n in notes.get((sid, "segment_start"), []):
            out += [f"> {n}", ""]
        for pi, ph in enumerate(seg["phrases"]):
            out += [ph["text"].strip(), ""]
            if pi < len(seg["phrases"]) - 1:
                out += [f"**Pause {secs_words(ph.get('pause_after_ms', 0))}.**", ""]
        for n in notes.get((sid, "speech_end"), []):
            out += [f"> {n}", ""]
        rest = float(seg.get("pause_after_ms", 0) or 0)
        if si < len(segs) - 1:
            extra = []
            if cr and str(cr.get("segment_id")) == sid:
                mn = cr.get("minimum_duration_ms")
                extra.append("closing rest" + (f"; keep at least {secs_words(mn)}" if mn else ""))
            if sid in free:
                extra.append("ambience only; no music")
            out += [f"**Rest {secs_words(rest)}{' (' + '; '.join(extra) + ')' if extra else ''}.**", ""]
            for n in notes.get((sid, "pause_end"), []):
                out += [f"> {n}", ""]
    for n in notes.get(("session_end",), []):
        out += [f"> {n}", ""]
    lead, tail = timeline["lead_in_ms"], timeline["tail_ms"]
    out.append(f"_Session padding: {secs_words(lead)} before the first word, {secs_words(tail)} after the last._")
    return "\n".join(out) + "\n"


def render_txt(script):
    paras = [" ".join(strip_tags(p["text"]) for p in seg["phrases"]) for seg in script["segments"]]
    return "\n\n".join(paras) + "\n"


def render_timing(script, est, measured):
    title = script.get("title") or "Meditation"
    t = script.get("timing") or {}
    wpm = t.get("words_per_minute", 90)
    words = sum(count_words(p["text"]) for s in script["segments"] for p in s["phrases"])
    phrase_rest = sum(float(p.get("pause_after_ms", 0)) for s in script["segments"] for p in s["phrases"]) / 1000
    seg_rest = sum(float(s.get("pause_after_ms", 0)) for s in script["segments"]) / 1000
    target = script.get("target_seconds")
    out = [f"# {title}: timing plan", ""]
    head = f"**Estimate {fmt_time(est['duration_ms'] / 1000)}"
    if target:
        head = f"**Rough target ~{fmt_time(target)} ({script.get('duration_mode', 'approximate')}). " + head[2:]
    out += [head + ".** " + f"{words} spoken words planned at {wpm:g} words/minute, {phrase_rest:g} s of rests "
            f"between phrases, {seg_rest:g} s of segment rests, {(est['lead_in_ms'] + est['tail_ms']) / 1000:g} s "
            "of lead-in and tail.", ""]
    out += ["The word rate is a planning estimate, not a voice-speed instruction. Each segment is generated as "
            "connected speech; rests are added only between complete phrases, counting the natural gap once.", ""]
    cr = script.get("closing_rest") or {}
    if cr:
        seg = timeline_segment(est, cr.get("segment_id"))
        if seg:
            out += ["## Closing rest", "",
                    f"Keep **{cr.get('duration_ms', 0) / 1000:g} s** (minimum {cr.get('minimum_duration_ms', 0) / 1000:g} s), "
                    f"estimated {fmt_time(seg['speech_end_ms'] / 1000)}-{fmt_time(seg['pause_end_ms'] / 1000)}. "
                    f"{cr.get('music', '')}", ""]
    out += ["## Estimated segment timing", "",
            "| Segment | Words | Spoken interval | Phrase rests | Following rest |",
            "| --- | ---: | --- | ---: | ---: |"]
    for seg, ts in zip(script["segments"], est["segments"]):
        w = sum(count_words(p["text"]) for p in seg["phrases"])
        pr = sum(float(p.get("pause_after_ms", 0)) for p in seg["phrases"]) / 1000
        out.append(f"| {ts['id']} | {w} | {fmt_time(ts['segment_start_ms'] / 1000)}-{fmt_time(ts['speech_end_ms'] / 1000)} "
                   f"| {pr:g}s | {float(seg.get('pause_after_ms', 0)) / 1000:g}s |")
    out.append("")

    def cue_table(clock, label):
        rows = []
        music = script.get("music") or {}
        if music.get("enabled"):
            for c in music.get("cues") or []:
                try:
                    t0 = resolve_anchor(c["anchor"], clock)
                except (KeyError, ValueError):
                    continue
                t1 = t0 + float(c.get("fade_ms", 0))
                rows.append(f"| {c.get('id')} | {fmt_time(t0 / 1000)}-{fmt_time(t1 / 1000)} | {c.get('target_gain')} | "
                            f"{c.get('direction', '')} |")
        if not rows:
            return []
        return [f"## Music cues ({label})", "", "| Cue | Interval | Target gain | Direction |",
                "| --- | --- | ---: | --- |"] + rows + [""]

    out += cue_table(est, "estimated")
    sfx = script.get("sfx") or {}
    if sfx.get("enabled"):
        out += ["## SFX (estimated)", ""]
        amb = sfx.get("ambience") or {}
        if amb.get("enabled"):
            out.append(f"Ambience: from gain {amb.get('initial_gain', 0)} with a {amb.get('fade_in_ms', 0) / 1000:g} s fade-in "
                       f"at the start, fading out over {amb.get('fade_out_ms', 0) / 1000:g} s to the session end; it "
                       "continues after the narration.")
            out.append("")
        shots = sfx.get("one_shots") or []
        if shots:
            out += ["| One-shot | Time | Gain | Sound |", "| --- | --- | ---: | --- |"]
            for s in shots:
                try:
                    t0 = resolve_anchor(s["anchor"], est)
                except (KeyError, ValueError):
                    continue
                sound = s.get("prompt") or f"same as {s.get('same_as')}"
                out.append(f"| {s.get('id')} | {fmt_time(t0 / 1000)} | {s.get('gain', 1)} | {sound} |")
            out.append("")
    if measured:
        out += ["## Measured after assembly", "",
                f"Measured length **{fmt_time(measured['duration_ms'] / 1000)}** (estimate "
                f"{fmt_time(est['duration_ms'] / 1000)}, "
                f"{'+' if measured['duration_ms'] >= est['duration_ms'] else '-'}"
                f"{fmt_time(abs(measured['duration_ms'] - est['duration_ms']) / 1000)}).", "",
                "| Segment | Take | Spoken interval | Words/min | Rest after | Start drift |",
                "| --- | --- | --- | ---: | ---: | ---: |"]
        for ms, es in zip(measured["segments"], est["segments"]):
            drift = (ms["segment_start_ms"] - es["segment_start_ms"]) / 1000
            wpm_s = f"{ms.get('wpm', 0):.0f}" if ms.get("wpm") else "-"
            out.append(f"| {ms['id']} | {ms.get('take', '-')} | {fmt_time(ms['segment_start_ms'] / 1000)}-"
                       f"{fmt_time(ms['speech_end_ms'] / 1000)} | {wpm_s} | "
                       f"{(ms['pause_end_ms'] - ms['speech_end_ms']) / 1000:g}s | {drift:+.1f}s |")
        out.append("")
        out += cue_table(measured, "measured clock")
    return "\n".join(out) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("session", help="session folder (or its script.json)")
    args = ap.parse_args()
    root = session_root(args.session)
    script = load_script(root)
    voice = read_json(root / "voice.json")
    est = estimated_timeline(script)
    measured = read_json(root / "voice" / "timeline.json")
    Path(root / "script.md").write_text(render_md(script, voice, est), encoding="utf-8")
    Path(root / "script.txt").write_text(render_txt(script), encoding="utf-8")
    Path(root / "timing.md").write_text(render_timing(script, est, measured), encoding="utf-8")
    print(f"wrote {root / 'script.md'}, script.txt, timing.md" + (" (with measured timing)" if measured else ""))


if __name__ == "__main__":
    main()
