#!/usr/bin/env python3
# /// script
# requires-python = ">=3.9"
# dependencies = ["numpy"]
# ///
"""
FIT MUSIC
Place composed music on the measured narration clock. Music composed for a session (a composition
plan whose sections recede, withdraw and return with the script's cues) only fits the timeline it
was planned on; after assembly, or after any retake that moves the timeline, this script re-times
it for free. Each part (one generated source, up to ten minutes) lists anchors: points in the
source that must land on a timeline boundary. Between two anchors the part repeats, or cuts, a
stretch of a steady section so the second anchor lands too, with one crossfaded join at the pair
of points whose rhythm, harmony, level and spectrum match best, placed in a rest.

  python3 scripts/fit_music.py SESSION            # music/timed.json -> music/fitted.flac, selected for mix.py
  python3 scripts/fit_music.py SESSION --dry-run  # the joins it would make

music/timed.json:
  {"parts": [
     {"source": "source-02", "use_s": [0, 340],
      "anchors": [{"at_s": 0, "anchor": {"boundary": "session_start"}},
                  {"at_s": 290, "anchor": {"segment_id": "07", "boundary": "speech_end"}, "section": "Withdrawal"}],
      "stretch_s": [[100, 215]]},
     {"source": "source-03", "gain_db": -3.3,
      "anchors": [{"at_s": 36, "anchor": {"segment_id": "09", "boundary": "segment_start"}},
                  {"at_s": 210, "anchor": {"segment_id": "12", "boundary": "speech_end"}}],
      "stretch_s": [[36, 150]]}]}

`stretch_s` names the steady sections (source seconds) where material may be repeated or cut;
never an intro, a withdrawal or an ending. Parts may overlap only where the cues keep the music
silent. The result is written as music/fitted.flac (the session length plus 3 s) with a report in
music/fitted.json, and selected, so mix.py plays it straight through from 0 s.
"""
from __future__ import annotations

import argparse

import numpy as np

from gm_common import (
    SR, cue_envelope, decode_audio, die, fmt_time, load_script, now_iso, read_json, resolve_anchor, session_root,
    sha256_file, to_mono, write_flac, write_json,
)

HOP = 0.02
EDGES = [60, 120, 250, 500, 1000, 2000, 4000, 8000, 16000]


def features(x):
    """Per 20 ms: onset strength, chroma (pitch classes 55 Hz-2 kHz), level (dB) and 8 band energies."""
    m = to_mono(x).astype(np.float64)
    win, hop = 4096, int(HOP * SR)
    n = max(1, 1 + (len(m) - win) // hop)
    w = np.hanning(win)
    f = np.fft.rfftfreq(win, 1.0 / SR)
    band = (f >= 55) & (f <= 2000)
    pc = np.mod(np.round(12 * np.log2(f[band] / 440.0)), 12).astype(int)
    bidx = np.digitize(f, EDGES) - 1
    bsel = (bidx >= 0) & (bidx < 8)
    onset, chroma, lev, bands = np.zeros(n), np.zeros((n, 12)), np.zeros(n), np.zeros((n, 8))
    prev = None
    for i in range(n):
        seg = m[i * hop: i * hop + win]
        if len(seg) < win:
            seg = np.pad(seg, (0, win - len(seg)))
        spec = np.abs(np.fft.rfft(seg * w))
        lm = np.log1p(spec * 1e3)
        if prev is not None:
            onset[i] = np.maximum(lm - prev, 0).mean()
        prev = lm
        chroma[i] = np.bincount(pc, weights=spec[band] ** 2, minlength=12)
        lev[i] = 10 * np.log10((spec ** 2).mean() + 1e-12)
        bands[i] = np.bincount(bidx[bsel], weights=spec[bsel] ** 2, minlength=8)
    chroma /= chroma.sum(1, keepdims=True) + 1e-12
    return onset, chroma, lev, np.vstack([np.zeros((1, 8)), np.cumsum(bands, 0)])


def corr(a, b):
    a, b = a - a.mean(), b - b.mean()
    return float((a * b).sum() / (np.sqrt((a * a).sum() * (b * b).sum()) + 1e-12))


def best_join(x, feats, d, lo, hi, out_at, in_rest, c_s=8.0, ctx_s=4.0, slack_s=1.0):
    """A join that lengthens the audio by about d seconds (d < 0 shortens it): play x up to T, then
    crossfade into x from S = T - d', with d' within slack_s of d (the anchor lands that close, and
    the join matches better). S, T and the crossfade stay inside [lo, hi]; the crossfade must land in
    a rest (out_at(T) on the timeline). Returns (S, T, report) or None."""
    onset, chroma, lev, cum = feats
    c, ctx = int(c_s / HOP), int(ctx_s / HOP)

    def band_db(i):
        return 10 * np.log10((cum[i + c] - cum[i]) / c + 1e-12)

    best = None
    n = len(onset)
    for t in np.arange(lo + ctx_s, hi - c_s, 0.1):
        o = out_at(t)
        if not in_rest(o, o + c_s):
            continue
        j = int(round(t / HOP))
        if j + c + ctx >= n:
            continue
        wo_t, ch_t, lv_t, bd_t = onset[j - ctx: j + c + ctx], chroma[j - ctx: j + c + ctx].mean(0), \
            lev[j: j + c].mean(), band_db(j)
        for dd in np.arange(d - slack_s, d + slack_s + 1e-9, 0.1):
            s = t - dd
            if s < lo + ctx_s or s + c_s > hi:
                continue
            # Beats line up only to within a few frames: take the best of +-80 ms around this restart.
            r_on, i = max((corr(onset[ii - ctx: ii + c + ctx], wo_t), ii)
                          for ii in range(int(round(s / HOP)) - 4, int(round(s / HOP)) + 5)
                          if ii - ctx >= 0 and ii + c + ctx < n) if s / HOP + c + ctx + 4 < n else (-2.0, 0)
            if r_on <= -2.0:
                continue
            s = i * HOP
            r_ch = corr(chroma[i - ctx: i + c + ctx].mean(0), ch_t)
            dl = abs(lev[i: i + c].mean() - lv_t)
            db = float(np.mean(np.abs(band_db(i) - bd_t)))
            score = r_on + r_ch - dl / 3.0 - db / 4.0 - abs(dd - d) / 4.0
            if best is None or score > best[0]:
                best = (score, s, t, {"rhythm_match": round(r_on, 2), "harmony_match": round(r_ch, 2),
                                      "level_diff_db": round(dl, 2), "band_diff_db": round(db, 2)})
    if best is None:
        return None
    _, s, t, rep = best
    # Refine the join to 5 ms so note attacks coincide inside the crossfade.
    i0, j0, cs, step = int(round(s * SR)), int(round(t * SR)), int(c_s * SR), int(0.005 * SR)
    a = np.abs(to_mono(x[i0: i0 + cs]))[::8]
    j0 = max(((corr(a, np.abs(to_mono(x[j: j + cs]))[::8]), j)
              for j in range(j0 - 12 * step, j0 + 13 * step, step)), key=lambda r: r[0])[1]
    return i0 / SR, j0 / SR, rep


def splice(x, s, t, c_s=8.0):
    """x up to t, a crossfade of x[t:] into x[s:], then x from s on (length changes by t - s)."""
    i0, j0, cs = int(round(s * SR)), int(round(t * SR)), int(c_s * SR)
    tail, head = x[j0: j0 + cs], x[i0: i0 + cs]
    r = corr(to_mono(tail), to_mono(head))
    u = ((np.arange(cs) + 0.5) / cs)[:, None]
    fout, fin = (1 - u, u) if r > 0.6 else (np.cos(u * np.pi / 2), np.sin(u * np.pi / 2))
    return np.vstack([x[:j0], tail * fout + head * fin, x[i0 + cs:]]).astype(np.float32), \
        "equal-gain" if r > 0.6 else "equal-power"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("session", help="session folder (or its script.json)")
    ap.add_argument("--crossfade-s", type=float, default=8.0, help="length of each join (default 8)")
    ap.add_argument("--dry-run", action="store_true", help="report the joins; write nothing")
    args = ap.parse_args()

    root = session_root(args.session)
    script = load_script(root)
    tl = read_json(root / "voice" / "timeline.json")
    spec = read_json(root / "music" / "timed.json")
    if not tl:
        die("run assemble_voice.py first: the fit needs the measured timeline")
    if not spec or not spec.get("parts"):
        die("music/timed.json lists no parts (see --help)")
    total = tl["duration_ms"] / 1000.0
    segs = tl["segments"]
    rests = [(a["speech_end_ms"] / 1000 + 1.0, b["segment_start_ms"] / 1000 - 1.0) for a, b in zip(segs, segs[1:])]
    rests.append((segs[-1]["speech_end_ms"] / 1000 + 1.0, total))

    def in_rest(t0, t1):
        return any(a <= t0 and t1 <= b for a, b in rests)

    out = np.zeros((int((total + 3.0) * SR), 2), dtype=np.float32)
    report = {"created_at": now_iso(), "file": "fitted.flac", "session_s": round(total, 2), "parts": []}
    for k, part in enumerate(spec["parts"], 1):
        meta = read_json(root / "music" / f"{part['source']}.json") or {}
        src = root / "music" / (meta.get("audio_file") or f"{part['source']}.mp3")
        if not src.exists():
            die(f"part {k}: {src} not found")
        x = decode_audio(src, SR, 2)
        a0, a1 = part.get("use_s") or (0.0, len(x) / SR)
        x = (x[int(a0 * SR): int(a1 * SR)] * (10 ** (float(part.get("gain_db", 0)) / 20))).astype(np.float32)
        anchors = sorted(({"at_s": float(an["at_s"]) - a0, "t": resolve_anchor(an["anchor"], tl) / 1000.0,
                           "label": an.get("section") or an["anchor"].get("boundary")} for an in part["anchors"]),
                         key=lambda an: an["at_s"])
        if not anchors:
            die(f"part {k} ({part['source']}) needs at least one anchor")
        start = anchors[0]["t"] - anchors[0]["at_s"]       # the first anchor places the part
        stretch = [(lo - a0, hi - a0) for lo, hi in part.get("stretch_s") or [[a0, a1]]]
        joins = []
        feats = features(x)
        for an, nx in zip(anchors, anchors[1:]):
            d = (nx["t"] - an["t"]) - (nx["at_s"] - an["at_s"])
            if abs(d) < 0.25:
                continue
            found = None
            for lo, hi in stretch:
                lo, hi = max(lo, an["at_s"]), min(hi, nx["at_s"])
                if hi - lo > abs(d) + args.crossfade_s + 8.0:
                    found = best_join(x, feats, d, lo, hi, lambda t: start + t, in_rest, args.crossfade_s)
                    if found:
                        break
            if not found:
                die(f"part {k}: no place to {'repeat' if d > 0 else 'cut'} {abs(d):.1f}s between {an['label']} and "
                    f"{nx['label']} inside a rest; widen stretch_s or check the anchors")
            s, t, rep = found
            x, fade = splice(x, s, t, args.crossfade_s)
            feats = features(x) if nx is not anchors[-1] else feats
            joins.append({"between": f"{an['label']} -> {nx['label']}", "change_s": round(t - s, 2),
                          "part_restart_s": round(s, 2), "part_join_s": round(t, 2),
                          "timeline_s": round(start + t, 2), "fade": fade, **rep})
            for later in anchors:
                if later["at_s"] > t:
                    later["at_s"] += t - s
            stretch = [(lo + (t - s if lo > t else 0), hi + (t - s if hi > t else 0)) for lo, hi in stretch]
        i = int(round(start * SR))
        lo_i, hi_i = max(0, i), min(len(out), i + len(x))
        if hi_i > lo_i:
            out[lo_i:hi_i] += x[lo_i - i: hi_i - i]
        report["parts"].append({"source": part["source"], "starts_s": round(start, 2),
                                "ends_s": round(start + len(x) / SR, 2), "gain_db": part.get("gain_db", 0),
                                "anchors": [{"section": an["label"], "timeline_s": round(an["t"], 2)} for an in anchors],
                                "joins": joins})
        print(f"part {k} {part['source']}: {fmt_time(start)}-{fmt_time(start + len(x) / SR)}; "
              + ("; ".join(f"{j['between']}: {'repeats' if j['change_s'] > 0 else 'cuts'} {abs(j['change_s']):.1f}s, "
                           f"join at {fmt_time(j['timeline_s'])} (harmony {j['harmony_match']}, rhythm "
                           f"{j['rhythm_match']}, level {j['level_diff_db']} dB)" for j in joins) or "no joins needed"))
    for p, q in zip(report["parts"], report["parts"][1:]):
        if q["starts_s"] < p["ends_s"]:
            print(f"parts overlap {fmt_time(q['starts_s'])}-{fmt_time(p['ends_s'])}: make sure the cues keep the music "
                  "silent there")
    # Where the cues ask for music but no part plays, the bed would fall silent.
    music = script.get("music") or {}
    env = cue_envelope(float(music.get("initial_gain", 0)), music.get("cues") or [], tl)
    uncovered = [t for t in np.arange(0.0, total, 0.5) if env.value_at(t * 1000.0) > 0.05
                 and not any(p["starts_s"] <= t < p["ends_s"] for p in report["parts"])]
    if uncovered:
        report["uncovered_s"] = [round(uncovered[0], 1), round(uncovered[-1], 1)]
        print(f"WARNING: the cues ask for music between {fmt_time(uncovered[0])} and {fmt_time(uncovered[-1])} "
              "where no part plays; add a part or move the anchors")
    if args.dry_run:
        return
    write_flac(root / "music" / "fitted.flac", out)
    meta = {"audio_file": "fitted.flac", "timed": True, "sha256": sha256_file(root / "music" / "fitted.flac"), **report}
    write_json(root / "music" / "fitted.json", meta)
    sel = read_json(root / "music" / "selection.json", {}) or {}
    write_json(root / "music" / "selection.json", {"selected": "fitted", "updated_at": now_iso(),
                                                   "previous": sel.get("selected")})
    print(f"music/fitted.flac ({fmt_time(len(out) / SR)}) selected; mix.py plays it from 0 s. Listen at each join.")


if __name__ == "__main__":
    main()
