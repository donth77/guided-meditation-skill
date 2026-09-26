#!/usr/bin/env python3
# /// script
# requires-python = ">=3.9"
# dependencies = ["numpy"]
# ///
"""
ASSEMBLE VOICE
Build the raw narration track from the selected take of every segment. Each passage is split
only at its authored phrase boundaries (character alignment corroborated by the waveform);
at each boundary the natural gap counts once and only the missing rest is added, keeping
breaths. Segment rests, the lead-in and the tail follow. Writes:

  voice/voice-track.wav   float32 mono 44.1 kHz, unnormalised
  voice/timeline.json     measured segment_start / speech_end / pause_end for every segment,
                          phrase times, natural gaps and inserted rests; every music, ambience
                          and SFX cue resolves against this clock

  python3 scripts/assemble_voice.py SESSION
  python3 scripts/assemble_voice.py SESSION --lead-keep-ms 1500 --tail-keep-ms 1000

Nothing inside a phrase is cut, moved, stretched or re-timed. Screening notes (internal gaps,
unusual speech rates, clipping, missing alignment) are printed and stored in the timeline;
they point at passages to listen to, they do not approve or reject anything.
"""
from __future__ import annotations

import argparse
import math

import numpy as np

from gm_common import (
    SR, WavWriter, align_map, count_words, decode_audio, die, fmt_time, format_pauses, is_v3, level_db, load_script,
    now_iso,
    quiet_threshold, read_json, runs, segment_request, session_pads, session_root, sha256_file, spoken_mask, warn,
    write_json,
)

HOP_MS, WIN_MS = 10.0, 30.0


class Frames:
    def __init__(self, x, sr=SR):
        self.lev, self.hop, self.win = level_db(x, sr, WIN_MS, HOP_MS)
        self.sr = sr
        # Whispered onsets and decays sit 35-45 dB under the loudest frame, so "speech" reaches
        # down to peak - 40 dB; true silence between phrases measures lower than peak - 52 dB.
        peak = float(np.max(self.lev))
        self.t_speech = max(peak - 40.0, -65.0)
        self.t_sil = max(peak - 52.0, -75.0)
        # What a listener hears as a pause (breath and room tone included); used to measure the
        # natural pause at a phrase boundary, never to decide what may be cut.
        self.quiet = self.lev < quiet_threshold(self.lev)

    def center(self, i):
        return (i * self.hop + self.win / 2) / self.sr

    def quiet_run(self, t, lo_s, hi_s):
        """(start, end) seconds of the run of pause-quiet frames around t, within [lo_s, hi_s];
        (t, t) when the frame at t is not quiet."""
        A, B = self.span(lo_s, hi_s)
        if B <= A:
            return t, t
        i = min(max(int(round((t * self.sr - self.win / 2) / self.hop)), A), B - 1)
        if not self.quiet[i]:
            return t, t
        a, b = i, i
        while a > A and self.quiet[a - 1]:
            a -= 1
        while b < B - 1 and self.quiet[b + 1]:
            b += 1
        half = self.hop / (2 * self.sr)
        return self.center(a) - half, self.center(b) + half

    def span(self, a_s, b_s):
        a = max(0, int(math.floor(a_s * self.sr / self.hop)))
        b = min(len(self.lev), int(math.ceil(b_s * self.sr / self.hop)) + 1)
        return a, max(a, b)

    def loud(self, a_s, b_s):
        a, b = self.span(a_s, b_s)
        return [a + i for i in np.where(self.lev[a:b] >= self.t_speech)[0].tolist()]

    def quietest(self, a_s, b_s):
        a, b = self.span(a_s, b_s)
        if b <= a:
            return a_s
        return self.center(a + int(np.argmin(self.lev[a:b])))


def phrase_edges(alignment, text, spans):
    """(first spoken char start, last spoken char end) per phrase from the alignment, else None."""
    if not alignment or not alignment.get("characters"):
        return None
    st = alignment["character_start_times_seconds"]
    en = alignment["character_end_times_seconds"]
    mapping = align_map(text, alignment["characters"])
    mask = spoken_mask(text)
    edges = []
    for a, b in spans:
        idx = [mapping[k] for k in range(a, b) if mask[k] and mapping[k] is not None]
        if not idx:
            return None
        edges.append((float(st[idx[0]]), float(en[idx[-1]])))
    return edges


def fallback_edges(fr, text, spans, onset, offset):
    """Estimate phrase edges without alignment: proportional position, snapped to the nearest
    quiet run of at least 100 ms."""
    mask = spoken_mask(text)
    cum = np.cumsum(mask)
    total = max(1, int(cum[-1])) if len(cum) else 1
    edges = [[onset, None]]
    for a, b in spans[:-1]:
        t = onset + (cum[b - 1] / total) * (offset - onset)
        A, B = fr.span(t - 1.2, t + 1.2)
        rs = [(s, e) for s, e in runs(fr.lev[A:B] < fr.t_sil) if (e - s) * fr.hop / fr.sr >= 0.1]
        if rs:
            s, e = min(rs, key=lambda r: abs(fr.center(A + (r[0] + r[1]) / 2) - t))
            t_end, t_start = fr.center(A + s), fr.center(A + e - 1)
        else:
            t_end = t_start = t
        edges[-1][1] = t_end
        edges.append([t_start, None])
    edges[-1][1] = offset
    return [tuple(e) for e in edges]


def analyze(x, meta, alignment, n_phrases, pauses_ms, args, sr=SR):
    """Speech edges, keep region and phrase cuts for one passage (times in seconds)."""
    fr = Frames(x, sr)
    notes = []
    loud_all = np.where(fr.lev >= fr.t_speech)[0]
    audible = np.where(fr.lev >= fr.t_sil)[0]
    if not len(loud_all):
        die(f"segment {meta.get('segment_id')}: take {meta.get('take')} contains no speech")
    text, spans = meta["text"], [tuple(s) for s in meta["phrase_spans"]]
    if len(spans) != n_phrases:
        die(f"segment {meta.get('segment_id')}: {meta.get('take')} has {len(spans)} phrases but the script has "
            f"{n_phrases}; retake it (synthesize.py --segments {meta.get('segment_id')} --retake)")
    edges = phrase_edges(alignment, text, spans)
    if edges is None:
        notes.append("no usable alignment: phrase boundaries found from the waveform alone; listen to the joins")
        edges = fallback_edges(fr, text, spans, fr.center(loud_all[0]), fr.center(loud_all[-1]))
    # Segment speech edges, refined against the waveform. Alignments often start the first
    # character at 0.0 s ahead of a breath, so the onset search reaches well past it.
    t0, t1 = edges[0][0], edges[-1][1]
    loud = fr.loud(t0 - 0.35, t0 + 0.7)
    onset = fr.center(loud[0]) - fr.hop / (2 * sr) if loud else t0
    loud = fr.loud(t1 - 0.4, t1 + 0.5)
    offset = fr.center(loud[-1]) + fr.hop / (2 * sr) if loud else t1
    onset, offset = max(0.0, onset), min(len(x) / sr, offset)
    aud_start = fr.center(audible[0]) - fr.win / (2 * sr)
    aud_end = fr.center(audible[-1]) + fr.win / (2 * sr)
    keep_start = max(0.0, max(aud_start, onset - args.lead_keep_ms / 1000) - 0.03)
    keep_end = min(len(x) / sr, min(aud_end, offset + args.tail_keep_ms / 1000) + 0.06)
    keep_start, keep_end = min(keep_start, onset), max(keep_end, offset)

    phrases = [[onset, None]]
    cuts = []
    for i in range(n_phrases - 1):
        t_end, t_start = edges[i][1], edges[i + 1][0]
        lo, hi = min(t_end, t_start), max(t_end, t_start)
        mid = (lo + hi) / 2
        # Last loud frame of phrase i: search up to the middle of the gap only, so the onset of
        # phrase i+1 (whose analysis window starts early) is never mistaken for it.
        half = fr.win / (2 * sr)
        loud = fr.loud(t_end - 0.4, max(t_end - 0.4, min(mid - half, t_end + 0.35)))
        p_off = fr.center(loud[-1]) + fr.hop / (2 * sr) if loud else t_end
        loud = fr.loud(max(p_off, mid + half, t_start - 0.35), t_start + 0.45)
        p_on = fr.center(loud[0]) - fr.hop / (2 * sr) if loud else t_start
        if p_on <= p_off + 0.005:
            cut = fr.quietest(lo - 0.1, hi + 0.1)
            p_on = p_off = cut
            gap = 0.0
        else:
            a, b = fr.span(p_off, p_on)
            rs = runs(fr.lev[a:b] < fr.t_sil)
            if rs:
                s, e = max(rs, key=lambda r: r[1] - r[0])
                cut = fr.center(a + (s + e - 1) / 2)
            else:
                cut = fr.quietest(p_off, p_on)
            cut = min(max(cut, p_off), p_on)
            gap = p_on - p_off
        # The pause as heard: breath or room tone between the phrases counts as pause, so a voice
        # recorded with an audible floor does not get a longer rest than asked.
        q0, q1 = fr.quiet_run(cut, lo - 0.6, hi + 0.6)
        gap = max(gap, q1 - q0)
        if gap < 0.005:
            notes.append(f"phrases {i + 1}/{i + 2} are connected with no audible gap; the rest is inserted at the "
                         f"quietest point ({cut:.2f}s)")
        want = pauses_ms[i] / 1000
        inserted = max(0.0, want - gap)
        if gap > want + 0.4:
            notes.append(f"natural gap after phrase {i + 1} is {gap:.2f}s, longer than the {want:.2f}s asked")
        # Fade the quiet material on either side of an inserted rest (up to 150 ms) instead of
        # switching a breathy floor off in 5 ms.
        cuts.append({"cut": cut, "inserted": inserted, "gap": gap,
                     "fade_out": min(max(cut - q0, 0.005), 0.15), "fade_in": min(max(q1 - cut, 0.005), 0.15)})
        phrases[-1][1] = p_off
        phrases.append([p_on, None])
    phrases[-1][1] = offset

    # Screening: speech rate and internal gaps per phrase, clipping.
    for i, ((a, b), (s, e)) in enumerate(zip(spans, phrases)):
        w = count_words(text[a:b])
        dur = max(e - s, 0.05)
        if w >= 5:
            wpm = w / dur * 60
            if wpm > 185:
                notes.append(f"phrase {i + 1}: ~{wpm:.0f} words/min, which may sound rushed")
            elif wpm < 70:
                notes.append(f"phrase {i + 1}: ~{wpm:.0f} words/min, which may sound dragged or word-by-word")
        A, B = fr.span(s + 0.05, e - 0.05)
        for rs, re_ in runs(fr.lev[A:B] < fr.t_sil):
            g = (re_ - rs) * fr.hop / sr
            if g >= args.internal_gap_ms / 1000:
                notes.append(f"phrase {i + 1}: {g:.2f}s silence inside the phrase at {fr.center(A + rs):.2f}s "
                             "(a misplaced gap? listen, retake the passage if it sounds broken)")
    clipped = int(np.sum(np.abs(x) >= 0.999))
    if clipped:
        notes.append(f"{clipped} clipped samples in the take")
    return {"onset": onset, "offset": offset, "keep_start": keep_start, "keep_end": keep_end,
            "phrases": phrases, "cuts": cuts, "notes": notes, "aligned": "no usable alignment" not in " ".join(notes)}


def render_segment(x, an, sr=SR):
    """Audio for one segment from keep_start to keep_end with rests inserted at the cuts.
    Returns (audio, map_fn) where map_fn maps take seconds -> seconds from the segment's first sample."""
    ks = int(round(an["keep_start"] * sr))
    ke = int(round(an["keep_end"] * sr))

    def ramp(n):
        return (0.5 - 0.5 * np.cos(np.pi * np.arange(n) / n)).astype(np.float32)

    x = x.copy()
    pieces, pos = [], ks
    for c in an["cuts"]:
        cs = int(round(c["cut"] * sr))
        piece = x[pos:cs]
        ins = int(round(c["inserted"] * sr))
        if ins:
            fo = min(len(piece), int(round(c.get("fade_out", 0.005) * sr)))
            fi = min(max(0, ke - cs), int(round(c.get("fade_in", 0.005) * sr)))
            if fo:
                piece[-fo:] *= ramp(fo)[::-1]
            if fi:
                x[cs: cs + fi] *= ramp(fi)
        pieces.append(piece)
        pieces.append(np.zeros(ins, dtype=np.float32))
        pos = cs
    pieces.append(x[pos:ke])
    audio = np.concatenate(pieces) if pieces else np.zeros(0, dtype=np.float32)
    edge = int(0.01 * sr)
    if len(audio) > 2 * edge:
        audio[:edge] *= np.linspace(0.0, 1.0, edge, dtype=np.float32)
        audio[-edge:] *= np.linspace(1.0, 0.0, edge, dtype=np.float32)
    cut_list = [(c["cut"], c["inserted"]) for c in an["cuts"]]

    def map_fn(t):
        return (t - an["keep_start"]) + sum(ins for cut, ins in cut_list if cut < t)

    return audio, map_fn


def preview(x, meta, alignment, seg, sr=SR):
    """One take with its segment's phrase rests inserted and its edges trimmed as assembly
    would: how an audition will sit in the voice track. Returns (audio, analysis)."""
    args = argparse.Namespace(lead_keep_ms=1500.0, tail_keep_ms=1000.0, internal_gap_ms=1200.0)
    pauses = [float(p.get("pause_after_ms", 0)) for p in seg["phrases"][:-1]]
    an = analyze(x, meta, alignment, len(seg["phrases"]), pauses, args, sr)
    audio, _ = render_segment(x, an, sr)
    return audio, an


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("session", help="session folder (or its script.json)")
    ap.add_argument("--lead-keep-ms", type=float, default=1500.0,
                    help="audible material kept before a segment's first word (breaths, exhales); default 1500")
    ap.add_argument("--tail-keep-ms", type=float, default=1000.0,
                    help="audible material kept after a segment's last word; default 1000")
    ap.add_argument("--internal-gap-ms", type=float, default=1200.0,
                    help="flag silences this long inside a phrase; default 1200")
    args = ap.parse_args()

    root = session_root(args.session)
    script = load_script(root)
    sel = (read_json(root / "voice" / "selection.json", {}) or {}).get("segments", {})
    lead_ms, tail_ms = session_pads(script)
    segs = script["segments"]

    loaded = []
    mock = False
    for seg in segs:
        sid = str(seg["id"])
        entry = sel.get(sid)
        if not entry:
            die(f"segment {sid} has no selected take; run synthesize.py first")
        folder = root / "voice" / "passages" / sid
        meta = read_json(folder / f"{entry['take']}.json")
        if not meta:
            die(f"segment {sid}: {entry['take']}.json is missing")
        mock |= bool(meta.get("mock"))
        current, _ = segment_request(seg, is_v3(meta.get("model_id")))
        if current != meta.get("text"):
            warn(f"segment {sid}: the script text changed after {entry['take']} was generated; the take's own "
                 f"text is used. Retake it if the words matter (synthesize.py --segments {sid} --retake)")
        alignment = read_json(folder / meta["alignment_file"]) if meta.get("alignment_file") else None
        gap_note = [f"pauses inside phrases: {format_pauses(meta['inner_pauses'])}"
                    " (word-by-word delivery? listen; retake the passage if it drags)"] if meta.get("inner_pauses") else []
        x = decode_audio(folder / meta["audio_file"], SR, 1)[:, 0]
        pauses = [float(p.get("pause_after_ms", 0)) for p in seg["phrases"][:-1]]
        an = analyze(x, meta, alignment, len(seg["phrases"]), pauses, args)
        an["notes"] += gap_note
        loaded.append((seg, entry, meta, x, an))

    # Fit each segment rest (word to word) around the material kept at the edges.
    for i, (seg, _, _, _, an) in enumerate(loaded[:-1]):
        nxt = loaded[i + 1][4]
        rest = float(seg.get("pause_after_ms", 0)) / 1000
        tail_mat = an["keep_end"] - an["offset"]
        lead_mat = nxt["onset"] - nxt["keep_start"]
        over = tail_mat + lead_mat - rest
        if over > 0:
            cut_tail = min(over, max(0.0, tail_mat - 0.15))
            an["keep_end"] -= cut_tail
            over -= cut_tail
            cut_lead = min(over, max(0.0, lead_mat - 0.1))
            nxt["keep_start"] += cut_lead
            over -= cut_lead
            if over > 0.001:
                an["notes"].append(f"rest after {seg['id']} is {over:.2f}s longer than planned (too short for the "
                                   "kept breath and decay)")

    out_path = root / "voice" / "voice-track.wav"
    timeline = {"kind": "measured", "sample_rate": SR, "lead_in_ms": lead_ms, "tail_ms": tail_ms,
                "voice_track": "voice/voice-track.wav", "created_at": now_iso(), "mock": mock, "segments": []}
    all_notes = []
    with WavWriter(out_path, SR, 1) as w:
        t_out = 0.0
        for i, (seg, entry, meta, x, an) in enumerate(loaded):
            sid = str(seg["id"])
            audio, map_fn = render_segment(x, an)
            if i == 0:
                pre = max(0.0, lead_ms / 1000 - (an["onset"] - an["keep_start"]))
            else:
                prev_seg, _, _, _, prev_an = loaded[i - 1]
                prev_rest = float(prev_seg.get("pause_after_ms", 0)) / 1000
                pre = max(0.0, prev_rest - (prev_an["keep_end"] - prev_an["offset"]) - (an["onset"] - an["keep_start"]))
            w.write_silence(int(round(pre * SR)))
            t_out += int(round(pre * SR)) / SR
            origin = t_out
            w.write(audio)
            t_out += len(audio) / SR
            phrases = []
            text, spans = meta["text"], meta["phrase_spans"]
            for k, ((a, b), (s, e)) in enumerate(zip(spans, an["phrases"])):
                wds = count_words(text[a:b])
                start_ms, end_ms = (origin + map_fn(s)) * 1000, (origin + map_fn(e)) * 1000
                row = {"index": k, "start_ms": round(start_ms, 1), "end_ms": round(end_ms, 1), "words": wds,
                       "wpm": round(wds / max((end_ms - start_ms) / 60000, 1e-6), 1)}
                if k < len(an["cuts"]):
                    c = an["cuts"][k]
                    row.update({"pause_after_ms": seg["phrases"][k].get("pause_after_ms", 0),
                                "natural_gap_ms": round(c["gap"] * 1000, 1), "inserted_ms": round(c["inserted"] * 1000, 1)})
                phrases.append(row)
            seg_start = (origin + map_fn(an["onset"])) * 1000
            speech_end = (origin + map_fn(an["offset"])) * 1000
            words_total = sum(p["words"] for p in phrases)
            speaking = sum(p["end_ms"] - p["start_ms"] for p in phrases) / 1000
            timeline["segments"].append({
                "id": sid, "take": entry["take"], "frozen": bool(entry.get("frozen")),
                "source_sha256": meta.get("audio_sha256"), "segment_start_ms": round(seg_start, 1),
                "speech_end_ms": round(speech_end, 1), "pause_end_ms": None,
                "planned_pause_after_ms": float(seg.get("pause_after_ms", 0)), "words": words_total,
                "speech_seconds": round(speaking, 2), "wpm": round(words_total / max(speaking / 60, 1e-6), 1),
                "aligned": an["aligned"], "phrases": phrases, "notes": an["notes"]})
            all_notes += [f"[{sid}] {n}" for n in an["notes"]]
        last_an = loaded[-1][4]
        tail = max(0.0, tail_ms / 1000 - (last_an["keep_end"] - last_an["offset"]))
        w.write_silence(int(round(tail * SR)))
        t_out += int(round(tail * SR)) / SR
    segs_tl = timeline["segments"]
    for i, s in enumerate(segs_tl):
        s["pause_end_ms"] = segs_tl[i + 1]["segment_start_ms"] if i + 1 < len(segs_tl) else s["speech_end_ms"]
    timeline["duration_ms"] = round(t_out * 1000, 1)
    timeline["voice_track_sha256"] = sha256_file(out_path)
    timeline["notes"] = all_notes
    write_json(root / "voice" / "timeline.json", timeline)

    print(f"{'seg':<5}{'take':<9}{'start':>9}{'speech':>8}{'wpm':>6}{'rest after':>12}  phrase joins (natural+inserted)")
    for s in segs_tl:
        joins = ", ".join(f"{p['natural_gap_ms'] / 1000:.2f}+{p['inserted_ms'] / 1000:.2f}" for p in s["phrases"]
                          if "natural_gap_ms" in p)
        print(f"{s['id']:<5}{s['take']:<9}{fmt_time(s['segment_start_ms'] / 1000):>9}{s['speech_seconds']:>7.1f}s"
              f"{s['wpm']:>6.0f}{(s['pause_end_ms'] - s['speech_end_ms']) / 1000:>11.1f}s  {joins}")
    print(f"voice track {fmt_time(timeline['duration_ms'] / 1000)} -> {out_path.relative_to(root)}; timeline -> "
          "voice/timeline.json")
    if mock:
        print("MOCK takes: timing and mixing rehearsal only, not a deliverable voice")
    if all_notes:
        print("Screening notes (listen to these spots):")
        for n in all_notes:
            print(f"  - {n}")


if __name__ == "__main__":
    main()
