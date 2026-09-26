#!/usr/bin/env python3
# /// script
# requires-python = ">=3.9"
# dependencies = ["numpy"]
# ///
"""
QA REPORT
Measure the delivered files, check them against the script and the measured timeline, and write
qa/report.md and qa/report.json with a listening checklist.

  python3 scripts/qa_report.py SESSION
  python3 scripts/qa_report.py SESSION --transcribe            # + speech-to-text check per passage (paid)
  python3 scripts/qa_report.py SESSION --transcribe --dry-run  # audio seconds to be transcribed

Checks: every file in output/manifest.json exists and decodes; duration, sample rate, channels;
integrated loudness, loudness range, true peak; sample peak and clipped samples; the closing rest
and music-free interludes on the measured clock; music starting and ending in silence; ambience
still present after the last word; one-shots audible at their times; speech-rate and
internal-gap screening per phrase; transcript word differences (with --transcribe).
Exit 1 when a check FAILs. These checks screen; only listening approves accent, calm, phrasing
and balance.
"""
from __future__ import annotations

import argparse
import difflib
import json
from pathlib import Path

import numpy as np

from gm_common import (
    SR, ApiError, api_key, api_request, die, fail_api, fmt_time, iter_decode, ledger, load_script,
    measure_loudness, multipart, now_iso, open_wav_f32, probe_stream, read_json, session_root, strip_tags,
    usage_count, words, write_json,
)

CHECKLIST = [
    ("Voice", [
        "Accent and vocal character stay the same from the first passage to the last.",
        "Delivery is calm and unhurried without dragging; phrases are connected, no word-by-word separation.",
        "No misplaced pauses inside phrases; no rushed passages.",
        "Audio tags (if any) are performed, never spoken as words; no clicks, cut breaths or metallic artifacts.",
    ]),
    ("Timing", [
        "Rests feel intentional; early rests are short, sustained rests give room.",
        "The closing rest is long enough, and the final line arrives unhurried.",
    ]),
    ("Music", [
        "No vocals, beat, sudden swells or bright accents; it stays behind the voice.",
        "Loop joins are inaudible (no restart, gap or level jump); fades are smooth; silent where planned.",
    ]),
    ("SFX", [
        "The ambience is recognisable under the voice and still there after the last word.",
        "No obvious repetition of distinctive events; no clicks at loop joins.",
        "One-shots sit at a comfortable level and are not clipped at the start or end.",
    ]),
    ("Overall", [
        "A comfortable playback level at normal volume; nothing startling from start to finish.",
    ]),
]


def rms_db(x):
    if x is None or len(x) == 0:
        return -120.0
    a = np.asarray(x, dtype=np.float64)
    return float(10 * np.log10(np.mean(a * a) + 1e-12))


def window(stem, a_s, b_s):
    a, b = max(0, int(a_s * SR)), min(len(stem), int(b_s * SR))
    return stem[a:b] if b > a else None


def file_stats(path, channels):
    peak, clipped, frames = 0.0, 0, 0
    for blk in iter_decode(path, SR, channels):
        peak = max(peak, float(np.max(np.abs(blk))) if len(blk) else 0.0)
        clipped += int(np.sum(np.abs(blk) >= 0.999))
        frames += len(blk)
    return {"sample_peak_dbfs": round(20 * np.log10(max(peak, 1e-9)), 2), "clipped_samples": clipped,
            "decoded_s": round(frames / SR, 3)}


def norm_words(text):
    return [w.lower().replace("’", "'") for w in words(text)]


def transcribe(key, path, model, language):
    body, ctype = multipart({"model_id": model, "language_code": language, "timestamps_granularity": "word",
                             "tag_audio_events": "false"},
                            {"file": (Path(path).name, Path(path).read_bytes(), "application/octet-stream")})
    raw, _ = api_request("POST", "/v1/speech-to-text", key=key, raw=body, content_type=ctype, timeout=600, retries=2)
    return json.loads(raw.decode("utf-8"))


def word_diff(expected, heard):
    sm = difflib.SequenceMatcher(None, expected, heard, autojunk=False)
    edits, diffs = 0, []
    for op, a0, a1, b0, b1 in sm.get_opcodes():
        if op == "equal":
            continue
        edits += max(a1 - a0, b1 - b0)
        diffs.append(f"{op}: '{' '.join(expected[a0:a1])}' -> '{' '.join(heard[b0:b1])}'")
    return edits / max(1, len(expected)), diffs


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("session", help="session folder (or its script.json)")
    ap.add_argument("--transcribe", action="store_true", help="speech-to-text each selected passage and diff the words")
    ap.add_argument("--stt-model", default="scribe_v2")
    ap.add_argument("--force", action="store_true", help="re-transcribe passages that already have a transcript")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    root = session_root(args.session)
    script = load_script(root)
    timeline = read_json(root / "voice" / "timeline.json")
    manifest = read_json(root / "output" / "manifest.json")
    if not timeline or not manifest:
        die("run assemble_voice.py and mix.py first (voice/timeline.json, output/manifest.json)")
    checks, notes = [], []

    def check(status, name, detail):
        checks.append({"status": status, "check": name, "detail": detail})

    # Deliverables.
    files = {}
    durations = []
    for v, d in manifest["variants"].items():
        for rel_path in d["files"]:
            p = root / rel_path
            if not p.exists():
                check("FAIL", f"{v} file", f"{rel_path} is missing")
                continue
            info = probe_stream(p)
            loud = measure_loudness(p, dualmono=info.get("channels") == 1)
            stats = file_stats(p, max(1, info.get("channels") or 1))
            files[rel_path] = {"variant": v, **info, **loud, **stats}
            durations.append(info.get("duration") or 0)
            if stats["clipped_samples"]:
                check("FAIL", f"{rel_path} clipping", f"{stats['clipped_samples']} samples at full scale")
            if loud["TP"] is not None and loud["TP"] > manifest.get("peak_ceiling_dbtp", -1.0) + 0.7:
                check("WARN", f"{rel_path} true peak", f"{loud['TP']} dBTP (ceiling {manifest.get('peak_ceiling_dbtp')})")
    if durations:
        spread = max(durations) - min(durations)
        check("PASS" if spread < 0.2 else "WARN", "equal durations", f"all files within {spread:.2f}s")

    # Timeline.
    target = script.get("target_seconds")
    dur = timeline["duration_ms"] / 1000
    if target:
        mode = script.get("duration_mode", "approximate")
        diff = dur - target
        status = "PASS" if mode == "approximate" or abs(diff) <= max(5, 0.02 * target) else "FAIL"
        check(status, "session length", f"{fmt_time(dur)} measured vs ~{fmt_time(target)} target ({mode}, "
                                         f"{'+' if diff >= 0 else '-'}{fmt_time(abs(diff))})")
    segs = {s["id"]: s for s in timeline["segments"]}
    cr = script.get("closing_rest") or {}
    if cr and str(cr.get("segment_id")) in segs:
        s = segs[str(cr["segment_id"])]
        rest = s["pause_end_ms"] - s["speech_end_ms"]
        mn = float(cr.get("minimum_duration_ms", 0) or 0)
        check("PASS" if rest >= mn else "FAIL", "closing rest", f"{rest / 1000:.1f}s (minimum {mn / 1000:.0f}s)")
    for s in timeline["segments"]:
        for n in s.get("notes", []):
            notes.append(f"[{s['id']}] {n}")

    stems = root / "stems"
    music_path, sfx_path = stems / "music.wav", stems / "sfx.wav"
    end_s = manifest.get("duration_s", dur)
    if "music" in manifest.get("stems", {}) and music_path.exists():
        m, _ = open_wav_f32(music_path)
        a, b = rms_db(window(m, 0, 1.0)), rms_db(window(m, end_s - 1.0, end_s))
        check("PASS" if max(a, b) < -70 else "FAIL", "music starts and ends in silence",
              f"first second {a:.0f} dBFS, last second {b:.0f} dBFS")
        for sid in (script.get("music") or {}).get("music_free_pauses") or []:
            s = segs.get(str(sid))
            if s:
                lvl = rms_db(window(m, s["speech_end_ms"] / 1000, s["pause_end_ms"] / 1000))
                check("PASS" if lvl < -70 else "FAIL", f"music-free pause after {sid}", f"music at {lvl:.0f} dBFS")
    if "sfx" in manifest.get("stems", {}) and sfx_path.exists():
        sx, _ = open_wav_f32(sfx_path)
        amb = ((script.get("sfx") or {}).get("ambience") or {})
        if amb.get("enabled"):
            last = timeline["segments"][-1]["speech_end_ms"] / 1000
            fo = float(amb.get("fade_out_ms", 0) or 0) / 1000
            lvl = rms_db(window(sx, last + 0.5, max(last + 1.5, end_s - fo)))
            check("PASS" if lvl > -60 else "WARN", "ambience after the last word", f"{lvl:.0f} dBFS")
        for shot in manifest.get("one_shots", []):
            t = shot["start_s"]
            on, before = rms_db(window(sx, t, t + 1.0)), rms_db(window(sx, max(0, t - 1.2), max(0.01, t - 0.2)))
            check("PASS" if on > before + 3 or before < -80 else "WARN", f"one-shot {shot['id']} audible",
                  f"{on:.0f} dBFS at {fmt_time(t)} vs {before:.0f} dBFS just before")

    # Transcripts: cached ones for the current takes are always reported; --transcribe fills the rest.
    transcripts = {}
    sel = (read_json(root / "voice" / "selection.json", {}) or {}).get("segments", {})
    todo = []
    for seg in script["segments"]:
        sid = str(seg["id"])
        entry = sel.get(sid)
        meta = read_json(root / "voice" / "passages" / sid / f"{entry['take']}.json") if entry else None
        if not meta:
            continue
        cache = root / "qa" / "transcripts" / f"{sid}.json"
        cached = read_json(cache)
        if cached and cached.get("audio_sha256") == meta.get("audio_sha256") and not args.force:
            transcripts[sid] = cached
        else:
            todo.append((sid, meta, cache))
    if args.transcribe:
        secs = sum(float(m.get("duration_s") or 0) for _, m, _ in todo)
        print(f"transcribe: {len(todo)} passage(s), {secs:.0f}s of audio with {args.stt_model} (billed by audio length)")
        if todo and not args.dry_run:
            if any(m.get("mock") for _, m, _ in todo):
                print("  (mock passages: system TTS, transcript differences are expected)")
            key = api_key(root)
            before = usage_count(key)
            for sid, meta, cache in todo:
                path = root / "voice" / "passages" / sid / meta["audio_file"]
                try:
                    data = transcribe(key, path, args.stt_model, (script.get("language") or None))
                except ApiError as e:
                    fail_api(e, f"speech to text for segment {sid}")
                rec = {"segment_id": sid, "take": meta.get("take"), "audio_sha256": meta.get("audio_sha256"),
                       "model": args.stt_model, "text": data.get("text", ""), "created_at": now_iso()}
                write_json(cache, rec)
                transcripts[sid] = rec
                ledger(root, {"kind": "stt", "segment": sid, "seconds": meta.get("duration_s"), "model": args.stt_model})
            after = usage_count(key)
            if before is not None and after is not None:
                ledger(root, {"kind": "usage", "purpose": "stt", "before": before, "after": after, "delta": after - before})
    for seg in script["segments"]:
        sid = str(seg["id"])
        if sid not in transcripts:
            continue
        expected = norm_words(" ".join(strip_tags(p["text"]) for p in seg["phrases"]))
        heard = norm_words(transcripts[sid]["text"])
        wer, diffs = word_diff(expected, heard)
        transcripts[sid]["wer"] = round(wer, 3)
        transcripts[sid]["diffs"] = diffs
        check("PASS" if wer <= 0.05 else ("WARN" if wer <= 0.15 else "FAIL"), f"transcript {sid}",
              f"word error {wer:.0%}" + (f": {'; '.join(diffs[:4])}" if diffs else ""))
    if todo and not args.transcribe and transcripts:
        notes.append(f"{len(todo)} passage(s) changed since their transcript; rerun with --transcribe to recheck")

    # Credits from the ledger: exact header charges where the API sends them, estimates otherwise.
    charged, uncharged_est, est, calls = 0, 0.0, 0.0, 0
    ledger_path = root / "ledger.jsonl"
    if ledger_path.exists():
        for line in ledger_path.read_text(encoding="utf-8").splitlines():
            try:
                e = json.loads(line)
            except ValueError:
                continue
            if e.get("kind") in ("tts", "music", "sfx", "stt"):
                calls += 1
                est += float(e.get("estimated_credits") or 0)
                if e.get("credits") is not None:
                    charged += int(e["credits"])
                else:
                    uncharged_est += float(e.get("estimated_credits") or 0)
    mock = bool(manifest.get("mock") or timeline.get("mock"))
    status = "FAIL" if any(c["status"] == "FAIL" for c in checks) else (
        "WARN" if any(c["status"] == "WARN" for c in checks) else "PASS")
    report = {"session": str(root), "created_at": now_iso(), "status": status, "mock": mock,
              "duration_s": end_s, "checks": checks, "screening_notes": notes, "files": files,
              "transcripts": transcripts, "credits": {"calls": calls, "charged_from_headers": charged,
                                                     "estimated_without_header": round(uncharged_est),
                                                     "estimated_total": round(est)},
              "manifest_warnings": manifest.get("warnings", [])}
    write_json(root / "qa" / "report.json", report)

    title = script.get("title") or root.name
    md = [f"# QA: {title}", "", f"{now_iso()} · automated status **{status}**"
          + (" · **MOCK AUDIO: rehearsal only**" if mock else ""), "",
          "Automated checks screen for technical problems. They cannot approve accent, calm, phrasing or mix "
          "balance; the listening checklist below decides that.", "",
          "## Files", "", "| File | Duration | Ch | Integrated | True peak | LRA | Clipped |",
          "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for rel_path, f in files.items():
        md.append(f"| {rel_path} | {fmt_time(f.get('duration'))} | {f.get('channels')} | {f.get('I')} LUFS | "
                  f"{f.get('TP')} dBTP | {f.get('LRA')} LU | {f.get('clipped_samples')} |")
    md += ["", "Voice loudness per version: " + ", ".join(f"{v} {d.get('voice_lufs')} LUFS"
                                                          for v, d in manifest["variants"].items()), "",
           "## Checks", ""]
    md += [f"- **{c['status']}** {c['check']}: {c['detail']}" for c in checks]
    if manifest.get("warnings"):
        md += ["", "## Mix notes", ""] + [f"- {w}" for w in manifest["warnings"]]
    if notes:
        md += ["", "## Screening notes (listen to these spots)", ""] + [f"- {n}" for n in notes]
    md += ["", "## Credits", "", f"{calls} paid calls: {charged:,} credits charged (from response headers)"
           + (f" plus ~{uncharged_est:,.0f} estimated for calls without a cost header (music, transcription)"
              if uncharged_est else "")
           + f". Pre-call estimates totalled {est:,.0f}. The account balance (check_setup.py) is authoritative.",
           "", "## Listening checklist", ""]
    for section, items in CHECKLIST:
        md.append(f"**{section}**")
        md += [f"- [ ] {i}" for i in items]
        md.append("")
    (root / "qa" / "report.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    for c in checks:
        print(f"  {c['status']:<5} {c['check']}: {c['detail']}")
    print(f"QA {status}{' (mock audio)' if mock else ''} -> qa/report.md")
    raise SystemExit(1 if status == "FAIL" else 0)


if __name__ == "__main__":
    main()
