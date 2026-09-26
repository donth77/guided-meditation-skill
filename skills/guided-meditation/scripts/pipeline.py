#!/usr/bin/env python3
# /// script
# requires-python = ">=3.9"
# dependencies = ["numpy"]
# ///
"""
PIPELINE
Run the remaining phases in order for a session whose script (and, for a real voice, voice.json)
is ready: validate, render, synthesize passages without a take, assemble, generate the music and
SFX the requested versions need (only what is missing), mix, QA. Finished steps are skipped.

  python3 scripts/pipeline.py SESSION --outputs voice,voice+music+sfx     # plan + credit estimate only
  python3 scripts/pipeline.py SESSION --outputs all --yes                 # spend credits and run
  python3 scripts/pipeline.py SESSION --outputs all --mock                # offline rehearsal, no API calls
  python3 scripts/pipeline.py SESSION --yes --max-credits 8000 --formats wav,mp3,m4a --transcribe

Without --yes (or --mock) nothing paid runs: the plan and estimate are printed and the exit code
is 3. The listening gates in SKILL.md still apply: the voice is auditioned and accepted before a
full run, and the finished versions are listened to before they are called done.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from gm_common import (
    MUSIC_CREDITS_PER_MINUTE, SFX_CREDITS_PER_SECOND, die, estimated_timeline, fmt_time, is_v3, layers_for,
    load_script, parse_outputs, read_json, segment_request, session_root, tts_rate,
)

HERE = Path(__file__).resolve().parent


def run(script, *argv):
    cmd = [sys.executable, str(HERE / script), *[str(a) for a in argv]]
    print(f"\n$ {script} {' '.join(str(a) for a in argv[1:])}", flush=True)
    code = subprocess.call(cmd)
    if code not in (0,):
        raise SystemExit(code)


def missing_work(root, script, variants, mock):
    need = layers_for(variants)
    voice = read_json(root / "voice.json", {}) or {}
    model = voice.get("model_id") or "eleven_multilingual_v2"
    sel = (read_json(root / "voice" / "selection.json", {}) or {}).get("segments", {})
    segs = [s for s in script["segments"] if str(s["id"]) not in sel]
    chars = sum(len(segment_request(s, is_v3(model))[0]) for s in segs)
    work = {"segments": [str(s["id"]) for s in segs], "tts_chars": chars, "model": model,
            "tts_credits": 0 if mock else chars * tts_rate(model), "music": False, "music_credits": 0,
            "ambience": False, "one_shots": [], "sfx_credits": 0}
    music = script.get("music") or {}
    if "music" in need and not music.get("source_file") and not (read_json(root / "music" / "selection.json") or {}).get("selected"):
        work["music"] = True
        length = (music.get("generation") or {}).get("length_ms", 180000)
        need_ms = estimated_timeline(script)["duration_ms"]
        work["music_credits"] = 0 if mock else min(length, need_ms + 10000) / 60000 * MUSIC_CREDITS_PER_MINUTE
    if "sfx" in need:
        sfx = script.get("sfx") or {}
        ssel = read_json(root / "sfx" / "selection.json", {}) or {}
        amb = sfx.get("ambience") or {}
        if amb.get("enabled") and not amb.get("source_file") and not ssel.get("ambience"):
            work["ambience"] = True
            work["sfx_credits"] += (amb.get("generation") or {}).get("duration_seconds", 30) * SFX_CREDITS_PER_SECOND
        for s in sfx.get("one_shots") or []:
            if s.get("prompt") and not s.get("same_as") and not s.get("source_file") and \
                    not (ssel.get("one_shots") or {}).get(s["id"]):
                work["one_shots"].append(s["id"])
                work["sfx_credits"] += s.get("duration_seconds", 5) * SFX_CREDITS_PER_SECOND
        if mock:
            work["sfx_credits"] = 0
    work["total"] = work["tts_credits"] + work["music_credits"] + work["sfx_credits"]
    return work


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("session", help="session folder (or its script.json)")
    ap.add_argument("--outputs", help="voice, voice+music, voice+sfx, voice+music+sfx or all (voice is always made)")
    ap.add_argument("--yes", action="store_true", help="allow paid API calls")
    ap.add_argument("--mock", action="store_true", help="offline stand-ins for voice, music and SFX (rehearsal)")
    ap.add_argument("--max-credits", type=float, help="refuse to start above this estimate")
    ap.add_argument("--formats", default="wav,mp3")
    ap.add_argument("--lufs", type=float, default=-18.0)
    ap.add_argument("--normalize", choices=("voice", "integrated"), default="voice")
    ap.add_argument("--limit", action="store_true", help="limit peaks instead of lowering the gain (mix.py --limit)")
    ap.add_argument("--transcribe", action="store_true", help="speech-to-text check in QA (paid)")
    args = ap.parse_args()

    root = session_root(args.session)
    run("validate_script.py", root, "--write")
    run("render_script.py", root)
    script = load_script(root)
    variants = parse_outputs(args.outputs, script)
    work = missing_work(root, script, variants, args.mock)
    print(f"\nversions: {', '.join(variants)}")
    print(f"to generate: {len(work['segments'])} passage(s) ({work['tts_chars']:,} chars, {work['model']})"
          + (", music" if work["music"] else "") + (", ambience" if work["ambience"] else "")
          + (f", one-shots {','.join(work['one_shots'])}" if work["one_shots"] else ""))
    if not args.mock:
        print(f"estimated credits: {work['total']:,.0f} (narration {work['tts_credits']:,.0f}, music "
              f"{work['music_credits']:,.0f}, sfx {work['sfx_credits']:,.0f})")
    if args.max_credits is not None and work["total"] > args.max_credits:
        die(f"estimate {work['total']:,.0f} is above --max-credits {args.max_credits:,.0f}")
    paid = not args.mock and (work["segments"] or work["music"] or work["ambience"] or work["one_shots"])
    if paid and not args.yes:
        print("\nnothing paid was run: rerun with --yes to generate (or --mock to rehearse offline)")
        raise SystemExit(3)
    if work["segments"] and not args.mock and not (root / "voice.json").exists():
        die("no voice.json: audition the opening and accept a take first (synthesize.py --audition, --accept)")

    budget = ["--max-credits", args.max_credits] if args.max_credits is not None else []
    mock = ["--mock"] if args.mock else []
    if work["segments"]:
        run("synthesize.py", root, *mock, *budget)
    run("assemble_voice.py", root)
    if work["music"]:
        run("generate_music.py", root, *mock, *budget)
    if work["ambience"] or work["one_shots"]:
        run("generate_sfx.py", root, *mock, *budget)
    run("mix.py", root, "--outputs", ",".join(variants), "--formats", args.formats, "--lufs", args.lufs,
        "--normalize", args.normalize, *(["--limit"] if args.limit else []))
    run("render_script.py", root)
    qa = [sys.executable, str(HERE / "qa_report.py"), str(root)] + (["--transcribe"] if args.transcribe and not args.mock else [])
    print("\n$ qa_report.py", flush=True)
    code = subprocess.call(qa)
    manifest = read_json(root / "output" / "manifest.json") or {}
    print(f"\n{manifest.get('title', root.name)}: {fmt_time(manifest.get('duration_s'))}")
    for v, d in (manifest.get("variants") or {}).items():
        print(f"  {v:<17} {', '.join(d['files'])}")
    print("QA report: qa/report.md -- listen with the checklist before calling it done.")
    raise SystemExit(code)


if __name__ == "__main__":
    main()
