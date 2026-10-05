#!/usr/bin/env python3
# /// script
# requires-python = ">=3.9"
# dependencies = ["numpy"]
# ///
"""
GENERATE MUSIC
Compose the instrumental bed with the ElevenLabs Music API from the script's music prompt.
Candidates are saved as music/source-NN.mp3 with their request and response metadata; the newest
is selected unless another is chosen with --select. mix.py loops the selected source with long
crossfades to cover the whole session, so a few minutes of music is usually enough.

  python3 scripts/generate_music.py SESSION --dry-run
  python3 scripts/generate_music.py SESSION                        # one candidate, length from the script
  python3 scripts/generate_music.py SESSION --length-s 240 --candidates 2
  python3 scripts/generate_music.py SESSION --select source-02
  python3 scripts/generate_music.py SESSION --plan-only            # composition plan -> music/plan.json
  python3 scripts/generate_music.py SESSION --composition-plan music/plan.json
  python3 scripts/generate_music.py SESSION --mock                 # offline drone for pipeline rehearsal

Prompts that name artists, bands or songs are rejected (bad_prompt); the API's suggested prompt
is printed, and --accept-suggestion retries with it. The Music API needs a paid plan.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

from gm_common import (
    ApiError, MUSIC_CREDITS_PER_MINUTE, SR, api_json, api_key, api_request, check_budget, default_output_format,
    die, estimated_timeline, fail_api, get_subscription, header_credits, is_account_blocker, ledger, load_script,
    log_usage, now_iso, probe_duration, read_json, resolve_anchor, save_api_audio, session_root, sha256_file,
    write_json, write_wav_f32,
)


def needed_bed_ms(script, root):
    timeline = read_json(root / "voice" / "timeline.json") or estimated_timeline(script)
    cues = (script.get("music") or {}).get("cues") or []
    try:
        first = min(resolve_anchor(c["anchor"], timeline) for c in cues) if cues else 0.0
    except (KeyError, ValueError):
        first = 0.0
    return max(0.0, timeline["duration_ms"] - first)


def next_source(folder):
    n = 0
    for p in folder.glob("source-*.json"):
        m = re.fullmatch(r"source-(\d+)", p.stem)
        if m:
            n = max(n, int(m.group(1)))
    return f"source-{n + 1:02d}"


def mock_music(length_s, seed, sr=SR):
    """Slow drone (D major partials, drifting amplitudes, soft noise) with an intro and outro fade,
    so the looping code has something realistic to trim."""
    rng = np.random.default_rng(seed)
    n = int(length_s * sr)
    t = np.arange(n) / sr
    out = np.zeros((n, 2), dtype=np.float64)
    for f, a in ((73.42, 0.35), (110.0, 0.25), (146.83, 0.2), (185.0, 0.12), (220.0, 0.1), (293.66, 0.05)):
        for ch in range(2):
            lfo = 0.6 + 0.4 * np.sin(2 * np.pi * (0.013 + 0.01 * rng.random()) * t + rng.random() * 6.28)
            out[:, ch] += a * lfo * np.sin(2 * np.pi * f * (1 + 0.0007 * ch) * t + rng.random() * 6.28)
    k = 400
    for ch in range(2):
        cs = np.cumsum(np.concatenate((np.zeros(k), rng.standard_normal(n))))
        out[:, ch] += 0.4 * (cs[k:] - cs[:-k]) / k   # moving-average (dark) noise
    env = np.minimum(1.0, t / 6.0) * np.minimum(1.0, (length_s - t) / 8.0)
    out *= (env ** 2)[:, None]
    out *= 0.35 / max(1e-9, np.max(np.abs(out)))
    return out.astype(np.float32)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("session", help="session folder (or its script.json)")
    ap.add_argument("--length-s", type=float, help="length of each candidate (default: script music.generation.length_ms)")
    ap.add_argument("--candidates", type=int, default=1, help="how many to generate (default 1)")
    ap.add_argument("--model", help="music_v2_5 (default), music_v2, music_v1")
    ap.add_argument("--prompt", help="override the script's music.prompt for this run")
    ap.add_argument("--composition-plan", help="compose from a composition plan JSON instead of the prompt")
    ap.add_argument("--plan-only", action="store_true", help="ask the API for a composition plan and save it")
    ap.add_argument("--accept-suggestion", action="store_true", help="on bad_prompt, retry with the suggested prompt")
    ap.add_argument("--select", metavar="SOURCE", help="choose the source mix.py uses (e.g. source-02)")
    ap.add_argument("--keep-selection", action="store_true", help="add the candidate without selecting it (for example a loop for after the session, or one part of composed music)")
    ap.add_argument("--output-format", help="default mp3_44100_192 on Creator and above, else mp3_44100_128")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--max-credits", type=float)
    ap.add_argument("--mock", action="store_true", help="synthesize an offline drone instead (not for delivery)")
    args = ap.parse_args()

    root = session_root(args.session)
    script = load_script(root)
    music = script.get("music") or {}
    folder = root / "music"
    folder.mkdir(exist_ok=True)

    if args.select:
        if not (folder / f"{args.select}.json").exists():
            die(f"music/{args.select}.json not found")
        write_json(folder / "selection.json", {"selected": args.select, "updated_at": now_iso()})
        print(f"music source: {args.select}")
        return
    if not music.get("enabled"):
        die("music is disabled in script.json (music.enabled)")
    if music.get("source_file"):
        die(f"script uses music.source_file ({music['source_file']}); nothing to generate")

    gen = music.get("generation") or {}
    prompt = args.prompt or music.get("prompt") or ""
    model = args.model or gen.get("model_id") or "music_v2_5"
    length_ms = int(args.length_s * 1000) if args.length_s else int(gen.get("length_ms") or 180000)
    need = needed_bed_ms(script, root)
    if need and length_ms > need + 10000:
        length_ms = int(need + 10000)
        print(f"the bed only needs {need / 1000:.0f}s; generating {length_ms / 1000:.0f}s")
    length_ms = max(10000, min(600000, length_ms))
    plan = None
    if args.composition_plan:
        plan = read_json(Path(args.composition_plan) if Path(args.composition_plan).is_absolute()
                         else (root / args.composition_plan if (root / args.composition_plan).exists()
                               else Path(args.composition_plan)))
        if plan is None:
            die(f"composition plan {args.composition_plan} not found")
        # A plan sets its own length: the sum of its sections (chunks for music_v2, sections for music_v1).
        length_ms = int(sum(float(c.get("duration_ms") or 0) for c in plan.get("chunks") or plan.get("sections") or [])
                        or length_ms)
    est = length_ms / 60000.0 * MUSIC_CREDITS_PER_MINUTE * args.candidates
    print(f"{'mock ' if args.mock else ''}music: {model}, {length_ms / 1000:.0f}s x {args.candidates} candidate(s)"
          f"{'' if args.mock else f' ~ {est:,.0f} credits (approximate)'}; bed needed ~{need / 1000:.0f}s "
          f"(mix.py loops it)")
    print(f"prompt: {prompt}" if plan is None else f"composition plan: {args.composition_plan}")
    if args.dry_run:
        return

    if args.mock:
        for k in range(args.candidates):
            name = next_source(folder)
            path = folder / f"{name}.wav"
            write_wav_f32(path, mock_music(length_ms / 1000, k + 1))
            meta = {"source": name, "created_at": now_iso(), "mock": True, "model_id": "mock", "prompt": prompt,
                    "length_ms": length_ms, "audio_file": path.name, "audio_sha256": sha256_file(path),
                    "duration_s": round(probe_duration(path) or 0, 2)}
            write_json(folder / f"{name}.json", meta)
            if not args.keep_selection:
                write_json(folder / "selection.json", {"selected": name, "updated_at": now_iso()})
            print(f"  {name}: mock drone {meta['duration_s']:.0f}s -> music/{path.name}")
        return

    key = api_key(root)
    if args.plan_only:
        try:
            data, _ = api_json("POST", "/v1/music/plan", key=key,
                               body={"prompt": prompt, "music_length_ms": length_ms, "model_id": model})
        except ApiError as e:
            fail_api(e, "composition plan")
        write_json(folder / "plan.json", data)
        print(f"wrote music/plan.json; edit it, then: generate_music.py {root} --composition-plan music/plan.json")
        return

    fmt = args.output_format
    if not fmt:
        try:
            fmt = default_output_format(get_subscription(key).get("tier"))
        except ApiError as e:
            if is_account_blocker(e):
                fail_api(e, "account check")
            fmt = "mp3_44100_128"
    before = check_budget(est, args.max_credits, key, "music generation")
    charged = 0
    for k in range(args.candidates):
        body = {"model_id": model}
        if plan is not None:
            body["composition_plan"] = plan
        else:
            body.update({"prompt": prompt, "music_length_ms": length_ms, "force_instrumental": True})
        for attempt in range(2):
            try:
                audio, headers = api_request("POST", "/v1/music", key=key, body=body, query={"output_format": fmt},
                                             accept="audio/*", timeout=900, retries=1)
                break
            except ApiError as e:
                suggestion = e.find("prompt_suggestion") or e.find("composition_plan_suggestion")
                if suggestion and attempt == 0:
                    print(f"music prompt rejected: {e.args[0]}\nsuggested: {json.dumps(suggestion)[:600]}")
                    if args.accept_suggestion:
                        if isinstance(suggestion, str):
                            body["prompt"] = prompt = suggestion
                        else:
                            body.pop("prompt", None)
                            body["composition_plan"] = suggestion
                        continue
                    die("update music.prompt in script.json (or rerun with --accept-suggestion)", 2)
                fail_api(e, "music generation")
        name = next_source(folder)
        path = save_api_audio(audio, fmt, folder / name)
        meta = {"source": name, "created_at": now_iso(), "model_id": model, "prompt": body.get("prompt"),
                "composition_plan": body.get("composition_plan"), "length_ms": length_ms, "output_format": fmt,
                "headers": {h: v for h, v in headers.items() if "id" in h or "cost" in h or "character" in h},
                "audio_file": path.name, "audio_sha256": sha256_file(path),
                "duration_s": round(probe_duration(path) or 0, 2), "mock": False}
        write_json(folder / f"{name}.json", meta)
        if not args.keep_selection:
            write_json(folder / "selection.json", {"selected": name, "updated_at": now_iso()})
        cost = header_credits(headers)
        charged = None if cost is None or charged is None else charged + cost
        ledger(root, {"kind": "music", "source": name, "model": model, "seconds": length_ms / 1000,
                      "estimated_credits": length_ms / 60000.0 * MUSIC_CREDITS_PER_MINUTE, "credits": cost,
                      "song_id": headers.get("song-id"), "file": f"music/{path.name}"})
        print(f"  {name}: {meta['duration_s']:.0f}s -> music/{path.name}")
    log_usage(root, key, before, "music", charged, est)
    print("Listen for vocals, beats, sudden swells or bright events before mixing; select another candidate with "
          "--select if needed.")


if __name__ == "__main__":
    main()
