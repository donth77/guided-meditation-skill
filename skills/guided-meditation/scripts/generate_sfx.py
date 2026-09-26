#!/usr/bin/env python3
# /// script
# requires-python = ">=3.9"
# dependencies = ["numpy"]
# ///
"""
GENERATE SFX
Generate the ambience bed and the one-shot cues with the ElevenLabs Sound Effects API from the
script's sfx block. The ambience is requested as a seamless loop (loop: true, up to 30 s) that
mix.py repeats for the whole session; one-shots (a bell, a chime) are placed at their anchors.

  python3 scripts/generate_sfx.py SESSION --dry-run
  python3 scripts/generate_sfx.py SESSION                        # ambience + one-shots without a take
  python3 scripts/generate_sfx.py SESSION --ambience --candidates 2
  python3 scripts/generate_sfx.py SESSION --one-shots S01 --retake
  python3 scripts/generate_sfx.py SESSION --select ambience=ambience-01+ambience-02,S01=S01-02
  python3 scripts/generate_sfx.py SESSION --mock                 # offline rain and bell (not for delivery)

Several selected ambience takes are alternated by mix.py, which makes the repetition of a short
loop less noticeable. A one-shot with "same_as" reuses another one-shot's audio (no generation).
Cost: 40 credits per requested second.
"""
from __future__ import annotations

import argparse
import re

import numpy as np

from gm_common import (
    ApiError, SFX_CREDITS_PER_SECOND, SR, api_key, api_request, check_budget, default_output_format, die,
    fail_api, get_subscription, header_credits, is_account_blocker, ledger, load_script, log_usage, now_iso,
    probe_duration, read_json, save_api_audio, session_root, sha256_file, write_json, write_wav_f32,
)


def selection(root):
    sel = read_json(root / "sfx" / "selection.json", {}) or {}
    sel.setdefault("ambience", [])
    sel.setdefault("one_shots", {})
    return sel


def next_name(folder, prefix):
    n = 0
    if folder.exists():
        for p in folder.glob(f"{prefix}-*.json"):
            m = re.fullmatch(re.escape(prefix) + r"-(\d+)", p.stem)
            if m:
                n = max(n, int(m.group(1)))
    return f"{prefix}-{n + 1:02d}"


# ----------------------------------------------------------------------------- mock audio

def _smooth(x, k):
    cs = np.cumsum(np.concatenate((np.zeros(k), x)))
    return (cs[k:] - cs[:-k]) / k


def mock_rain(seconds, seed, sr=SR):
    """Noise bed with droplets, made seamless by folding a crossfaded tail into its start."""
    rng = np.random.default_rng(seed)
    n, xf = int(seconds * sr), int(1.5 * sr)
    out = np.zeros((n + xf, 2))
    for ch in range(2):
        w = rng.standard_normal(n + xf)
        bed = 0.5 * (_smooth(w, 3) - 0.6 * _smooth(w, 40))
        drops = np.zeros(n + xf)
        idx = rng.integers(0, n + xf - 2000, size=int(seconds * 25))
        drops[idx] = rng.uniform(0.2, 1.0, len(idx))
        t = np.arange(900) / sr
        ping = np.sin(2 * np.pi * rng.uniform(1800, 3200) * t) * np.exp(-t / 0.006)
        out[:, ch] = bed + 0.35 * np.convolve(drops, ping)[: n + xf]
    fin = np.sin(np.linspace(0, np.pi / 2, xf))[:, None]
    out[:xf] = out[:xf] * fin + out[n:] * np.cos(np.linspace(0, np.pi / 2, xf))[:, None]
    out = out[:n]
    return (0.3 * out / np.max(np.abs(out))).astype(np.float32)


def mock_bell(seconds, sr=SR):
    t = np.arange(int(seconds * sr)) / sr
    x = np.zeros_like(t)
    for ratio, amp, decay in ((1.0, 1.0, 3.5), (2.76, 0.45, 2.2), (5.40, 0.22, 1.2), (8.93, 0.1, 0.6)):
        f = 196.0 * ratio
        x += amp * np.exp(-t / decay) * (np.sin(2 * np.pi * f * t) + 0.5 * np.sin(2 * np.pi * (f + 0.8) * t))
    x *= np.minimum(1.0, t / 0.004)
    x = 0.5 * x / np.max(np.abs(x))
    return np.stack([x, x], axis=1).astype(np.float32)


# ----------------------------------------------------------------------------- main

def generate(key, text, seconds, influence, loop, fmt):
    body = {"text": text, "duration_seconds": float(seconds), "prompt_influence": float(influence),
            "model_id": "eleven_text_to_sound_v2"}
    if loop:
        body["loop"] = True
    audio, headers = api_request("POST", "/v1/sound-generation", key=key, body=body, query={"output_format": fmt},
                                 accept="audio/*", timeout=300, retries=2)
    return audio, headers, body


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("session", help="session folder (or its script.json)")
    ap.add_argument("--ambience", action="store_true", help="generate the ambience loop")
    ap.add_argument("--one-shots", nargs="?", const="all", metavar="IDS", help="generate one-shots (all or a comma list)")
    ap.add_argument("--candidates", type=int, default=1, help="takes per sound (default 1)")
    ap.add_argument("--retake", action="store_true", help="generate even if a take is already selected")
    ap.add_argument("--select", metavar="SPEC", help="ambience=ambience-01+ambience-02,S01=S01-02")
    ap.add_argument("--output-format", help="default mp3_44100_192 on Creator and above, else mp3_44100_128")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--max-credits", type=float)
    ap.add_argument("--mock", action="store_true", help="synthesized rain and bell instead (not for delivery)")
    args = ap.parse_args()

    root = session_root(args.session)
    script = load_script(root)
    sfx = script.get("sfx") or {}
    sel = selection(root)
    folder = root / "sfx"

    if args.select:
        for item in args.select.split(","):
            k, _, v = item.partition("=")
            k, v = k.strip(), v.strip()
            if k == "ambience":
                names = [n for n in v.split("+") if n]
                for n in names:
                    if not (folder / f"{n}.json").exists():
                        die(f"sfx/{n}.json not found")
                sel["ambience"] = names
            else:
                if not (folder / "one-shots" / f"{v}.json").exists():
                    die(f"sfx/one-shots/{v}.json not found")
                sel["one_shots"][k] = v
        write_json(folder / "selection.json", sel)
        print(f"sfx selection: ambience {'+'.join(sel['ambience']) or '-'}; one-shots "
              + ", ".join(f"{k}={v}" for k, v in sel["one_shots"].items()))
        return
    if not sfx.get("enabled"):
        die("sfx is disabled in script.json (sfx.enabled)")

    both = not args.ambience and not args.one_shots
    jobs = []   # (kind, id, prompt, seconds, influence, loop)
    amb = sfx.get("ambience") or {}
    if (args.ambience or both) and amb.get("enabled") and not amb.get("source_file"):
        if args.retake or args.ambience or not sel["ambience"]:
            g = amb.get("generation") or {}
            jobs.append(("ambience", "ambience", amb["prompt"], g.get("duration_seconds", 30),
                         g.get("prompt_influence", 0.3), g.get("loop", True)))
    if args.one_shots or both:
        wanted = None if args.one_shots in (None, "all") else {s.strip() for s in args.one_shots.split(",")}
        for s in sfx.get("one_shots") or []:
            sid = s["id"]
            if wanted is not None and sid not in wanted:
                continue
            if s.get("same_as") or s.get("source_file") or not s.get("prompt"):
                continue
            if sel["one_shots"].get(sid) and not (args.retake or wanted is not None):
                continue
            jobs.append(("one_shot", sid, s["prompt"], s.get("duration_seconds", 5), s.get("prompt_influence", 0.5), False))
    if not jobs:
        print("nothing to generate (every sound has a selected take; --retake to replace)")
        return
    est = sum(j[3] for j in jobs) * SFX_CREDITS_PER_SECOND * args.candidates
    print(f"{'mock ' if args.mock else ''}sfx: {len(jobs)} sound(s) x {args.candidates}"
          + ("" if args.mock else f" ~ {est:,.0f} credits"))
    for kind, sid, prompt, secs, infl, loop in jobs:
        print(f"  {sid}: {secs:g}s, influence {infl}, {'loop, ' if loop else ''}\"{prompt}\"")
    if args.dry_run:
        return

    key = fmt = before = None
    if not args.mock:
        key = api_key(root)
        fmt = args.output_format
        if not fmt:
            try:
                fmt = default_output_format(get_subscription(key).get("tier"))
            except ApiError as e:
                if is_account_blocker(e):
                    fail_api(e, "account check")
                fmt = "mp3_44100_128"
        before = check_budget(est, args.max_credits, key, "sound effects")
    charged = 0
    for kind, sid, prompt, secs, infl, loop in jobs:
        out_dir = folder if kind == "ambience" else folder / "one-shots"
        out_dir.mkdir(parents=True, exist_ok=True)
        for k in range(args.candidates):
            name = next_name(out_dir, sid)
            if args.mock:
                audio = mock_rain(secs, k + 1) if kind == "ambience" else mock_bell(secs)
                path = out_dir / f"{name}.wav"
                write_wav_f32(path, audio)
                headers, body = {}, {"text": prompt, "duration_seconds": secs, "loop": loop}
            else:
                try:
                    audio, headers, body = generate(key, prompt, secs, infl, loop, fmt)
                except ApiError as e:
                    fail_api(e, f"sound generation for {sid}")
                path = save_api_audio(audio, fmt, out_dir / name)
                cost = header_credits(headers)
                charged = None if cost is None or charged is None else charged + cost
                ledger(root, {"kind": "sfx", "sound": sid, "take": name, "seconds": secs, "credits": cost,
                              "estimated_credits": secs * SFX_CREDITS_PER_SECOND, "file": str(path.relative_to(root))})
            meta = {"sound": sid, "take": name, "kind": kind, "created_at": now_iso(), "request": body,
                    "output_format": fmt or "wav", "loop": bool(loop), "audio_file": path.name,
                    "audio_sha256": sha256_file(path), "duration_s": round(probe_duration(path) or 0, 3),
                    "headers": {h: v for h, v in headers.items() if "id" in h or "cost" in h}, "mock": args.mock}
            write_json(out_dir / f"{name}.json", meta)
            if kind == "ambience":
                if k == 0:
                    sel["ambience"] = []
                sel["ambience"].append(name)
            else:
                sel["one_shots"][sid] = name
            write_json(folder / "selection.json", sel)
            print(f"  {sid} -> {path.relative_to(root)} ({meta['duration_s']:.1f}s)")
    if not args.mock:
        log_usage(root, key, before, "sfx", charged, est)
    print("Listen for distinctive events that will repeat audibly in the loop, and for clipped starts or ends.")


if __name__ == "__main__":
    main()
