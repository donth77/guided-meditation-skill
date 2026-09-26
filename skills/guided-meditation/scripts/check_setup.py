#!/usr/bin/env python3
# /// script
# requires-python = ">=3.9"
# dependencies = ["numpy"]
# ///
"""
CHECK SETUP
Verify the tools and the ElevenLabs account before a session: Python and numpy, ffmpeg/ffprobe
with the filters and encoders the mix uses, the API key, the subscription (tier, credits left,
billing status), the output formats the tier allows, and the available TTS models.

  python3 scripts/check_setup.py
  python3 scripts/check_setup.py --session meditations/rain      # also looks for .env above the session
  python3 scripts/check_setup.py --probe                         # + one tiny TTS request (~20 credits)
  python3 scripts/check_setup.py --json

Exit 0 when ready, 1 when a tool or the key is missing, 2 when the account cannot generate
(billing, quota or key problem; a past-due subscription counts, since ElevenLabs refuses
generation until the invoice is paid).
"""
from __future__ import annotations

import argparse
import json
import platform
import shutil
import subprocess
import sys
from datetime import datetime

import numpy as np

from gm_common import (
    ApiError, api_json, api_key, api_request, credits_remaining, explain_api_error, get_models, load_dotenv,
    tier_level,
)

FILTERS = ("ebur128", "amerge", "pan", "acompressor", "equalizer", "highpass", "highshelf", "volume")
ENCODERS = ("libmp3lame", "aac", "pcm_s24le", "flac")


def ffmpeg_caps():
    out = {"ffmpeg": shutil.which("ffmpeg"), "ffprobe": shutil.which("ffprobe")}
    if not out["ffmpeg"]:
        return out
    v = subprocess.run(["ffmpeg", "-hide_banner", "-version"], capture_output=True, text=True).stdout.splitlines()
    out["version"] = v[0] if v else "?"
    f = subprocess.run(["ffmpeg", "-hide_banner", "-filters"], capture_output=True, text=True).stdout
    e = subprocess.run(["ffmpeg", "-hide_banner", "-encoders"], capture_output=True, text=True).stdout
    out["missing_filters"] = [x for x in FILTERS if f" {x} " not in f]
    out["missing_encoders"] = [x for x in ENCODERS if f" {x} " not in e]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--session", help="session folder (to find a .env above it)")
    ap.add_argument("--probe", action="store_true", help="make one tiny TTS request to prove generation works")
    ap.add_argument("--voice-id", help="voice for --probe (default: first voice in the account)")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    res = {"python": platform.python_version(), "numpy": np.__version__, "problems": [], "warnings": []}
    if sys.version_info < (3, 9):
        res["problems"].append("Python 3.9 or newer is required")
    caps = ffmpeg_caps()
    res["ffmpeg"] = caps
    if not caps.get("ffmpeg") or not caps.get("ffprobe"):
        res["problems"].append("ffmpeg and ffprobe must be on PATH (brew install ffmpeg / apt install ffmpeg)")
    else:
        if caps["missing_filters"]:
            res["problems"].append(f"ffmpeg lacks filters: {', '.join(caps['missing_filters'])}")
        if caps["missing_encoders"]:
            res["warnings"].append(f"ffmpeg lacks encoders: {', '.join(caps['missing_encoders'])} "
                                   "(those output formats will fail)")
    res["mock_voice"] = shutil.which("say") or shutil.which("espeak-ng") or shutil.which("espeak") or "tone bursts"
    envs = load_dotenv(args.session)
    res["env_files"] = [str(p) for p in envs]
    key = api_key(args.session, required=False)
    res["api_key"] = bool(key)
    code = 0
    if not key:
        res["problems"].append("ELEVENLABS_API_KEY not found (environment or a .env above the session/working dir)")
    else:
        try:
            sub, _ = api_json("GET", "/v1/user/subscription", key=key, retries=1)
            reset = sub.get("next_character_count_reset_unix")
            res["subscription"] = {
                "tier": sub.get("tier"), "status": sub.get("status"),
                "credits_used": sub.get("character_count"), "credit_limit": sub.get("character_limit"),
                "credits_left": credits_remaining(sub),
                "resets": datetime.fromtimestamp(reset).isoformat(timespec="minutes") if reset else None,
                "voice_slots": f"{sub.get('voice_slots_used')}/{sub.get('voice_limit')}",
                "open_invoices": bool(sub.get("has_open_invoices")),
            }
            lvl = tier_level(sub.get("tier"))
            res["output_formats"] = {"mp3_44100_192": lvl >= 2, "pcm_44100 / wav_44100": lvl >= 3,
                                     "mp3_44100_128": True}
            res["default_output_format"] = "mp3_44100_192" if lvl >= 2 else "mp3_44100_128"
            if lvl < 1:
                res["warnings"].append("the Music API needs a paid plan")
            status = (sub.get("status") or "").lower()
            if status in ("past_due", "unpaid", "incomplete", "incomplete_expired") or sub.get("has_open_invoices"):
                res["problems"].append(f"subscription status '{status}' with an open invoice: ElevenLabs refuses "
                                       "generation (payment_required) until it is paid")
                code = 2
            models = get_models(key)
            res["tts_models"] = [
                {"model_id": m, "max_chars": d.get("maximum_text_length_per_request"),
                 "credits_per_char": (d.get("model_rates") or {}).get("character_cost_multiplier"),
                 "style": d.get("can_use_style"), "speaker_boost": d.get("can_use_speaker_boost")}
                for m, d in models.items() if d.get("can_do_text_to_speech")]
        except ApiError as e:
            res["problems"].append(f"subscription check failed: {e}" + (f" -- {explain_api_error(e)}"
                                                                         if explain_api_error(e) else ""))
            code = 2
        if args.probe and key:
            try:
                vid = args.voice_id
                if not vid:
                    data, _ = api_json("GET", "/v2/voices", key=key, query={"page_size": 1})
                    vid = (data.get("voices") or [{}])[0].get("voice_id")
                raw, hdrs = api_request("POST", f"/v1/text-to-speech/{vid}", key=key, accept="audio/*",
                                        body={"text": "A short test.", "model_id": "eleven_flash_v2_5"},
                                        query={"output_format": "mp3_44100_128"}, retries=0)
                res["probe"] = {"ok": True, "voice_id": vid, "bytes": len(raw), "request_id": hdrs.get("request-id")}
                if code == 2 and all("subscription status" in p for p in res["problems"]):
                    res["problems"] = [p for p in res["problems"] if "subscription status" not in p]
                    res["warnings"].append("subscription is past due but generation still works for now")
                    code = 0
            except ApiError as e:
                res["probe"] = {"ok": False, "error": str(e), "explanation": explain_api_error(e)}
                res["problems"].append(f"generation probe failed: {e}")
                code = 2
    if res["problems"] and code == 0:
        code = 1
    if args.json:
        print(json.dumps(res, indent=2))
        raise SystemExit(code)

    print(f"Python {res['python']}, numpy {res['numpy']}")
    print(f"ffmpeg: {caps.get('version', 'missing')}")
    print(f"mock voice: {res['mock_voice']}")
    print(f"API key: {'found' if res['api_key'] else 'MISSING'}" + (f" ({', '.join(res['env_files'])})" if envs else ""))
    s = res.get("subscription")
    if s:
        print(f"account: {s['tier']} ({s['status']}), {s['credits_left']:,} of {s['credit_limit']:,} credits left, "
              f"resets {s['resets']}, voice slots {s['voice_slots']}")
        print(f"default output format: {res['default_output_format']} "
              f"(44.1 kHz PCM {'available' if res['output_formats']['pcm_44100 / wav_44100'] else 'needs Pro'})")
        for m in res.get("tts_models", []):
            print(f"  {m['model_id']:<26} {m['credits_per_char']} credit/char, max {m['max_chars']} chars")
    if "probe" in res:
        p = res["probe"]
        print(f"probe: {'ok' if p['ok'] else 'FAILED'} {p.get('request_id') or p.get('error', '')}")
    for w in res["warnings"]:
        print(f"warning: {w}")
    for p in res["problems"]:
        print(f"PROBLEM: {p}")
    print("ready" if code == 0 else ("account cannot generate" if code == 2 else "not ready"))
    raise SystemExit(code)


if __name__ == "__main__":
    main()
