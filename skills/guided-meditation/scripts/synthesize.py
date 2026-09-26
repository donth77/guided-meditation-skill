#!/usr/bin/env python3
# /// script
# requires-python = ">=3.9"
# dependencies = ["numpy"]
# ///
"""
SYNTHESIZE
Generate narration with ElevenLabs text to speech. Auditions come first (the script's opening,
a few takes, one recipe each), then one connected request per segment ("passage"), each saved
as a numbered take with its recipe, request id and character alignment. Takes are never
overwritten: a retake adds a take and selects it, and frozen segments are never regenerated.

  python3 scripts/synthesize.py SESSION --audition --voice-id ID --takes 2
  python3 scripts/synthesize.py SESSION --audition --voice-id ID1,ID2,ID3   # compare voices, same text
  python3 scripts/synthesize.py SESSION --audition --voice-id ID --model eleven_v3 --stability 0.5
  python3 scripts/synthesize.py SESSION --convert 0012 --voice-id ID   # re-voice a guide take (speech to speech)
  python3 scripts/synthesize.py SESSION --preview 0003,0005     # takes with the script's phrase rests (free)
  python3 scripts/synthesize.py SESSION --accept take-0003      # recipe -> voice.json (+ reuse as segment)
  python3 scripts/synthesize.py SESSION --dry-run               # requests, context and credits; no calls
  python3 scripts/synthesize.py SESSION                         # every segment without a take
  python3 scripts/synthesize.py SESSION --takes 3 --pick         # best of 3 per passage (breaks, then consistency)
  python3 scripts/synthesize.py SESSION --segments 03,05 --retake
  python3 scripts/synthesize.py SESSION --list | --freeze 01,02 | --unfreeze 03 | --select 04=take-01
  python3 scripts/synthesize.py SESSION --mock                  # offline stand-in voice (system TTS)

Recipe: voice.json (voice_id, model_id, voice_settings, seed, output_format), overridden by flags.
Context: for models that support request stitching, previous/next request ids of neighbouring
selected takes younger than two hours; otherwise previous_text/next_text; eleven_v3 accepts
neither, so its passages are generated without context. Audio tags are sent
only to eleven_v3 models and stripped for every other model. Pauses are never put in the text.
"""
from __future__ import annotations

import argparse
import base64
import json
import random
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from gm_common import (
    ApiError, SR, api_json, api_key, api_request, check_budget, count_words, decode_audio, default_output_format,
    die, fail_api, format_pauses, get_models, get_subscription, header_credits, inner_pauses, is_account_blocker,
    is_v3, ledger,
    load_script, log_usage, multipart, now_iso, now_unix, probe_duration, read_json, REQUEST_ID_MAX_AGE_S,
    save_api_audio, segment_request, session_root, sha256_file, speech_wpm, STS_CREDITS_PER_MINUTE, strip_tags,
    trim_digital_silence, tts_rate, voicing_ratio, warn, write_json, write_mp3, write_wav_f32,
)

DEFAULT_SETTINGS = {"stability": 0.5, "similarity_boost": 0.75, "style": 0.0, "use_speaker_boost": True, "speed": 1.0}
STS_MODEL = "eleven_multilingual_sts_v2"
PACE_SLOW, PACE_FAST = 90, 165   # words/min over the voiced span; calm narration mostly sits at 110-150
INTERESTING_HEADERS = ("request-id", "history-item-id", "character-cost", "x-character-count", "tts-latency-ms")


# ----------------------------------------------------------------------------- recipe

def voice_details(key, voice_id):
    """The account's record of a voice, or {} when the id is not in the account."""
    try:
        data, _ = api_json("GET", f"/v1/voices/{voice_id}", key=key, retries=1)
        return data
    except ApiError as e:
        if is_account_blocker(e):
            fail_api(e, "voice lookup")
        if "not_found" not in e.text and "not found" not in e.text:
            warn(f"could not read voice {voice_id}: {e}")
        return {}


def library_voice(key, voice_id):
    """Voice Library entry for an id (library voices work in TTS by id without being added)."""
    try:
        data, _ = api_json("GET", "/v1/shared-voices", key=key, query={"search": voice_id, "page_size": 5}, retries=1)
    except ApiError:
        return None
    return next((v for v in data.get("voices") or [] if v.get("voice_id") == voice_id), None)


def fine_tuned_models(v):
    state = ((v.get("fine_tuning") or {}).get("state")) or {}
    models = {m for m, s in state.items() if s == "fine_tuned"}
    models.update(v.get("high_quality_base_model_ids") or [])
    return models


def resolve_recipe(root, args, key, mock=False):
    base = read_json(root / "voice.json", {}) or {}
    r = {
        "voice_id": args.voice_id or base.get("voice_id"),
        "voice_name": base.get("voice_name") if not args.voice_id or args.voice_id == base.get("voice_id") else None,
        "model_id": args.model or base.get("model_id") or "eleven_multilingual_v2",
        "voice_settings": dict(base.get("voice_settings") or {}),
        "seed": args.seed if args.seed is not None else base.get("seed"),
        "output_format": args.output_format or base.get("output_format"),
        "language_code": args.language_code or base.get("language_code"),
        "lowercase": args.lowercase if args.lowercase is not None else bool(base.get("lowercase")),
    }
    if args.voice_id and args.voice_id != base.get("voice_id"):
        r["voice_settings"] = {}
    for flag, name in (("stability", "stability"), ("similarity", "similarity_boost"), ("style", "style"),
                       ("speed", "speed")):
        v = getattr(args, flag)
        if v is not None:
            r["voice_settings"][name] = v
    if args.speaker_boost is not None:
        r["voice_settings"]["use_speaker_boost"] = args.speaker_boost
    if mock:
        r["voice_id"] = r["voice_id"] or "mock"
        r["voice_name"] = r["voice_name"] or "system TTS stand-in"
        r["output_format"] = "wav_44100"
        for k, v in DEFAULT_SETTINGS.items():
            r["voice_settings"].setdefault(k, v)
        return r
    if not r["voice_id"]:
        die("no voice chosen: pass --voice-id, or accept an audition take (voices.py suggest lists candidates)")
    details = voice_details(key, r["voice_id"])
    if not details:
        lib = library_voice(key, r["voice_id"])
        if lib:
            r["voice_name"] = r["voice_name"] or lib.get("name")
            print(f"{lib.get('name')} is a Voice Library voice: used directly by id (no voice slot needed)")
        else:
            warn(f"voice {r['voice_id']} is neither in the account nor found in the Voice Library")
    if details:
        r["voice_name"] = r["voice_name"] or details.get("name")
        saved = details.get("settings") or {}
        for k in DEFAULT_SETTINGS:
            if k not in r["voice_settings"] and saved.get(k) is not None:
                r["voice_settings"][k] = saved[k]
        if details.get("category") == "professional":
            ft = fine_tuned_models(details)
            if ft and r["model_id"] not in ft:
                warn(f"{r['voice_name']} is a professional voice fine-tuned for {', '.join(sorted(ft))}, not "
                     f"{r['model_id']}; expect weaker likeness or accent drift. Audition before a full run.")
    for k, v in DEFAULT_SETTINGS.items():
        r["voice_settings"].setdefault(k, v)
    s = r["voice_settings"]
    if is_v3(r["model_id"]):
        snapped = min((0.0, 0.5, 1.0), key=lambda x: abs(x - float(s["stability"])))
        if snapped != s["stability"]:
            warn(f"eleven_v3 stability takes 0.0 (Creative), 0.5 (Natural) or 1.0 (Robust); using {snapped}")
            s["stability"] = snapped
    if float(s["stability"]) < 0.3 and not is_v3(r["model_id"]):
        warn(f"stability {s['stability']} is very low; long passages tend to rush or turn erratic")
    if not 0.7 <= float(s["speed"]) <= 1.2:
        die("speed must be between 0.7 and 1.2")
    if float(s["speed"]) < 0.9:
        warn(f"speed {s['speed']}: low speed settings tend to separate words (stop-start delivery), not calm them")
    if not r["output_format"]:
        try:
            r["output_format"] = default_output_format(get_subscription(key).get("tier"))
        except ApiError:
            r["output_format"] = "mp3_44100_128"
    return r


def settings_payload(recipe, models):
    s = dict(recipe["voice_settings"])
    m = models.get(recipe["model_id"]) or {}
    if m and not m.get("can_use_style"):
        s.pop("style", None)
    if m and not m.get("can_use_speaker_boost"):
        s.pop("use_speaker_boost", None)
    return s


def take_seed(base, k):
    if base is None:
        return random.randint(0, 4294967295)
    return (int(base) + k * 7919) % 4294967296


# ----------------------------------------------------------------------------- API call

def tts(key, recipe, text, models, context, seed):
    """One TTS request with timestamps. Returns (audio_bytes, alignment, headers, endpoint, sent_context)."""
    if recipe.get("lowercase"):
        # Sent lower-cased when the recipe asks (an audition option); the alignment maps back ignoring case.
        text = text.lower()
        context = {k: (v.lower() if isinstance(v, str) else v) for k, v in context.items()}
    body = {"text": text, "model_id": recipe["model_id"], "voice_settings": settings_payload(recipe, models),
            "apply_text_normalization": "auto"}
    if seed is not None:
        body["seed"] = int(seed)
    if recipe.get("language_code") and recipe["model_id"] != "eleven_multilingual_v2":
        body["language_code"] = recipe["language_code"]
    # eleven_v3 accepts neither request ids nor previous/next text (HTTP 400 unsupported_model).
    ctx = {} if is_v3(recipe["model_id"]) else {k: v for k, v in context.items() if v}
    query = {"output_format": recipe["output_format"]}
    path = f"/v1/text-to-speech/{recipe['voice_id']}"
    for attempt in range(3):
        payload = {**body, **ctx}
        try:
            raw, headers = api_request("POST", path + "/with-timestamps", key=key, body=payload, query=query, retries=2)
            data = json.loads(raw.decode("utf-8"))
            return base64.b64decode(data["audio_base64"]), data.get("alignment"), headers, "with-timestamps", ctx
        except ApiError as e:
            if is_account_blocker(e):
                raise
            t = e.text
            if e.status in (400, 422) and ctx and any(k in t for k in ("previous", "next", "request_id", "stitch")):
                warn(f"context rejected ({e}); retrying without previous/next context")
                ctx = {}
                continue
            if e.status in (400, 403, 422) and "output_format" in t and query["output_format"] != "mp3_44100_128":
                warn(f"{query['output_format']} refused ({e}); falling back to mp3_44100_128")
                query["output_format"] = recipe["output_format"] = "mp3_44100_128"
                continue
            if e.status in (400, 404, 422) and "timestamp" in t:
                raw, headers = api_request("POST", path, key=key, body=payload, query=query, accept="audio/*", retries=2)
                return raw, None, headers, "plain", ctx
            raise
    raise ApiError(0, "text to speech failed after fallbacks")


# ----------------------------------------------------------------------------- mock voice

def _system_tts(text, sr):
    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "s.aiff"
        if shutil.which("say"):
            cmd = ["say", "-r", "150", "-o", str(out), text]
        elif shutil.which("espeak-ng") or shutil.which("espeak"):
            out = Path(d) / "s.wav"
            cmd = [shutil.which("espeak-ng") or shutil.which("espeak"), "-s", "150", "-w", str(out), text]
        else:
            return None
        if subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode != 0:
            return None
        return decode_audio(out, sr, 1)[:, 0]


def _bursts(n_words, sr):
    rng = np.random.default_rng(n_words)
    out = []
    for _ in range(max(1, n_words)):
        n = int(0.28 * sr)
        t = np.arange(n) / sr
        env = np.sin(np.pi * t / t[-1]) ** 2 * (0.6 + 0.4 * np.sin(2 * np.pi * 4.5 * t) ** 2)
        tone = np.sin(2 * np.pi * (170 + 20 * rng.random()) * t) + 0.3 * rng.standard_normal(n)
        out += [0.2 * env * tone, np.zeros(int(0.08 * sr))]
    return np.concatenate(out).astype(np.float32)


def mock_tts(text, spans, sr=SR):
    """Offline stand-in: each phrase from the system TTS (or tone bursts), joined by 280 ms gaps,
    with a proportional character alignment. Returns (audio, alignment)."""
    lead, gap = int(0.15 * sr), np.zeros(int(0.28 * sr), dtype=np.float32)
    parts, t, times = [np.zeros(lead, dtype=np.float32)], lead / sr, []
    for a, b in spans:
        phrase = strip_tags(text[a:b])
        x = _system_tts(phrase, sr)
        x = _bursts(count_words(phrase), sr) if x is None or not len(x) else x
        x = trim_digital_silence(x, -60.0)
        times.append((t, t + len(x) / sr))
        parts += [x.astype(np.float32), gap]
        t += (len(x) + len(gap)) / sr
    audio = np.concatenate(parts)
    starts, ends = [0.0] * len(text), [0.0] * len(text)
    for (a, b), (s, e) in zip(spans, times):
        step = (e - s) / max(1, b - a)
        for k in range(a, b):
            starts[k], ends[k] = s + (k - a) * step, s + (k - a + 1) * step
    for i, (a, _) in enumerate(spans[1:], 1):
        sep = a - 1
        starts[sep], ends[sep] = times[i - 1][1], times[i][0]
    return audio, {"characters": list(text), "character_start_times_seconds": starts,
                   "character_end_times_seconds": ends}


# ----------------------------------------------------------------------------- takes & selection

def passages_dir(root, sid):
    return root / "voice" / "passages" / str(sid)


def load_selection(root):
    sel = read_json(root / "voice" / "selection.json", {}) or {}
    sel.setdefault("segments", {})
    return sel


def save_selection(root, sel):
    write_json(root / "voice" / "selection.json", sel)


def take_meta(root, sid, take):
    return read_json(passages_dir(root, sid) / f"{take}.json")


def selected_meta(root, sel, sid):
    entry = sel["segments"].get(str(sid))
    return take_meta(root, sid, entry["take"]) if entry else None


def list_takes(folder):
    """Take names (take-01, take-0003...) that have a metadata file, in order."""
    if not folder.exists():
        return []
    return sorted(p.stem for p in folder.glob("take-*.json") if re.fullmatch(r"take-\d+", p.stem))


def next_take(folder, width=2):
    n = max([int(t.split("-")[1]) for t in list_takes(folder)] or [0])
    return f"take-{n + 1:0{width}d}"


def fresh_id(meta, recipe):
    if not meta or meta.get("mock") or not meta.get("request_id"):
        return None
    if meta.get("voice_id") != recipe["voice_id"] or meta.get("model_id") != recipe["model_id"]:
        return None
    if now_unix() - int(meta.get("created_unix", 0)) > REQUEST_ID_MAX_AGE_S:
        return None
    return meta["request_id"]


def build_context(root, script, sel, idx, recipe, keep_tags, batch=()):
    """Stitching context for segment idx. Neighbours that this run is about to regenerate
    (`batch`) are described by text only, never by the request id of a take being replaced.
    eleven_v3 takes no context at all."""
    if is_v3(recipe["model_id"]):
        return {}
    segs = script["segments"]
    prev_ids, prev_text = [], []
    for j in range(max(0, idx - 3), idx):
        rid = fresh_id(selected_meta(root, sel, segs[j]["id"]), recipe)
        if rid:
            prev_ids.append(rid)
        else:
            prev_ids = []   # ids must be contiguous up to this segment
        prev_text.append(segment_request(segs[j], False)[0])
    nxt_ids, nxt_text = [], ""
    if idx + 1 < len(segs):
        rid = None if str(segs[idx + 1]["id"]) in batch else fresh_id(selected_meta(root, sel, segs[idx + 1]["id"]), recipe)
        if rid:
            nxt_ids.append(rid)
        nxt_text = segment_request(segs[idx + 1], False)[0]
    ctx = {"previous_request_ids": prev_ids or None, "next_request_ids": nxt_ids or None}
    if not prev_ids:
        ctx["previous_text"] = " ".join(prev_text)[-600:] or None
    if not nxt_ids:
        ctx["next_text"] = nxt_text[:400] or None
    return ctx


def save_take(folder, name, audio_bytes, output_format, alignment, meta):
    folder.mkdir(parents=True, exist_ok=True)
    if isinstance(audio_bytes, np.ndarray):
        path = folder / f"{name}.wav"
        write_wav_f32(path, audio_bytes)
    else:
        path = save_api_audio(audio_bytes, output_format, folder / name)
    meta["audio_file"] = path.name
    meta["audio_sha256"] = sha256_file(path)
    meta["duration_s"] = round(probe_duration(path) or 0.0, 3)
    screen(meta, decode_audio(path, SR, 1)[:, 0], alignment)
    meta["credits"] = header_credits(meta.get("headers"))
    if alignment:
        write_json(folder / f"{name}.alignment.json", alignment)
        meta["alignment_file"] = f"{name}.alignment.json"
    write_json(folder / f"{name}.json", meta)
    return path


def screen(meta, x, alignment):
    """Screening numbers stored with every take: pace, voiced share, pauses inside phrases."""
    meta["speech_wpm"] = round(speech_wpm(x, meta.get("text", "")), 1)
    meta["voicing"] = round(voicing_ratio(x), 3)
    meta["inner_pauses"] = inner_pauses(x, alignment, meta.get("text", ""),
                                        [tuple(sp) for sp in meta.get("phrase_spans") or []])
    meta.pop("inner_gaps", None)


def pauses_text(meta):
    if "inner_pauses" not in meta:
        return "- (run --rescreen)"
    return format_pauses(meta["inner_pauses"]) or "none"


def voice_reference(root):
    """(words/min, voiced share) of the accepted audition: what every passage should stay close to."""
    v = read_json(root / "voice.json", {}) or {}
    wpm, voiced = v.get("measured_wpm"), v.get("measured_voicing")
    if voiced is None and v.get("accepted_take"):
        voiced = (read_json(root / "voice" / "audition" / f"{v['accepted_take']}.json", {}) or {}).get("voicing")
    return wpm, voiced


def take_score(meta, ref):
    """Sort key, lower is better: breaks inside phrases first (count, then total length), then
    distance from the accepted voice's breathiness and pace. Screening, not a verdict."""
    pauses = meta.get("inner_pauses") or []
    wpm_ref, voiced_ref = ref
    drift = 0.0
    if voiced_ref is not None and meta.get("voicing") is not None:
        drift += abs(meta["voicing"] - voiced_ref)
    if wpm_ref and meta.get("speech_wpm"):
        drift += 0.5 * abs(meta["speech_wpm"] - wpm_ref) / wpm_ref
    return len(pauses), round(sum(float(g) for _, _, g in pauses), 2), round(drift, 3)


def drift_note(meta, ref):
    """Plain words when a take strays from the accepted voice (pace 25%, breathiness 10 points)."""
    wpm_ref, voiced_ref = ref
    out = []
    if wpm_ref and meta.get("speech_wpm") and abs(meta["speech_wpm"] - wpm_ref) > 0.25 * wpm_ref:
        out.append(f"pace {meta['speech_wpm']:.0f} vs {wpm_ref:.0f} words/min in the accepted audition")
    if voiced_ref is not None and meta.get("voicing") is not None and abs(meta["voicing"] - voiced_ref) > 0.10:
        out.append(f"voiced {meta['voicing'] * 100:.0f}% vs {voiced_ref * 100:.0f}% in the accepted audition")
    return "; ".join(out)


def write_preview(folder, name, x, meta, alignment, seg):
    """The take with the segment's phrase rests inserted: what the listener should judge."""
    from assemble_voice import preview
    audio, _ = preview(x, meta, alignment, seg)
    path = folder / f"{name}.preview.mp3"
    write_mp3(path, audio)
    return path


def generate(key, recipe, text, spans, models, context, seed, mock):
    if mock:
        audio, alignment = mock_tts(text, spans)
        return audio, alignment, {}, "mock", {}
    return tts(key, recipe, text, models, context, seed)


def pace(wpm):
    """Words per minute with a flag outside the usual range of calm narration."""
    if not wpm:
        return "-"
    return f"{wpm:.0f}" + (" fast" if wpm > PACE_FAST else " slow" if wpm < PACE_SLOW else "")


def audition_row(meta, accepted=None):
    s = meta.get("voice_settings") or {}
    gaps = pauses_text(meta)
    voiced = f"{meta['voicing'] * 100:.0f}%" if meta.get("voicing") is not None else "-"
    model = MODEL_LABELS.get(meta.get("model_id"), (meta.get("model_id") or "").replace("eleven_", ""))
    if meta.get("lowercase"):
        model += " lc"
    label = meta.get("voice_name") or meta.get("voice_id") or ""
    if meta.get("method") == "speech_to_speech":
        label = f"{label[:13]} < {str(meta.get('source_voice') or 'recording')[:13]}"
    return (f"{meta['take']:<11}{label[:30]:<31}{model[:15]:<16}"
            f"{s.get('speed', '-')!s:<6}{s.get('stability', '-')!s:<6}{s.get('similarity_boost', '-')!s:<5}"
            f"{pace(meta.get('speech_wpm')):<9}{voiced:<8}"
            f"{gaps}{'  (accepted)' if meta['take'] == accepted else ''}")


AUDITION_HEADER = (f"{'take':<11}{'voice':<31}{'model':<16}{'speed':<6}{'stab':<6}{'sim':<5}{'wpm':<9}{'voiced':<8}"
                   "pauses inside phrases")
MODEL_LABELS = {"eleven_multilingual_v2": "multiling v2", "eleven_turbo_v2_5": "turbo v2.5", "eleven_flash_v2_5":
                "flash v2.5", "eleven_v3": "v3", "eleven_multilingual_sts_v2": "convert", "eleven_english_sts_v2":
                "convert (en)"}


# ----------------------------------------------------------------------------- commands

def cmd_list(root, script):
    sel = load_selection(root)
    print(f"{'seg':<5}{'takes':>6}  {'selected':<10}{'frozen':<8}{'age':>6}  {'wpm':<9}{'voiced':<8}{'pauses':<8}text")
    for seg in script["segments"]:
        sid = str(seg["id"])
        takes = list_takes(passages_dir(root, sid))
        entry = sel["segments"].get(sid) or {}
        meta = take_meta(root, sid, entry["take"]) if entry.get("take") else None
        age = "-"
        if meta:
            mins = (now_unix() - int(meta.get("created_unix", 0))) / 60
            age = f"{mins:.0f}m" if mins < 120 else f"{mins / 60:.0f}h"
            if meta.get("mock"):
                age += "*"
        text = segment_request(seg, False)[0]
        m = meta or {}
        voiced = f"{m['voicing'] * 100:.0f}%" if m.get("voicing") is not None else "-"
        npause = str(len(m["inner_pauses"])) if "inner_pauses" in m else "-"
        print(f"{sid:<5}{len(takes):>6}  {entry.get('take', '-'):<10}{'yes' if entry.get('frozen') else '':<8}"
              f"{age:>6}  {pace(m.get('speech_wpm')):<9}{voiced:<8}{npause:<8}{text[:44]}{'...' if len(text) > 44 else ''}")
    print("(* = mock take; request ids older than 2h are not used for stitching; pauses = pauses inside phrases, "
          "- = not screened yet: --rescreen)")
    folder = root / "voice" / "audition"
    auditions = [read_json(folder / f"{t}.json") for t in list_takes(folder)]
    if auditions:
        accepted = (read_json(root / "voice.json", {}) or {}).get("accepted_take")
        print(f"\nAuditions (voice/audition):\n{AUDITION_HEADER}")
        for meta in auditions:
            print(audition_row(meta, accepted))


def cmd_selection(root, script, args):
    sel = load_selection(root)
    ids = {str(s["id"]) for s in script["segments"]}
    changed = []
    for item in (args.select or "").split(","):
        if not item.strip():
            continue
        sid, _, take = item.partition("=")
        sid, take = sid.strip(), take.strip()
        if sid not in ids or not (passages_dir(root, sid) / f"{take}.json").exists():
            die(f"--select {item}: unknown segment or take")
        sel["segments"].setdefault(sid, {})["take"] = take
        sel["segments"][sid]["updated_at"] = now_iso()
        changed.append(f"{sid}={take}")
    for flag, value in (("freeze", True), ("unfreeze", False)):
        for sid in (getattr(args, flag) or "").split(","):
            sid = sid.strip()
            if not sid:
                continue
            if sid not in sel["segments"]:
                die(f"--{flag} {sid}: segment has no selected take")
            sel["segments"][sid]["frozen"] = value
            changed.append(f"{flag} {sid}")
    save_selection(root, sel)
    print("updated selection: " + ", ".join(changed))


def cmd_accept(root, script, take):
    meta = read_json(root / "voice" / "audition" / f"{take}.json")
    if not meta:
        die(f"voice/audition/{take}.json not found")
    recipe = {k: meta.get(k) for k in ("voice_id", "voice_name", "model_id", "voice_settings", "seed",
                                       "output_format", "language_code", "lowercase")}
    recipe.update({"accepted_take": take, "accepted_text": meta.get("text"), "accepted_at": now_iso(),
                   "measured_wpm": meta.get("speech_wpm"), "measured_voicing": meta.get("voicing")})
    if meta.get("mock"):
        warn("accepting a mock take: the recipe is a placeholder, not a listened-to voice")
    if meta.get("method") == "speech_to_speech":
        die(f"{take} is a conversion ({meta.get('source_voice')} performance in {meta.get('voice_name')}'s voice). "
            "Passages cannot be produced that way yet: every passage would need the guide voice's take and a "
            "conversion. Accept the guide's own take, or build the guide-then-convert route first.")
    write_json(root / "voice.json", recipe)
    print(f"wrote voice.json from {take}: {recipe['voice_name']} ({recipe['voice_id']}), {recipe['model_id']}, "
          f"{json.dumps(recipe['voice_settings'])}, seed {recipe['seed']}")
    planned = (script.get("timing") or {}).get("words_per_minute", 90)
    if meta.get("speech_wpm") and abs(meta["speech_wpm"] - planned) > 15:
        print(f"this voice measured ~{meta['speech_wpm']:.0f} words/min against the script's planning figure of "
              f"{planned:g}; set timing.words_per_minute to {meta['speech_wpm']:.0f} and rerun validate_script.py "
              "--write for a realistic estimate (rests, not speech speed, fill the target)")
    # Reuse the accepted audition as a segment take when it is exactly that segment's passage.
    keep = is_v3(recipe["model_id"])
    for seg in script["segments"]:
        text, spans = segment_request(seg, keep)
        if text == meta.get("text"):
            folder = passages_dir(root, seg["id"])
            name = next_take(folder)
            folder.mkdir(parents=True, exist_ok=True)
            src = root / "voice" / "audition"
            audio = src / meta["audio_file"]
            shutil.copy2(audio, folder / f"{name}{audio.suffix}")
            new = dict(meta, segment_id=str(seg["id"]), take=name, audio_file=f"{name}{audio.suffix}",
                       phrase_spans=spans, reused_from=f"audition/{take}")
            if meta.get("alignment_file"):
                shutil.copy2(src / meta["alignment_file"], folder / f"{name}.alignment.json")
                new["alignment_file"] = f"{name}.alignment.json"
            write_json(folder / f"{name}.json", new)
            sel = load_selection(root)
            sel["segments"][str(seg["id"])] = {"take": name, "frozen": True, "updated_at": now_iso(),
                                               "note": f"accepted audition {take}"}
            save_selection(root, sel)
            print(f"reused {take} as segment {seg['id']} {name} (frozen)")
            break


def cmd_preview(root, script, takes):
    """Previews of existing audition takes with the current script's phrase rests. Works when a
    take's words and punctuation equal a segment's text, so phrasing can change without a retake."""
    folder = root / "voice" / "audition"
    names = list_takes(folder) if takes == "all" else [
        t if t.startswith("take-") else f"take-{int(t):04d}" for t in takes.split(",") if t.strip()]
    for name in names:
        meta = read_json(folder / f"{name}.json")
        if not meta:
            die(f"voice/audition/{name}.json not found")
        keep = is_v3(meta.get("model_id"))
        seg = next((s for s in script["segments"] if segment_request(s, keep)[0] == meta.get("text")), None)
        if seg is None:
            print(f"  {name}: its text matches no segment of the current script; retake to hear new wording")
            continue
        _, spans = segment_request(seg, keep)
        if len(spans) < 2:
            print(f"  {name}: segment {seg['id']} is a single phrase; nothing to insert")
            continue
        meta["phrase_spans"] = spans
        alignment = read_json(folder / meta["alignment_file"]) if meta.get("alignment_file") else None
        x = decode_audio(folder / meta["audio_file"], SR, 1)[:, 0]
        screen(meta, x, alignment)
        meta["preview_file"] = write_preview(folder, name, x, meta, alignment, seg).name
        write_json(folder / f"{name}.json", meta)
        rests = ", ".join(f"{p.get('pause_after_ms', 0) / 1000:g}s" for p in seg["phrases"][:-1])
        print(f"  {name}: {folder / meta['preview_file']}  (segment {seg['id']}, rests {rests})")


def cmd_rescreen(root):
    """Recompute the screening numbers of every take (after the measures improve)."""
    folders = [root / "voice" / "audition"] + sorted(p for p in (root / "voice" / "passages").glob("*") if p.is_dir())
    n = 0
    for folder in folders:
        for name in list_takes(folder):
            meta = read_json(folder / f"{name}.json")
            if not meta or not meta.get("audio_file"):
                continue
            alignment = read_json(folder / meta["alignment_file"]) if meta.get("alignment_file") else None
            screen(meta, decode_audio(folder / meta["audio_file"], SR, 1)[:, 0], alignment)
            write_json(folder / f"{name}.json", meta)
            n += 1
    print(f"rescreened {n} take(s); see --list")


def conversion_source(root, script, spec, segments):
    """(audio path, text, spans, alignment, description) for --convert: an audition take whose
    phrasing is wanted, or a recording of a segment's words (for example the listener's own)."""
    folder = root / "voice" / "audition"
    name = spec if spec.startswith("take-") else (f"take-{int(spec):04d}" if spec.isdigit() else None)
    if name and (folder / f"{name}.json").exists():
        meta = read_json(folder / f"{name}.json")
        alignment = read_json(folder / meta["alignment_file"]) if meta.get("alignment_file") else None
        return (folder / meta["audio_file"], meta["text"], [tuple(sp) for sp in meta["phrase_spans"]], alignment,
                {"source_take": name, "source_voice": meta.get("voice_name") or meta.get("voice_id"),
                 "source_recipe": {k: meta.get(k) for k in ("voice_id", "model_id", "voice_settings", "seed")}})
    path = Path(spec).expanduser()
    if not path.is_file():
        die(f"--convert {spec}: neither an audition take nor an audio file")
    seg = script["segments"][0] if not segments else next(
        (sg for sg in script["segments"] if str(sg["id"]) == segments.split(",")[0]), None)
    if seg is None:
        die(f"segment {segments} not found")
    text, spans = segment_request(seg, False)
    return path, text, spans, None, {"source_file": str(path), "source_voice": "recording"}


def cmd_convert(root, script, args, key):
    """Speech to speech: re-voice a guide performance in the target voice. Timing, phrasing,
    pace and accent come from the guide; the timbre from the target voice."""
    src, text, spans, alignment, source = conversion_source(root, script, args.convert, args.segments)
    seg = next((sg for sg in script["segments"] if segment_request(sg, False)[0] == text), None)
    if seg is not None:
        spans = segment_request(seg, False)[1]   # the script's current phrasing, not the guide's
    recipe = resolve_recipe(root, args, key)
    settings = {k: v for k, v in recipe["voice_settings"].items() if k != "speed"}
    dur = probe_duration(src) or 0.0
    est = dur / 60 * STS_CREDITS_PER_MINUTE
    print(f"convert: {source.get('source_take') or src.name} ({source['source_voice']}) -> {recipe['voice_name']} "
          f"({recipe['voice_id']}), {args.sts_model}, {json.dumps(settings)}, {dur:.1f}s ~ {est:,.0f} credits")
    if args.dry_run:
        return
    before = check_budget(est, args.max_credits, key, "conversion")
    seed = take_seed(recipe["seed"], 0)
    body, ctype = multipart({"model_id": args.sts_model, "voice_settings": json.dumps(settings), "seed": seed,
                             "remove_background_noise": False},
                            {"audio": (src.name, src.read_bytes(), "audio/mpeg" if src.suffix == ".mp3" else "audio/wav")})
    try:
        raw, headers = api_request("POST", f"/v1/speech-to-speech/{recipe['voice_id']}", key=key, raw=body,
                                   content_type=ctype, query={"output_format": recipe["output_format"]},
                                   accept="audio/*", retries=2)
    except ApiError as e:
        fail_api(e, "speech to speech")
    folder = root / "voice" / "audition"
    name = next_take(folder, width=4)
    out_dur = probe_duration(save_api_audio(raw, recipe["output_format"], folder / name)) or 0.0
    # The conversion keeps the guide's timing, so the guide's character alignment still applies.
    if alignment is not None and abs(out_dur - dur) > 0.08:
        warn(f"converted audio is {out_dur:.2f}s against the guide's {dur:.2f}s; the guide's alignment is not reused")
        alignment = None
    meta = {"take": name, "created_at": now_iso(), "created_unix": now_unix(), "method": "speech_to_speech",
            **source, "voice_id": recipe["voice_id"], "voice_name": recipe["voice_name"], "model_id": args.sts_model,
            "voice_settings": settings, "seed": seed, "output_format": recipe["output_format"], "text": text,
            "phrase_spans": spans, "source_seconds": round(dur, 3), "estimated_credits": est,
            "request_id": headers.get("request-id"),
            "headers": {h: headers[h] for h in INTERESTING_HEADERS if h in headers},
            "endpoint": "speech-to-speech", "mock": False}
    path = save_take(folder, name, raw, recipe["output_format"], alignment, meta)
    ledger(root, {"kind": "sts", "purpose": "audition", "take": name, "model": args.sts_model,
                  "voice_id": recipe["voice_id"], "source": source.get("source_take") or str(src),
                  "seconds": round(dur, 2), "estimated_credits": est, "credits": meta.get("credits"),
                  "request_id": meta["request_id"], "file": str(path.relative_to(root))})
    if before is not None:
        log_usage(root, key, before, "conversion", meta.get("credits") or 0, est)
    if seg is not None and len(spans) > 1:
        meta["preview_file"] = write_preview(folder, name, decode_audio(path, SR, 1)[:, 0], meta, alignment, seg).name
        write_json(folder / f"{name}.json", meta)
    print(f"{AUDITION_HEADER}\n{audition_row(meta)}\n{'':<11}{path}")
    if meta.get("preview_file"):
        print(f"{'':<11}{folder / meta['preview_file']}  (with the script's phrase rests)")


def cmd_audition(root, script, args, key, mock):
    models = {} if mock else get_models(key)
    voice_ids = [v.strip() for v in (args.voice_id or "").split(",") if v.strip()] or [None]
    recipes = [resolve_recipe(root, argparse.Namespace(**{**vars(args), "voice_id": vid}), key, mock)
               for vid in voice_ids]
    keep = is_v3(recipes[0]["model_id"])
    seg = None
    if args.text:
        text, spans = args.text.strip(), [(0, len(args.text.strip()))]
    else:
        seg = script["segments"][0] if not args.segments else next(
            (s for s in script["segments"] if str(s["id"]) == args.segments.split(",")[0]), None)
        if seg is None:
            die(f"segment {args.segments} not found")
        text, spans = segment_request(seg, keep)
    rate = 0.0 if mock else tts_rate(recipes[0]["model_id"], models)
    est = len(text) * rate * args.takes * len(recipes)
    for recipe in recipes:
        print(f"{'mock ' if mock else ''}audition: {recipe['voice_name']} ({recipe['voice_id']}), {recipe['model_id']}, "
              f"{json.dumps(recipe['voice_settings'])}, {args.takes} take(s) x {len(text)} chars")
    print(f"text: {text}\nestimate: ~{est:,.0f} credits")
    if args.dry_run:
        return
    before = None if mock else check_budget(est, args.max_credits, key, "audition")
    folder = root / "voice" / "audition"
    rows = []
    for recipe in recipes:
        for k in range(args.takes):
            seed = take_seed(recipe["seed"], k)
            try:
                audio, alignment, headers, endpoint, _ = generate(key, recipe, text, spans, models, {}, seed, mock)
            except ApiError as e:
                fail_api(e, "text to speech")
            name = next_take(folder, width=4)
            meta = {"take": name, "created_at": now_iso(), "created_unix": now_unix(), **recipe, "seed": seed,
                    "text": text, "phrase_spans": spans, "characters": len(text), "estimated_credits": len(text) * rate,
                    "request_id": headers.get("request-id"),
                    "headers": {h: headers[h] for h in INTERESTING_HEADERS if h in headers},
                    "endpoint": endpoint, "mock": mock}
            path = save_take(folder, name, audio, recipe["output_format"], alignment, meta)
            if not mock:
                ledger(root, {"kind": "tts", "purpose": "audition", "take": name, "model": recipe["model_id"],
                              "voice_id": recipe["voice_id"], "chars": len(text), "estimated_credits": len(text) * rate,
                              "credits": meta.get("credits"), "request_id": meta["request_id"],
                              "file": str(path.relative_to(root))})
            if seg is not None and len(spans) > 1:
                meta["preview_file"] = write_preview(folder, name, decode_audio(path, SR, 1)[:, 0], meta, alignment,
                                                     seg).name
                write_json(folder / f"{name}.json", meta)
            rows.append((meta, path))
    if before is not None:
        log_usage(root, key, before, "audition", sum(m.get("credits") or 0 for m, _ in rows), est)
    print(AUDITION_HEADER)
    for meta, path in rows:
        print(f"{audition_row(meta)}\n{'':<11}{path}")
        if meta.get("preview_file"):
            print(f"{'':<11}{folder / meta['preview_file']}  (with the script's phrase rests)")
    print(f"(voiced = share of speech with pitch: ~0% full whisper, 30-45% heavy breathy whisper, 50-65% soft and "
          f"breathy, ~80% ordinary speech; fast/slow = outside {PACE_SLOW}-{PACE_FAST} words/min; screening only)")
    if len(rows) > 1:
        clean = [m["take"] for m, _ in rows if not m.get("inner_pauses")]
        print(f"without breaks inside phrases: {', '.join(clean) if clean else 'none of these'}")
    print("Listen to each take against the brief (accent, calm, connected phrasing) before accepting one:")
    print(f"  python3 scripts/synthesize.py {root} --accept {rows[-1][0]['take']}")


def cmd_generate(root, script, args, key, mock):
    if not mock and not (root / "voice.json").exists() and not args.voice_id:
        die("no voice.json: audition and --accept a take first (or pass --voice-id to override)")
    models = {} if mock else get_models(key)
    recipe = resolve_recipe(root, args, key, mock)
    keep = is_v3(recipe["model_id"])
    sel = load_selection(root)
    wanted = [s.strip() for s in args.segments.split(",")] if args.segments else None
    segs = script["segments"]
    ids = [str(s["id"]) for s in segs]
    if wanted:
        for w in wanted:
            if w not in ids:
                die(f"segment {w} not found")
    limit = int((models.get(recipe["model_id"]) or {}).get("maximum_text_length_per_request") or 5000)
    todo = []
    for i, seg in enumerate(segs):
        sid = str(seg["id"])
        if wanted and sid not in wanted:
            continue
        entry = sel["segments"].get(sid) or {}
        if entry.get("frozen") and not args.force:
            print(f"  {sid}: frozen ({entry.get('take')}), skipped")
            continue
        if entry.get("take") and not args.retake:
            if wanted:
                print(f"  {sid}: already has {entry['take']} (add --retake to replace it)")
            continue
        text, spans = segment_request(seg, keep)
        if len(text) > limit:
            die(f"segment {sid} is {len(text)} characters; {recipe['model_id']} takes at most {limit} per request. "
                "Split it into two segments.")
        todo.append((i, seg, text, spans))
    if not todo:
        print("nothing to generate: every segment has a selected take (use --segments/--retake to replace)")
        return
    batch = {str(seg["id"]) for _, seg, _, _ in todo}
    rate = 0.0 if mock else tts_rate(recipe["model_id"], models)
    chars = sum(len(t) for _, _, t, _ in todo) * args.takes
    est = chars * rate
    print(f"{'mock ' if mock else ''}synthesis: {recipe['voice_name']} ({recipe['voice_id']}), {recipe['model_id']}, "
          f"{json.dumps(recipe['voice_settings'])}, {recipe['output_format']}")
    print(f"{len(todo)} segment(s) x {args.takes} take(s), {chars:,} chars ~ {est:,.0f} credits")
    if args.dry_run:
        for i, seg, text, _ in todo:
            ctx = build_context(root, script, sel, i, recipe, keep, batch)
            shown = {k: (v if isinstance(v, list) else f"{len(v)} chars") for k, v in ctx.items() if v}
            print(f"  {seg['id']}: {len(text)} chars, context {shown or 'none'}\n      {text[:110]}{'...' if len(text) > 110 else ''}")
        return
    before = None if mock else check_budget(est, args.max_credits, key, "synthesis")
    charged = 0
    ref = voice_reference(root)
    picked = []
    for i, seg, text, spans in todo:
        sid = str(seg["id"])
        folder = passages_dir(root, sid)
        made = []
        for k in range(args.takes):
            seed = take_seed(recipe["seed"], len(list_takes(folder)))
            ctx = build_context(root, script, sel, i, recipe, keep, batch)
            try:
                audio, alignment, headers, endpoint, sent = generate(key, recipe, text, spans, models, ctx, seed, mock)
            except ApiError as e:
                fail_api(e, f"text to speech for segment {sid}")
            name = next_take(folder)
            meta = {"segment_id": sid, "take": name, "created_at": now_iso(), "created_unix": now_unix(), **recipe,
                    "seed": seed, "text": text, "phrase_spans": spans, "characters": len(text),
                    "estimated_credits": len(text) * rate, "request_id": headers.get("request-id"),
                    "headers": {h: headers[h] for h in INTERESTING_HEADERS if h in headers}, "endpoint": endpoint,
                    "context": {k: (v if isinstance(v, list) else len(v)) for k, v in sent.items()}, "mock": mock}
            path = save_take(folder, name, audio, recipe["output_format"], alignment, meta)
            charged += meta.get("credits") or 0
            if not mock:
                ledger(root, {"kind": "tts", "segment": sid, "take": name, "model": recipe["model_id"],
                              "voice_id": recipe["voice_id"], "chars": len(text), "estimated_credits": len(text) * rate,
                              "credits": meta.get("credits"), "request_id": meta["request_id"],
                              "file": str(path.relative_to(root))})
            made.append(meta)
            if not args.pick and (not args.keep_selection or sid not in sel["segments"]):
                sel["segments"][sid] = {"take": name, "frozen": False, "updated_at": now_iso()}
                save_selection(root, sel)
            print(f"  {sid} {name}: {meta['duration_s']:.1f}s, speech ~{pace(meta['speech_wpm'])} words/min, voiced "
                  f"{meta['voicing'] * 100:.0f}%, "
                  + (f"PAUSES INSIDE PHRASES {pauses_text(meta)}, " if meta["inner_pauses"] else "")
                  + f"seed {seed}, "
                  f"{'alignment' if alignment else 'NO alignment'}, context "
                  f"{','.join(sorted(meta['context'])) or 'none'}  -> {path.relative_to(root)}")
        if args.pick and made:
            best = min(made, key=lambda m: take_score(m, ref))
            sel["segments"][sid] = {"take": best["take"], "frozen": False, "updated_at": now_iso(),
                                    "note": f"picked from {len(made)} by screening"}
            save_selection(root, sel)
            clean = sum(1 for m in made if not m.get("inner_pauses"))
            print(f"  {sid} picked {best['take']}: {clean} of {len(made)} takes without breaks inside phrases"
                  + ("" if clean else f"; best has {pauses_text(best)}: listen, retake, or reword"))
            picked.append(best)
    if picked:
        notes = [(m["segment_id"], drift_note(m, ref)) for m in picked]
        notes = [(sid, n) for sid, n in notes if n]
        print("consistency with the accepted voice: " + ("all picked passages within range" if not notes else
              "; ".join(f"{sid}: {n}" for sid, n in notes)))
    if before is not None:
        log_usage(root, key, before, "synthesis", charged, est)
    print("Next: python3 scripts/assemble_voice.py " + str(root))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("session", help="session folder (or its script.json)")
    mode = ap.add_argument_group("mode")
    mode.add_argument("--audition", action="store_true", help="generate takes of the opening passage (or --text)")
    mode.add_argument("--accept", metavar="TAKE", help="write voice.json from an audition take (e.g. take-0003)")
    mode.add_argument("--list", action="store_true", help="show takes, selection and freeze state per segment")
    mode.add_argument("--preview", metavar="TAKES", help="render audition takes (comma list or 'all') with the "
                      "current script's phrase rests; free")
    mode.add_argument("--rescreen", action="store_true", help="recompute pace, voiced share and pauses for every take")
    mode.add_argument("--convert", metavar="SOURCE", help="speech to speech: re-voice an audition take (or a recording "
                      "of the opening's words) in --voice-id's voice, keeping its phrasing")
    mode.add_argument("--select", metavar="SEG=TAKE,...", help="choose which take each segment uses")
    mode.add_argument("--freeze", metavar="SEGS", help="protect segments from regeneration (comma list)")
    mode.add_argument("--unfreeze", metavar="SEGS", help="allow segments to be regenerated again")
    gen = ap.add_argument_group("generation")
    gen.add_argument("--segments", help="comma list of segment ids (default: all without a take)")
    gen.add_argument("--retake", action="store_true", help="generate a new take even if one is selected")
    gen.add_argument("--force", action="store_true", help="also regenerate frozen segments")
    gen.add_argument("--takes", type=int, default=1, help="takes per segment / audition (default 1)")
    gen.add_argument("--keep-selection", action="store_true", help="add takes without selecting them")
    gen.add_argument("--pick", action="store_true", help="with --takes N: select each passage's best take by screening "
                     "(fewest breaks inside phrases, then closest to the accepted voice)")
    gen.add_argument("--text", help="audition this text instead of the opening passage")
    gen.add_argument("--dry-run", action="store_true", help="print requests and credits; call nothing")
    gen.add_argument("--max-credits", type=float, help="refuse to start above this estimate")
    gen.add_argument("--mock", action="store_true", help="offline stand-in voice from the system TTS (not for delivery)")
    rec = ap.add_argument_group("recipe overrides (default: voice.json)")
    rec.add_argument("--voice-id", help="voice to use; with --audition, a comma list compares voices on the same text")
    rec.add_argument("--model", help="eleven_multilingual_v2 (default), eleven_v3, eleven_flash_v2_5, ...")
    rec.add_argument("--stability", type=float)
    rec.add_argument("--similarity", type=float)
    rec.add_argument("--style", type=float)
    rec.add_argument("--speed", type=float, help="0.7-1.2")
    rec.add_argument("--speaker-boost", dest="speaker_boost", action="store_true", default=None)
    rec.add_argument("--no-speaker-boost", dest="speaker_boost", action="store_false")
    rec.add_argument("--seed", type=int)
    rec.add_argument("--output-format", help="mp3_44100_192 (Creator+), pcm_44100 (Pro+), mp3_44100_128")
    rec.add_argument("--language-code", help="ISO 639-1, for models that accept it")
    rec.add_argument("--lowercase", dest="lowercase", action="store_true", default=None,
                     help="send the text lower-cased (an option to audition when a voice breaks phrases)")
    rec.add_argument("--no-lowercase", dest="lowercase", action="store_false")
    rec.add_argument("--sts-model", default=STS_MODEL, help="speech to speech model for --convert "
                     "(eleven_multilingual_sts_v2, eleven_english_sts_v2)")
    args = ap.parse_args()

    root = session_root(args.session)
    script = load_script(root)
    if args.list:
        return cmd_list(root, script)
    if args.preview:
        return cmd_preview(root, script, args.preview)
    if args.rescreen:
        return cmd_rescreen(root)
    if args.select or args.freeze or args.unfreeze:
        return cmd_selection(root, script, args)
    if args.accept:
        return cmd_accept(root, script, args.accept)
    if args.voice_id and "," in args.voice_id and not args.audition:
        die("several --voice-id values are for --audition only; synthesis uses one voice")
    key = None if args.mock else api_key(root)
    if args.convert:
        if args.mock:
            die("--convert needs the API; there is no mock conversion")
        return cmd_convert(root, script, args, key)
    if args.audition:
        return cmd_audition(root, script, args, key, args.mock)
    return cmd_generate(root, script, args, key, args.mock)


if __name__ == "__main__":
    main()
