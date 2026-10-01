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
  python3 scripts/synthesize.py SESSION --accept take-0030   # a conversion: every passage becomes guide + convert
  python3 scripts/synthesize.py SESSION --preview 0003,0005     # takes with the script's phrase rests (free)
  python3 scripts/synthesize.py SESSION --accept take-0003      # recipe -> voice.json (+ reuse as segment)
  python3 scripts/synthesize.py SESSION --dry-run               # requests, context and credits; no calls
  python3 scripts/synthesize.py SESSION                         # every segment without a take
  python3 scripts/synthesize.py SESSION --takes 3 --pick         # best of 3 per passage (breaks, then consistency)
  python3 scripts/synthesize.py SESSION --segments 03,05 --retake
  python3 scripts/synthesize.py SESSION --list | --freeze 01,02 | --unfreeze 03 | --select 04=take-01
  python3 scripts/synthesize.py SESSION --mock                  # offline stand-in voice (system TTS)

Recipe: voice.json (voice_id, model_id, voice_settings, seed, output_format), overridden by flags.
Conversion recipe (accepting a --convert take): voice.json also holds a `guide` voice; each passage is
read by the guide (N takes with --takes, best by screening), and only that take is converted into the
target voice, keeping the guide's timing, phrasing and alignment.
Context: for models that support request stitching, previous/next request ids of neighbouring
selected takes younger than two hours; otherwise previous_text/next_text; eleven_v3 accepts
neither, so its passages are generated without context. Some voices read much faster with context
than in their audition: when a reading with context runs faster than the accepted pace, the rest of
the run reads without it (as --no-context does). A multilingual v2 reading longer than about 23.7 s
comes back squeezed to that length; such takes are flagged, and the fix is a shorter segment.
Audio tags are sent only to eleven_v3 models and stripped for every other model. Pauses are never
put in the text.
"""
from __future__ import annotations

import argparse
import base64
import json
import random
import re
import shutil
import tempfile
from pathlib import Path

import numpy as np

from gm_common import (
    ApiError, SR, api_json, api_key, api_request, at_length_limit, check_budget, count_words, decode_audio,
    default_output_format, die, fail_api, format_pauses, get_models, get_subscription, header_credits, inner_pauses,
    is_account_blocker, is_v3, ledger, load_script, log_usage, multipart, now_iso, now_unix, probe_duration, read_json,
    REQUEST_ID_MAX_AGE_S, run_tool, save_api_audio, segment_request, session_root, sha256_file, speech_wpm,
    STS_CREDITS_PER_MINUTE, strip_tags, trim_digital_silence, tts_rate, V2_REQUEST_MAX_S, voicing_ratio, warn, write_json,
    write_mp3, write_wav_f32,
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
    if base.get("method") == "speech_to_speech":
        # A conversion recipe: text-to-speech auditions start from its guide voice, never its STS model.
        g = dict(base.get("guide") or {})
        base = {**g, "output_format": g.get("output_format") or base.get("output_format")}
    r = {
        "voice_id": args.voice_id or base.get("voice_id"),
        "voice_name": base.get("voice_name") if not args.voice_id or args.voice_id == base.get("voice_id") else None,
        "model_id": args.model or base.get("model_id") or "eleven_multilingual_v2",
        "voice_settings": dict(base.get("voice_settings") or {}),
        "seed": args.seed if args.seed is not None else base.get("seed"),
        "output_format": args.output_format or base.get("output_format"),
        "language_code": args.language_code or base.get("language_code"),
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
    if "_sts_" in str(r["model_id"]):
        die(f"{r['model_id']} is a speech-to-speech model; pass a text-to-speech --model")
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
        if run_tool(cmd).returncode != 0:
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
    """The TTS request id behind a take, usable for stitching; for a conversion, its guide reading's."""
    if not meta or meta.get("mock"):
        return None
    src = meta.get("guide") if meta.get("method") == "speech_to_speech" else meta
    if not src or not src.get("request_id"):
        return None
    if src.get("voice_id") != recipe["voice_id"] or src.get("model_id") != recipe["model_id"]:
        return None
    if now_unix() - int(src.get("created_unix") or meta.get("created_unix", 0)) > REQUEST_ID_MAX_AGE_S:
        return None
    return src["request_id"]


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
    meta["at_length_limit"] = at_length_limit(meta.get("duration_s"))
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
    drift, off = 0.0, 0.0
    if voiced_ref is not None and meta.get("voicing") is not None:
        drift += abs(meta["voicing"] - voiced_ref)
    if wpm_ref and meta.get("speech_wpm"):
        off = abs(meta["speech_wpm"] - wpm_ref) / wpm_ref
        drift += 0.5 * off
    # A take far from the accepted pace (over 20 percent) counts like one more break: listeners
    # reject a rushed passage as readily as a broken one.
    count = len(pauses) + (1 if off > 0.20 else 0)
    return count, round(sum(float(g) for _, _, g in pauses), 2), round(drift, 3)


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


def guides_dir(root, sid):
    return root / "voice" / "guides" / str(sid)


TAG_PREFIX_RE = re.compile(r"^\s*(\[[^\[\]]+\])\s*")


def conversion_recipes(root, mock=False):
    """(guide, target) from a conversion voice.json: the guide reads, the target is the voice heard."""
    v = read_json(root / "voice.json", {}) or {}
    g = dict(v.get("guide") or {})
    if not g.get("voice_id") and not mock:
        die("voice.json is a conversion recipe without a guide voice; accept a --convert take again")
    guide = {"voice_id": g.get("voice_id") or "mock", "voice_name": g.get("voice_name") or "guide",
             "model_id": g.get("model_id") or "eleven_multilingual_v2",
             "voice_settings": {**DEFAULT_SETTINGS, **(g.get("voice_settings") or {})}, "seed": g.get("seed"),
             "output_format": "wav_44100" if mock else (g.get("output_format") or v.get("output_format") or "mp3_44100_128"),
             "language_code": None, "text_prefix": g.get("text_prefix") or ""}
    target = {"voice_id": v.get("voice_id") or "mock", "voice_name": v.get("voice_name"),
              "model_id": v.get("model_id") or STS_MODEL,
              "voice_settings": {k: val for k, val in (v.get("voice_settings") or {}).items() if k != "speed"},
              "seed": v.get("seed"), "output_format": v.get("output_format") or "mp3_44100_128"}
    return guide, target


def sts(key, target, src, seed):
    """One speech-to-speech request: the audio at `src` in the target voice. Returns (bytes, headers)."""
    body, ctype = multipart({"model_id": target["model_id"], "voice_settings": json.dumps(target["voice_settings"]),
                             "seed": seed, "remove_background_noise": False},
                            {"audio": (src.name, src.read_bytes(), "audio/mpeg" if src.suffix == ".mp3" else "audio/wav")})
    return api_request("POST", f"/v1/speech-to-speech/{target['voice_id']}", key=key, raw=body, content_type=ctype,
                       query={"output_format": target["output_format"]}, accept="audio/*", retries=2)


def pace(wpm, ref=None):
    """Words per minute, flagged fast or slow: against the accepted voice's pace (25 percent either
    way) once there is one, else outside the usual range of calm narration."""
    if not wpm:
        return "-"
    fast, slow = (1.25 * ref, 0.75 * ref) if ref else (PACE_FAST, PACE_SLOW)
    return f"{wpm:.0f}" + (" fast" if wpm > fast else " slow" if wpm < slow else "")


def audition_row(meta, accepted=None):
    s = meta.get("voice_settings") or {}
    gaps = pauses_text(meta)
    voiced = f"{meta['voicing'] * 100:.0f}%" if meta.get("voicing") is not None else "-"
    model = MODEL_LABELS.get(meta.get("model_id"), (meta.get("model_id") or "").replace("eleven_", ""))
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
    ref = voice_reference(root)[0]
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
              f"{age:>6}  {pace(m.get('speech_wpm'), ref):<9}{voiced:<8}{npause:<8}{text[:44]}{'...' if len(text) > 44 else ''}")
    print("(* = mock take; request ids older than 2h are not used for stitching; pauses = pauses inside phrases, "
          "- = not screened yet: --rescreen)")
    limited = [str(sg["id"]) for sg in script["segments"]
               if (sel["segments"].get(str(sg["id"])) or {}).get("take")
               and (take_meta(root, sg["id"], sel["segments"][str(sg["id"])]["take"]) or {}).get("at_length_limit")]
    if limited:
        print(f"at the {V2_REQUEST_MAX_S:.1f}s request limit (reading squeezed to fit): {', '.join(limited)}; listen for "
              "a hurried pace, and split the segment if it is")
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
                                       "output_format", "language_code")}
    recipe.update({"accepted_take": take, "accepted_text": meta.get("text"), "accepted_at": now_iso(),
                   "measured_wpm": meta.get("speech_wpm"), "measured_voicing": meta.get("voicing")})
    if meta.get("mock"):
        warn("accepting a mock take: the recipe is a placeholder, not a listened-to voice")
    prefix = ""
    if meta.get("method") == "speech_to_speech":
        src_meta = read_json(root / "voice" / "audition" / f"{meta.get('source_take')}.json", {}) or {}
        if not src_meta:
            die(f"{take} is a conversion, but its guide take {meta.get('source_take')} is missing")
        m = TAG_PREFIX_RE.match(src_meta.get("text") or "")
        if m and is_v3(src_meta.get("model_id")):
            prefix = m.group(1) + " "     # a delivery direction put before the guide's text, e.g. [softly]
        recipe["method"] = "speech_to_speech"
        recipe["guide"] = {"voice_id": src_meta.get("voice_id"), "voice_name": src_meta.get("voice_name"),
                           "model_id": src_meta.get("model_id"), "voice_settings": src_meta.get("voice_settings"),
                           "seed": src_meta.get("seed"), "output_format": src_meta.get("output_format"),
                           "text_prefix": prefix, "accepted_take": src_meta.get("take"),
                           "measured_voicing": src_meta.get("voicing")}
    write_json(root / "voice.json", recipe)
    if recipe.get("guide"):
        g = recipe["guide"]
        print(f"conversion recipe: {g['voice_name']} reads ({g['model_id']}, {json.dumps(g['voice_settings'])}"
              + (f", text prefixed {prefix.strip()}" if prefix else "") + f"), then speech to speech into "
              f"{recipe['voice_name']}")
    print(f"wrote voice.json from {take}: {recipe['voice_name']} ({recipe['voice_id']}), {recipe['model_id']}, "
          f"{json.dumps(recipe['voice_settings'])}, seed {recipe['seed']}")
    planned = (script.get("timing") or {}).get("words_per_minute", 90)
    if meta.get("speech_wpm") and abs(meta["speech_wpm"] - planned) > 15:
        print(f"this voice measured ~{meta['speech_wpm']:.0f} words/min against the script's planning figure of "
              f"{planned:g}; set timing.words_per_minute to {meta['speech_wpm']:.0f} and rerun validate_script.py "
              "--write for a realistic estimate (rests, not speech speed, fill the target)")
    # Reuse the accepted audition as a segment take when it is exactly that segment's passage.
    keep = is_v3(recipe["model_id"])
    accepted_text = meta.get("text") or ""
    if prefix and accepted_text.startswith(prefix):
        accepted_text = accepted_text[len(prefix):]
    for seg in script["segments"]:
        text, spans = segment_request(seg, keep)
        if text == accepted_text:
            folder = passages_dir(root, seg["id"])
            name = next_take(folder)
            folder.mkdir(parents=True, exist_ok=True)
            src = root / "voice" / "audition"
            audio = src / meta["audio_file"]
            shutil.copy2(audio, folder / f"{name}{audio.suffix}")
            new = dict(meta, segment_id=str(seg["id"]), take=name, audio_file=f"{name}{audio.suffix}", text=text,
                       phrase_spans=spans, reused_from=f"audition/{take}")
            if meta.get("method") == "speech_to_speech":
                src_meta = read_json(root / "voice" / "audition" / f"{meta.get('source_take')}.json", {}) or {}
                new["guide"] = {"take": f"audition/{meta.get('source_take')}", "request_id": src_meta.get("request_id"),
                                "voice_id": src_meta.get("voice_id"), "model_id": src_meta.get("model_id"),
                                "created_unix": src_meta.get("created_unix")}
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
    if not path.is_file() and (root / spec).is_file():
        path = root / spec
    if not path.is_file():
        die(f"--convert {spec}: neither an audition take nor an audio file")
    meta = read_json(path.with_suffix(".json"), {}) or {}
    if meta.get("text") and meta.get("phrase_spans"):     # a take elsewhere (e.g. voice/guides/03/take-02.mp3)
        alignment = read_json(path.parent / meta["alignment_file"]) if meta.get("alignment_file") else None
        return (path, meta["text"], [tuple(sp) for sp in meta["phrase_spans"]], alignment,
                {"source_take": str(path.relative_to(root)) if root in path.parents else str(path),
                 "source_voice": meta.get("voice_name") or meta.get("voice_id"),
                 "source_recipe": {k: meta.get(k) for k in ("voice_id", "model_id", "voice_settings", "seed")}})
    seg = script["segments"][0] if not segments else next(
        (sg for sg in script["segments"] if str(sg["id"]) == segments.split(",")[0]), None)
    if seg is None:
        die(f"segment {segments} not found")
    text, spans = segment_request(seg, False)
    return path, text, spans, None, {"source_file": str(path), "source_voice": "recording"}


def cmd_convert(root, script, args, key, mock=False):
    """Speech to speech: re-voice a guide performance in the target voice. Timing, phrasing,
    pace and accent come from the guide; the timbre from the target voice."""
    src, text, spans, alignment, source = conversion_source(root, script, args.convert, args.segments)
    seg = next((sg for sg in script["segments"] if segment_request(sg, False)[0] == text), None)
    if seg is not None:
        spans = segment_request(seg, False)[1]   # the script's current phrasing, not the guide's
    recipe = resolve_recipe(root, args, key, mock)
    settings = {k: v for k, v in recipe["voice_settings"].items() if k != "speed"}
    dur = probe_duration(src) or 0.0
    est = dur / 60 * STS_CREDITS_PER_MINUTE
    print(f"convert: {source.get('source_take') or src.name} ({source['source_voice']}) -> {recipe['voice_name']} "
          f"({recipe['voice_id']}), {args.sts_model}, {json.dumps(settings)}, {dur:.1f}s ~ {est:,.0f} credits")
    if args.dry_run:
        return
    before = None if mock else check_budget(est, args.max_credits, key, "conversion")
    seed = take_seed(recipe["seed"], 0)
    target = {"voice_id": recipe["voice_id"], "model_id": args.sts_model, "voice_settings": settings,
              "output_format": recipe["output_format"]}
    folder = root / "voice" / "audition"
    name = next_take(folder, width=4)
    if mock:        # rehearsal: the guide's own audio stands in for the conversion
        raw, headers, out_dur = decode_audio(src, SR, 1)[:, 0], {}, dur
    else:
        try:
            raw, headers = sts(key, target, src, seed)
        except ApiError as e:
            fail_api(e, "speech to speech")
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
            "endpoint": "mock" if mock else "speech-to-speech", "mock": mock}
    path = save_take(folder, name, raw, recipe["output_format"], alignment, meta)
    if not mock:
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
        m = TAG_PREFIX_RE.match(text)
        if m:   # a delivery direction before a segment's own words: keep that segment's phrasing and preview
            rest = text[m.end():]
            match = next((sg for sg in script["segments"] if segment_request(sg, keep)[0] == rest), None)
            if match is not None:
                seg = match
                spans = [(a + m.end(), b + m.end()) for a, b in segment_request(match, keep)[1]]
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


def conversion_cps(root):
    """Characters per second of the accepted audition: estimates the audio a conversion will bill."""
    v = read_json(root / "voice.json", {}) or {}
    m = read_json(root / "voice" / "audition" / f"{v.get('accepted_take')}.json", {}) or {}
    if m.get("duration_s") and m.get("text"):
        return max(5.0, len(m["text"]) / float(m["duration_s"]))
    return 9.0


def convert_guide(root, key, sid, guide_folder, best, target, recipe, text, spans, mock):
    """Speech to speech of one guide take into the target voice, saved as the segment's next passage
    take with the guide's alignment (the conversion keeps timing). Returns the new take's meta."""
    src = guide_folder / best["audio_file"]
    pfolder = passages_dir(root, sid)
    pfolder.mkdir(parents=True, exist_ok=True)
    pname = next_take(pfolder)
    seed = take_seed(target["seed"], len(list_takes(pfolder)))
    if mock:
        audio, headers, fmt = decode_audio(src, SR, 1)[:, 0], {}, "wav_44100"
    else:
        try:
            audio, headers = sts(key, target, src, seed)
        except ApiError as e:
            fail_api(e, f"speech to speech for segment {sid}")
        fmt = target["output_format"]
        out_dur = probe_duration(save_api_audio(audio, fmt, pfolder / pname)) or 0.0
    alignment = read_json(guide_folder / best["alignment_file"]) if best.get("alignment_file") else None
    if alignment is not None and not mock and abs(out_dur - float(best.get("duration_s") or 0)) > 0.08:
        warn(f"segment {sid}: converted audio is {out_dur:.2f}s against the guide's {best.get('duration_s')}s; "
             "the guide's alignment is not reused (phrase edges come from the waveform)")
        alignment = None
    meta = {"segment_id": sid, "take": pname, "created_at": now_iso(), "created_unix": now_unix(),
            "method": "speech_to_speech", "voice_id": target["voice_id"], "voice_name": target["voice_name"],
            "model_id": target["model_id"], "voice_settings": target["voice_settings"], "seed": seed,
            "output_format": fmt, "text": text, "phrase_spans": spans, "source_seconds": best.get("duration_s"),
            "guide": {"take": f"guides/{sid}/{best['take']}", "request_id": best.get("request_id"),
                      "voice_id": recipe["voice_id"], "voice_name": recipe["voice_name"], "model_id": recipe["model_id"],
                      "created_unix": best.get("created_unix"), "inner_pauses": best.get("inner_pauses")},
            "request_id": headers.get("request-id"), "headers": {h: headers[h] for h in INTERESTING_HEADERS if h in headers},
            "endpoint": "mock" if mock else "speech-to-speech", "mock": mock}
    save_take(pfolder, pname, audio, fmt, alignment, meta)
    return meta


def cmd_convert_guide(root, script, spec, key, mock=False):
    """Convert a chosen guide take (SEG:TAKE, e.g. 01:take-04) into the segment's next passage take
    and select it: overrules the automatic pick without new readings."""
    sid, _, take = spec.partition(":")
    seg = next((sg for sg in script["segments"] if str(sg["id"]) == sid), None)
    folder = guides_dir(root, sid)
    best = read_json(folder / f"{take}.json")
    if seg is None or not best:
        die(f"--convert-guide {spec}: segment or guide take not found")
    recipe, target = conversion_recipes(root, mock)
    text, spans = segment_request(seg, is_v3(recipe["model_id"]))
    before = None if mock else check_budget((best.get("duration_s") or 0) / 60 * STS_CREDITS_PER_MINUTE, None, key,
                                             "conversion")
    cmeta = convert_guide(root, key, sid, folder, best, target, recipe, text, spans, mock)
    pfolder = passages_dir(root, sid)
    cal = read_json(pfolder / cmeta["alignment_file"]) if cmeta.get("alignment_file") else None
    cmeta["preview_file"] = write_preview(pfolder, cmeta["take"], decode_audio(pfolder / cmeta["audio_file"], SR, 1)[:, 0],
                                          cmeta, cal, seg).name
    write_json(pfolder / f"{cmeta['take']}.json", cmeta)
    if not mock:
        ledger(root, {"kind": "sts", "purpose": "passage", "segment": sid, "take": cmeta["take"], "model": target["model_id"],
                      "voice_id": target["voice_id"], "guide": cmeta["guide"]["take"], "seconds": best.get("duration_s"),
                      "credits": cmeta.get("credits"), "request_id": cmeta.get("request_id")})
        log_usage(root, key, before, "conversion", cmeta.get("credits") or 0, 0)
    sel = load_selection(root)
    sel["segments"][sid] = {"take": cmeta["take"], "frozen": False, "updated_at": now_iso(), "note": f"converted guide {take}"}
    save_selection(root, sel)
    print(f"  {sid} converted guide {take} -> {cmeta['take']}: voiced {cmeta['voicing'] * 100:.0f}%, "
          f"~{pace(cmeta['speech_wpm'], voice_reference(root)[0])} words/min, pauses inside phrases "
          f"{pauses_text(cmeta)}\n"
          f"      preview: {pfolder / cmeta['preview_file']}")


def cmd_generate(root, script, args, key, mock):
    if not mock and not (root / "voice.json").exists() and not args.voice_id:
        die("no voice.json: audition and --accept a take first (or pass --voice-id to override)")
    models = {} if mock else get_models(key)
    base = read_json(root / "voice.json", {}) or {}
    conversion = base.get("method") == "speech_to_speech" and not args.voice_id
    target = None
    if conversion:
        # The guide reads (flags adjust its delivery); the target voice is fixed by the accepted conversion.
        recipe, target = conversion_recipes(root, mock)
        for flag, name in (("stability", "stability"), ("similarity", "similarity_boost"), ("style", "style"),
                           ("speed", "speed")):
            if getattr(args, flag) is not None:
                recipe["voice_settings"][name] = getattr(args, flag)
        if args.seed is not None:
            recipe["seed"] = args.seed
    else:
        recipe = resolve_recipe(root, args, key, mock)
    keep = is_v3(recipe["model_id"])
    prefix = recipe.get("text_prefix") or ""
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
        if len(text) + len(prefix) > limit:
            die(f"segment {sid} is {len(text)} characters; {recipe['model_id']} takes at most {limit} per request. "
                "Split it into two segments.")
        todo.append((i, seg, text, spans))
    if not todo:
        print("nothing to generate: every segment has a selected take (use --segments/--retake to replace)")
        return
    batch = {str(seg["id"]) for _, seg, _, _ in todo}
    rate = 0.0 if mock else tts_rate(recipe["model_id"], models)
    chars = sum(len(t) + len(prefix) for _, _, t, _ in todo) * args.takes
    est = chars * rate
    sts_seconds = sum(len(t) for _, _, t, _ in todo) / conversion_cps(root) if conversion else 0.0
    if conversion and not mock:
        est += sts_seconds / 60 * STS_CREDITS_PER_MINUTE
    print(f"{'mock ' if mock else ''}synthesis: {recipe['voice_name']} ({recipe['voice_id']}), {recipe['model_id']}, "
          f"{json.dumps(recipe['voice_settings'])}, {recipe['output_format']}" + (f", text prefixed {prefix.strip()}" if prefix else ""))
    if conversion:
        print(f"conversion: each passage's best guide take -> {target['voice_name']} ({target['voice_id']}), "
              f"{target['model_id']}, {json.dumps(target['voice_settings'])}; about {sts_seconds:.0f}s of audio converted")
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
    guide_ref = (ref[0], ((base.get("guide") or {}).get("measured_voicing")))
    picked = []
    use_ctx = not args.no_context
    for i, seg, text, spans in todo:
        sid = str(seg["id"])
        folder = guides_dir(root, sid) if conversion else passages_dir(root, sid)
        made = []

        def off_pace(m):
            """How far a reading is from the accepted pace (share, signed), or 0 when that says little."""
            wpm = m.get("speech_wpm") or 0
            if not ref[0] or not wpm or count_words(text) < 12:     # pace means little over a few words
                return 0.0
            return (wpm - ref[0]) / ref[0]

        def good_enough(m):
            """A reading worth keeping (or converting): near the accepted pace, at most one short break."""
            pauses = m.get("inner_pauses") or []
            return (abs(off_pace(m)) <= args.pace_tolerance and len(pauses) <= 1
                    and all(float(g) < 0.35 for _, _, g in pauses))

        for k in range(args.takes):
            if made and good_enough(made[-1]):
                break           # stop at the first good reading; for a conversion, convert only that one
            if made and made[-1].get("at_length_limit"):
                break           # the text reads longer than one request allows: more takes squeeze it the same way
            seed = take_seed(recipe["seed"], len(list_takes(folder)))
            # Stitching context makes some voices read like continuous narration, much faster than in
            # their audition; once a reading with context runs fast, the run continues without it.
            ctx = build_context(root, script, sel, i, recipe, keep, batch) if use_ctx else {}
            req_text = prefix + text
            req_spans = [(a + len(prefix), b + len(prefix)) for a, b in spans]
            try:
                audio, alignment, headers, endpoint, sent = generate(key, recipe, req_text, req_spans, models, ctx, seed, mock)
            except ApiError as e:
                fail_api(e, f"text to speech for segment {sid}")
            name = next_take(folder)
            meta = {"segment_id": sid, "take": name, "created_at": now_iso(), "created_unix": now_unix(), **recipe,
                    "seed": seed, "text": text, "phrase_spans": spans, "characters": len(req_text),
                    "estimated_credits": len(req_text) * rate, "request_id": headers.get("request-id"),
                    "headers": {h: headers[h] for h in INTERESTING_HEADERS if h in headers}, "endpoint": endpoint,
                    "context": {k: (v if isinstance(v, list) else len(v)) for k, v in sent.items()}, "mock": mock}
            path = save_take(folder, name, audio, recipe["output_format"], alignment, meta)
            charged += meta.get("credits") or 0
            if not mock:
                ledger(root, {"kind": "tts", "purpose": "guide" if conversion else "passage", "segment": sid,
                              "take": name, "model": recipe["model_id"], "voice_id": recipe["voice_id"],
                              "chars": len(req_text), "estimated_credits": len(req_text) * rate,
                              "credits": meta.get("credits"), "request_id": meta["request_id"],
                              "file": str(path.relative_to(root))})
            made.append(meta)
            if use_ctx and meta["context"] and off_pace(meta) > args.pace_tolerance:
                use_ctx = False
                print(f"  {sid}: with stitching context this voice read {meta['speech_wpm']:.0f} words/min against "
                      f"{ref[0]:.0f} in the accepted audition; the rest of this run reads without context "
                      + ("(the next take is one)" if k + 1 < args.takes else
                         f"(retake {sid} with --no-context, or use --takes 2)"))
            if not conversion and not args.pick and (not args.keep_selection or sid not in sel["segments"]):
                sel["segments"][sid] = {"take": name, "frozen": False, "updated_at": now_iso()}
                save_selection(root, sel)
            print(f"  {sid} {'guide ' if conversion else ''}{name}: {meta['duration_s']:.1f}s, speech "
                  f"~{pace(meta['speech_wpm'], ref[0])} words/min, voiced {meta['voicing'] * 100:.0f}%, "
                  + (f"PAUSES INSIDE PHRASES {pauses_text(meta)}, " if meta["inner_pauses"] else "")
                  + (f"AT THE {V2_REQUEST_MAX_S:.1f}s REQUEST LIMIT (squeezed to fit: split this segment), "
                     if meta.get("at_length_limit") else "")
                  + f"seed {seed}, "
                  f"{'alignment' if alignment else 'NO alignment'}, context "
                  f"{','.join(sorted(meta['context'])) or 'none'}  -> {path.relative_to(root)}")
        if conversion and made:
            best = min(made, key=lambda m: take_score(m, guide_ref))
            if abs(off_pace(best)) > args.pace_tolerance and not args.force_convert:
                if best.get("at_length_limit"):
                    print(f"  {sid}: the reading fills the {V2_REQUEST_MAX_S:.1f}s request limit, squeezed to "
                          f"{best.get('speech_wpm'):.0f} words/min; not converted. Split the segment in two, or "
                          f"convert it anyway with --convert-guide {sid}:{best['take']}")
                else:
                    print(f"  {sid}: no reading within {args.pace_tolerance:.0%} of {guide_ref[0]:.0f} words/min "
                          f"(best {best.get('speech_wpm'):.0f}); not converted. Run again for more readings, convert "
                          f"one with --convert-guide {sid}:TAKE, or pass --force-convert")
                continue
            cmeta = convert_guide(root, key, sid, folder, best, target, recipe, text, spans, mock)
            charged += cmeta.get("credits") or 0
            if not mock:
                ledger(root, {"kind": "sts", "purpose": "passage", "segment": sid, "take": cmeta["take"],
                              "model": target["model_id"], "voice_id": target["voice_id"], "guide": cmeta["guide"]["take"],
                              "seconds": best.get("duration_s"), "credits": cmeta.get("credits"),
                              "request_id": cmeta.get("request_id"),
                              "file": f"voice/passages/{sid}/{cmeta['audio_file']}"})
            if not args.keep_selection or sid not in sel["segments"]:
                sel["segments"][sid] = {"take": cmeta["take"], "frozen": False, "updated_at": now_iso(),
                                        "note": f"converted guide {best['take']} (best of {len(made)})"}
                save_selection(root, sel)
            clean = sum(1 for m in made if not m.get("inner_pauses"))
            pfolder = passages_dir(root, sid)
            cal = read_json(pfolder / cmeta["alignment_file"]) if cmeta.get("alignment_file") else None
            cmeta["preview_file"] = write_preview(pfolder, cmeta["take"], decode_audio(pfolder / cmeta["audio_file"], SR, 1)[:, 0],
                                                  cmeta, cal, seg).name
            write_json(pfolder / f"{cmeta['take']}.json", cmeta)
            print(f"  {sid} converted guide {best['take']} -> {cmeta['take']}: voiced {cmeta['voicing'] * 100:.0f}%, "
                  f"~{pace(cmeta['speech_wpm'], ref[0])} words/min, pauses inside phrases {pauses_text(cmeta)} "
                  f"({clean} of {len(made)} guide takes clean)\n      preview: {pfolder / cmeta['preview_file']}")
            picked.append(cmeta)
        elif args.pick and made:
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
    mode.add_argument("--convert-guide", metavar="SEG:TAKE", help="conversion recipe: convert this guide take "
                      "(e.g. 01:take-04) into the segment's passage take and select it")
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
    gen.add_argument("--pace-tolerance", type=float, default=0.15, help="a reading this far from the accepted pace "
                     "(share, default 0.15) is not good enough: --takes N makes another, and a conversion recipe "
                     "does not convert it")
    gen.add_argument("--force-convert", action="store_true", help="convert the best guide reading even when off pace")
    gen.add_argument("--no-context", action="store_true", help="no stitching context (previous/next request ids or "
                     "text): each passage is read on its own, as in auditions (some voices read much slower so); "
                     "the run switches to this by itself when a reading with context runs fast")
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
        return cmd_convert(root, script, args, key, args.mock)
    if args.convert_guide:
        return cmd_convert_guide(root, script, args.convert_guide, key, args.mock)
    if args.audition:
        return cmd_audition(root, script, args, key, args.mock)
    return cmd_generate(root, script, args, key, args.mock)


if __name__ == "__main__":
    main()
