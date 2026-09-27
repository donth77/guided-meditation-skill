"""
GM COMMON
Shared helpers for the guided-meditation scripts: .env loading, the ElevenLabs REST client and
error explanations, ffmpeg audio I/O, loudness measurement, float WAV streaming, gain envelopes,
loop construction, and script/timeline helpers.

Imported by the sibling scripts (a script's own folder is on sys.path when it runs); not run
directly. Needs Python 3.9+, numpy, and ffmpeg/ffprobe on PATH.
"""
from __future__ import annotations

import difflib
import hashlib
import json
import math
import os
import re
import shutil
import struct
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from datetime import datetime, timezone
from pathlib import Path

try:
    import numpy as np
except ImportError:  # pragma: no cover - environment problem, reported plainly
    sys.stderr.write("error: numpy is required (pip install numpy), or run the script with `uv run`.\n")
    raise SystemExit(1)

SR = 44100
BLOCK = SR * 10
API_BASE = os.environ.get("ELEVENLABS_API_BASE", "https://api.elevenlabs.io").rstrip("/")
USER_AGENT = "guided-meditation-skill/1.0"
VARIANTS = ("voice", "voice+music", "voice+sfx", "voice+music+sfx")

# Credit estimates. TTS rates are read from /v1/models when the API is reachable; these are the
# fallbacks (character_cost_multiplier, September 2026). Music is billed by length, SFX by the
# requested duration.
TTS_RATE_FALLBACK = {
    "eleven_v3": 1.0, "eleven_multilingual_v2": 1.0, "eleven_v3_conversational": 0.5,
    "eleven_flash_v2_5": 0.5, "eleven_turbo_v2_5": 0.5, "eleven_flash_v2": 0.5, "eleven_turbo_v2": 0.5,
}
SFX_CREDITS_PER_SECOND = 40
MUSIC_CREDITS_PER_MINUTE = 900
STS_CREDITS_PER_MINUTE = 1000   # speech to speech (voice changer), per minute of source audio
REQUEST_ID_MAX_AGE_S = 110 * 60   # ElevenLabs ignores stitching ids older than two hours
# The longest audio one eleven_multilingual_v2 request returned in September 2026 (174 takes): a
# reading that would run longer comes back exactly this long, with every word present but the pace
# squeezed to fit and the final breath clipped. A conversion keeps its guide's length.
V2_REQUEST_MAX_S = 23.684


# ----------------------------------------------------------------------------- misc

def die(msg, code=1):
    sys.stderr.write(f"error: {msg}\n")
    raise SystemExit(code)


def warn(msg):
    sys.stderr.write(f"warning: {msg}\n")


def now_iso():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def now_unix():
    return int(time.time())


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path, default=None):
    p = Path(path)
    if not p.exists():
        return default
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def write_json(path, data):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")
    os.replace(tmp, p)


def fmt_time(seconds):
    """1234.5 -> '20:34.5'; hours when needed."""
    if seconds is None:
        return "-"
    sign = "-" if seconds < 0 else ""
    s = abs(float(seconds))
    m, s = divmod(s, 60.0)
    if m >= 60:
        h, m = divmod(m, 60)
        return f"{sign}{int(h)}:{int(m):02d}:{s:04.1f}"
    return f"{sign}{int(m):02d}:{s:04.1f}"


def db_to_gain(db):
    return 10.0 ** (float(db) / 20.0)


def gain_to_db(g):
    return 20.0 * math.log10(max(float(g), 1e-12))


def smoothstep(u):
    u = np.clip(u, 0.0, 1.0)
    return u * u * (3.0 - 2.0 * u)


def rel(path, root):
    try:
        return str(Path(path).resolve().relative_to(Path(root).resolve()))
    except ValueError:
        return str(path)


def session_root(path):
    p = Path(path).expanduser().resolve()
    if p.is_file():
        p = p.parent
    if not p.exists():
        die(f"session folder {p} does not exist")
    return p


def load_script(root):
    path = Path(root) / "script.json"
    if not path.exists():
        die(f"{path} not found; write the script first (references/script-writing.md)")
    try:
        return read_json(path)
    except json.JSONDecodeError as e:
        die(f"{path} is not valid JSON: {e}")


def slug_of(script, root):
    s = script.get("slug") or script.get("title") or Path(root).name
    s = re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")
    return s or "meditation"


def parse_outputs(spec, script=None):
    """'voice,voice+music' / 'all' / None -> canonical ordered variant list; voice is always made."""
    wanted = {"voice"}
    if not spec:
        layers = enabled_layers(script) if script else set()
        if "music" in layers:
            wanted.add("voice+music")
        if "sfx" in layers:
            wanted.add("voice+sfx")
        if {"music", "sfx"} <= layers:
            wanted.add("voice+music+sfx")
    else:
        for item in re.split(r"[,\s]+", spec.strip()):
            if not item:
                continue
            if item.lower() == "all":
                wanted.update(VARIANTS)
                continue
            parts = {p for p in re.split(r"[+\-]", item.lower()) if p}
            unknown = parts - {"voice", "music", "sfx"}
            if unknown:
                die(f"unknown output '{item}' (use voice, voice+music, voice+sfx, voice+music+sfx or all)")
            parts.add("voice")
            wanted.add("+".join(p for p in ("voice", "music", "sfx") if p in parts))
    return [v for v in VARIANTS if v in wanted]


def variant_slug(variant):
    return variant.replace("+", "-")


def enabled_layers(script):
    layers = set()
    if (script.get("music") or {}).get("enabled"):
        layers.add("music")
    sfx = script.get("sfx") or {}
    if sfx.get("enabled") and ((sfx.get("ambience") or {}).get("enabled") or sfx.get("one_shots")):
        layers.add("sfx")
    return layers


def layers_for(variants):
    need = set()
    for v in variants:
        need.update(v.split("+"))
    need.discard("voice")
    return need


# ----------------------------------------------------------------------------- env / API

def _parse_env_file(path):
    out = {}
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        key, sep, val = line.partition("=")
        if not sep:
            continue
        key, val = key.strip(), val.strip()
        if len(val) >= 2 and val[0] == val[-1] and val[0] in "'\"":
            val = val[1:-1]
        else:
            val = re.split(r"\s+#", val, maxsplit=1)[0].strip()
        out[key] = val
    return out


def load_dotenv(*start_dirs):
    """Load the nearest .env above each start folder (and the working directory).
    Variables already in the environment win. Returns the files read."""
    found = []
    for d in [*start_dirs, Path.cwd()]:
        if d is None:
            continue
        p = Path(d).expanduser().resolve()
        if p.is_file():
            p = p.parent
        for parent in [p, *p.parents]:
            f = parent / ".env"
            if f.is_file():
                if f not in found:
                    found.append(f)
                break
    for f in found:
        for k, v in _parse_env_file(f).items():
            os.environ.setdefault(k, v)
    return found


def api_key(start_dir=None, required=True):
    load_dotenv(start_dir)
    key = os.environ.get("ELEVENLABS_API_KEY", "").strip()
    if not key and required:
        die("ELEVENLABS_API_KEY is not set and no .env containing it was found above the session "
            "folder or the working directory.")
    return key


class ApiError(Exception):
    def __init__(self, status, message, *, kind=None, code=None, request_id=None, detail=None, headers=None):
        super().__init__(message)
        self.status = status
        self.kind = kind
        self.code = code
        self.request_id = request_id
        self.detail = detail
        self.headers = headers or {}

    def __str__(self):
        head = f"HTTP {self.status}" if self.status else "network"
        tags = "/".join(t for t in dict.fromkeys([self.kind, self.code]) if t)
        s = f"{head}{' ' + tags if tags else ''}: {self.args[0]}"
        if self.request_id:
            s += f" (request_id {self.request_id})"
        return s

    def find(self, key):
        """Search the error detail for a key (e.g. prompt_suggestion)."""
        stack = [self.detail]
        while stack:
            cur = stack.pop()
            if isinstance(cur, dict):
                if key in cur:
                    return cur[key]
                stack.extend(cur.values())
            elif isinstance(cur, list):
                stack.extend(cur)
        return None

    @property
    def text(self):
        return f"{self.kind or ''} {self.code or ''} {self.args[0]}".lower()


def _parse_error(status, body, headers):
    text = body.decode("utf-8", "replace") if body else ""
    message, kind, code, request_id, detail = (text.strip()[:600] or f"HTTP {status}"), None, None, None, None
    try:
        payload = json.loads(text) if text else None
    except ValueError:
        payload = None
    if isinstance(payload, dict):
        detail = payload.get("detail", payload)
        if isinstance(detail, dict):
            message = detail.get("message") or detail.get("msg") or json.dumps(detail)[:600]
            kind = detail.get("type") or detail.get("status")
            code = detail.get("code") or detail.get("status")
            request_id = detail.get("request_id")
        elif isinstance(detail, list):
            parts = []
            for d in detail:
                if isinstance(d, dict):
                    loc = ".".join(str(x) for x in d.get("loc", []))
                    parts.append(f"{loc}: {d.get('msg')}" if loc else str(d.get("msg")))
            message = "; ".join(parts) or text[:600]
            kind = "validation_error"
        elif isinstance(detail, str):
            message = detail
    request_id = request_id or headers.get("request-id") or headers.get("x-request-id")
    return ApiError(status, message, kind=kind, code=code, request_id=request_id, detail=detail, headers=headers)


def api_request(method, path, *, key, body=None, query=None, raw=None, content_type=None,
                accept="application/json", timeout=600, retries=3):
    """One ElevenLabs REST call. Returns (bytes, lower-cased headers). Retries 429/5xx and
    network errors with backoff; raises ApiError otherwise."""
    url = API_BASE + path
    if query:
        q = {k: v for k, v in query.items() if v is not None}
        if q:
            url += "?" + urllib.parse.urlencode(q, doseq=True)
    headers = {"xi-api-key": key, "Accept": accept, "User-Agent": USER_AGENT}
    data = None
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        headers["Content-Type"] = "application/json"
    elif raw is not None:
        data = raw
        headers["Content-Type"] = content_type
    attempt = 0
    while True:
        req = urllib.request.Request(url, data=data, method=method, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                payload = resp.read()
                return payload, {k.lower(): v for k, v in resp.headers.items()}
        except urllib.error.HTTPError as e:
            payload = e.read() or b""
            hdrs = {k.lower(): v for k, v in (e.headers or {}).items()}
            err = _parse_error(e.code, payload, hdrs)
            if e.code in (429, 500, 502, 503, 504) and attempt < retries and "quota" not in err.text:
                wait = float(hdrs.get("retry-after") or 0) or min(60.0, 4.0 * 2 ** attempt)
                attempt += 1
                warn(f"{err} -- retrying in {wait:.0f}s ({attempt}/{retries})")
                time.sleep(wait)
                continue
            raise err
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as e:
            if attempt < retries:
                attempt += 1
                wait = min(60.0, 4.0 * 2 ** attempt)
                warn(f"network error ({e}) -- retrying in {wait:.0f}s ({attempt}/{retries})")
                time.sleep(wait)
                continue
            raise ApiError(0, f"network error: {e}")


def api_json(method, path, *, key, **kw):
    payload, headers = api_request(method, path, key=key, **kw)
    return json.loads(payload.decode("utf-8")) if payload else {}, headers


def multipart(fields, files):
    """Encode multipart/form-data. files: {name: (filename, bytes, content_type)}."""
    boundary = "----gm" + uuid.uuid4().hex
    out = bytearray()
    for name, value in fields.items():
        if value is None:
            continue
        for v in (value if isinstance(value, (list, tuple)) else [value]):
            if isinstance(v, bool):
                v = "true" if v else "false"
            out += (f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n{v}\r\n').encode()
    for name, (filename, content, ctype) in files.items():
        out += (f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"; filename="{filename}"\r\n'
                f"Content-Type: {ctype}\r\n\r\n").encode()
        out += content + b"\r\n"
    out += f"--{boundary}--\r\n".encode()
    return bytes(out), f"multipart/form-data; boundary={boundary}"


def explain_api_error(e):
    """Plain-language explanation for the account-level failures that need the user."""
    t = e.text
    if "payment" in t:
        return ("ElevenLabs refused the request because the subscription has a failed or incomplete "
                "payment. Pay the latest invoice in the ElevenLabs account (Subscription page), then rerun.")
    if "quota" in t or "credits" in t and "insufficient" in t:
        return "The account does not have enough credits left for this request."
    if "invalid_api_key" in t or (e.status == 401 and "api key" in t):
        return "The API key was rejected. Check ELEVENLABS_API_KEY."
    if "voice_not_found" in t or ("voice" in t and "not found" in t):
        return ("Voice not found: check the id. Voice Library voices work directly by id (voices.py library "
                "lists them); a voice removed from the library by its owner stops working.")
    if "bad_prompt" in t or "bad_composition_plan" in t:
        return "The music prompt was rejected (often a named artist, band or song)."
    return None


def fail_api(e, what="request"):
    msg = explain_api_error(e)
    sys.stderr.write(f"error: ElevenLabs {what} failed: {e}\n")
    if msg:
        sys.stderr.write(f"       {msg}\n")
    raise SystemExit(2 if msg else 1)


def is_account_blocker(e):
    t = e.text
    return e.status in (401, 402, 403) and any(k in t for k in ("payment", "quota", "invalid_api_key", "unauthorized"))


def get_subscription(key):
    data, _ = api_json("GET", "/v1/user/subscription", key=key, retries=1)
    return data


def credits_remaining(sub):
    return max(0, int(sub.get("character_limit") or 0) - int(sub.get("character_count") or 0))


def usage_count(key):
    """Current character/credit count, or None when it cannot be read."""
    try:
        return int(get_subscription(key).get("character_count") or 0)
    except ApiError:
        return None


def tier_level(tier):
    t = (tier or "").lower()
    if any(k in t for k in ("enterprise", "business", "scale", "publisher", "pro", "growing")):
        return 3
    if "creator" in t:
        return 2
    if "starter" in t:
        return 1
    if "free" in t:
        return 0
    return 1


def default_output_format(tier):
    return "mp3_44100_192" if tier_level(tier) >= 2 else "mp3_44100_128"


_MODELS_CACHE = {}


def get_models(key):
    if "models" not in _MODELS_CACHE:
        try:
            data, _ = api_json("GET", "/v1/models", key=key, retries=1)
            _MODELS_CACHE["models"] = {m["model_id"]: m for m in data}
        except ApiError:
            _MODELS_CACHE["models"] = {}
    return _MODELS_CACHE["models"]


def tts_rate(model_id, models=None):
    m = (models or {}).get(model_id) or {}
    rate = (m.get("model_rates") or {}).get("character_cost_multiplier")
    if rate is None:
        rate = TTS_RATE_FALLBACK.get(model_id, 1.0)
    return float(rate)


def is_v3(model_id):
    return (model_id or "").startswith("eleven_v3")


def audio_ext(output_format):
    codec = (output_format or "mp3").split("_")[0]
    return {"mp3": ".mp3", "pcm": ".wav", "wav": ".wav", "opus": ".opus"}.get(codec, ".bin")


def save_api_audio(data, output_format, stem):
    """Write API audio bytes next to `stem` (no extension). Raw PCM gets a WAV header (16-bit mono)."""
    codec = output_format.split("_")[0]
    path = Path(str(stem) + audio_ext(output_format))
    path.parent.mkdir(parents=True, exist_ok=True)
    if codec == "pcm":
        rate = int(output_format.split("_")[1])
        with open(path, "wb") as f:
            f.write(b"RIFF" + struct.pack("<I", 36 + len(data)) + b"WAVE")
            f.write(b"fmt " + struct.pack("<IHHIIHH", 16, 1, 1, rate, rate * 2, 2, 16))
            f.write(b"data" + struct.pack("<I", len(data)))
            f.write(data)
    else:
        path.write_bytes(data)
    return path


def ledger(root, entry):
    """Append one line to <session>/ledger.jsonl (every paid call, plus usage snapshots)."""
    p = Path(root) / "ledger.jsonl"
    with open(p, "a", encoding="utf-8") as f:
        f.write(json.dumps({"time": now_iso(), **entry}, ensure_ascii=False) + "\n")


def header_credits(headers):
    """Credits charged for one call, from ElevenLabs' character-cost response header (None if absent)."""
    v = (headers or {}).get("character-cost")
    try:
        return int(float(v)) if v is not None else None
    except ValueError:
        return None


def log_usage(root, key, before, purpose, charged, estimate):
    """Record a batch's credits. character-cost headers are exact; the account counter can lag
    by minutes, so its delta is logged but not trusted for the printed total."""
    after = usage_count(key)
    entry = {"kind": "usage", "purpose": purpose, "charged": charged, "estimate": round(estimate)}
    if before is not None and after is not None:
        entry.update({"before": before, "after": after, "delta": after - before})
    ledger(root, entry)
    if charged is not None:
        print(f"credits charged: {charged:,} (from response headers; estimate {estimate:,.0f})")
    else:
        print(f"credits: ~{estimate:,.0f} estimated (this API reports no cost header; the account balance "
              "updates after a delay -- check_setup.py shows it)")


def check_budget(estimate, max_credits, key, what):
    """Refuse to start when the estimate exceeds --max-credits or the account's remaining credits."""
    if max_credits is not None and estimate > max_credits:
        die(f"{what} is estimated at {estimate:,.0f} credits, above --max-credits {max_credits:,.0f}")
    try:
        sub = get_subscription(key)
    except ApiError as e:
        if is_account_blocker(e):
            fail_api(e, "account check")
        warn(f"could not read the subscription ({e}); continuing without a credit check")
        return None
    status = (sub.get("status") or "").lower()
    if status in ("past_due", "unpaid", "incomplete"):
        warn(f"subscription status is '{status}'; generation may be refused until the invoice is paid")
    left = credits_remaining(sub)
    if estimate > left:
        die(f"{what} is estimated at {estimate:,.0f} credits but only {left:,} remain this period")
    return int(sub.get("character_count") or 0)


# ----------------------------------------------------------------------------- audio I/O

def ffmpeg_bin(name="ffmpeg"):
    b = shutil.which(name)
    if not b:
        die(f"{name} not found on PATH (install ffmpeg: brew install ffmpeg / apt install ffmpeg)")
    return b


def decode_audio(path, sr=SR, channels=1):
    """Decode any audio file to float32 [frames, channels] at `sr` with ffmpeg."""
    cmd = [ffmpeg_bin(), "-v", "error", "-nostdin", "-i", str(path), "-f", "f32le",
           "-acodec", "pcm_f32le", "-ac", str(channels), "-ar", str(sr), "-"]
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if p.returncode != 0:
        die(f"ffmpeg could not decode {path}: {p.stderr.decode(errors='replace').strip()[:400]}")
    a = np.frombuffer(p.stdout, dtype="<f4")
    a = a[: len(a) - len(a) % channels]
    return a.reshape(-1, channels).copy()


def iter_decode(path, sr=SR, channels=1, block=BLOCK):
    """Stream-decode a file in float32 blocks of [frames, channels]."""
    cmd = [ffmpeg_bin(), "-v", "error", "-nostdin", "-i", str(path), "-f", "f32le",
           "-acodec", "pcm_f32le", "-ac", str(channels), "-ar", str(sr), "-"]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    want = 4 * channels * block
    try:
        while True:
            buf = p.stdout.read(want)
            if not buf:
                break
            a = np.frombuffer(buf, dtype="<f4")
            a = a[: len(a) - len(a) % channels]
            yield a.reshape(-1, channels)
    finally:
        p.stdout.close()
        err = p.stderr.read()
        p.wait()
        if p.returncode not in (0, None, -13):
            warn(f"ffmpeg decode of {path} ended with {p.returncode}: {err.decode(errors='replace')[:200]}")


def probe_duration(path):
    cmd = [ffmpeg_bin("ffprobe"), "-v", "error", "-show_entries", "format=duration",
           "-of", "default=noprint_wrappers=1:nokey=1", str(path)]
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        return float(p.stdout.strip())
    except ValueError:
        return None


def probe_stream(path):
    cmd = [ffmpeg_bin("ffprobe"), "-v", "error", "-select_streams", "a:0", "-show_entries",
           "stream=codec_name,sample_rate,channels,bit_rate:format=duration,bit_rate",
           "-of", "json", str(path)]
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        d = json.loads(p.stdout)
    except ValueError:
        return {}
    s = (d.get("streams") or [{}])[0]
    f = d.get("format") or {}
    return {"codec": s.get("codec_name"), "sample_rate": int(s.get("sample_rate") or 0),
            "channels": int(s.get("channels") or 0), "duration": float(f.get("duration") or 0),
            "bit_rate": int(f.get("bit_rate") or s.get("bit_rate") or 0)}


class WavWriter:
    """Streaming IEEE-float32 WAV writer; the header is finalised on close."""

    def __init__(self, path, sr=SR, channels=1):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.sr, self.ch, self.frames = sr, channels, 0
        self.f = open(self.path, "wb")
        self._header()

    def _header(self):
        data_bytes = self.frames * self.ch * 4
        fmt = struct.pack("<HHIIHHH", 3, self.ch, self.sr, self.sr * self.ch * 4, self.ch * 4, 32, 0)
        fact = struct.pack("<I", self.frames & 0xFFFFFFFF)
        riff = 4 + (8 + len(fmt)) + (8 + len(fact)) + (8 + data_bytes)
        self.f.seek(0)
        self.f.write(b"RIFF" + struct.pack("<I", riff & 0xFFFFFFFF) + b"WAVE")
        self.f.write(b"fmt " + struct.pack("<I", len(fmt)) + fmt)
        self.f.write(b"fact" + struct.pack("<I", len(fact)) + fact)
        self.f.write(b"data" + struct.pack("<I", data_bytes & 0xFFFFFFFF))

    def write(self, block):
        a = np.asarray(block, dtype="<f4")
        if a.ndim == 1:
            a = a[:, None]
        if a.shape[1] != self.ch:
            raise ValueError(f"expected {self.ch} channels, got {a.shape[1]}")
        self.f.write(np.ascontiguousarray(a).tobytes())
        self.frames += a.shape[0]

    def write_silence(self, frames):
        frames = int(frames)
        while frames > 0:
            n = min(frames, BLOCK)
            self.write(np.zeros((n, self.ch), dtype="<f4"))
            frames -= n

    def close(self):
        if self.f.closed:
            return
        end = self.f.tell()
        self._header()
        self.f.seek(end)
        self.f.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def open_wav_f32(path):
    """Memory-map an IEEE-float32 WAV (as written by WavWriter). Returns (array[frames, ch], sr)."""
    with open(path, "rb") as f:
        head = f.read(12)
        if head[:4] != b"RIFF" or head[8:12] != b"WAVE":
            raise ValueError(f"{path} is not a WAV file")
        fmt = None
        while True:
            chunk = f.read(8)
            if len(chunk) < 8:
                raise ValueError(f"{path}: no data chunk")
            cid, size = chunk[:4], struct.unpack("<I", chunk[4:])[0]
            if cid == b"fmt ":
                fmt = struct.unpack("<HHIIHH", f.read(16))
                f.seek(size - 16 + (size & 1), 1)
            elif cid == b"data":
                offset = f.tell()
                break
            else:
                f.seek(size + (size & 1), 1)
    tag, ch, sr, _, _, bits = fmt
    if tag not in (3, 0xFFFE) or bits != 32:
        raise ValueError(f"{path} is not a float32 WAV")
    frames = size // (4 * ch)
    if frames == 0:
        return np.zeros((0, ch), dtype="<f4"), sr
    return np.memmap(path, dtype="<f4", mode="r", offset=offset, shape=(frames, ch)), sr


def write_wav_f32(path, data, sr=SR):
    a = np.asarray(data, dtype="<f4")
    with WavWriter(path, sr, 1 if a.ndim == 1 else a.shape[1]) as w:
        w.write(a)


def write_mp3(path, data, sr=SR, bitrate="192k"):
    """Encode float samples (mono or [n, ch]) to MP3 for listening copies."""
    a = np.asarray(data, dtype="<f4")
    ch = 1 if a.ndim == 1 else a.shape[1]
    cmd = [ffmpeg_bin(), "-v", "error", "-nostdin", "-y", "-f", "f32le", "-ar", str(sr), "-ac", str(ch), "-i", "pipe:0",
           "-c:a", "libmp3lame", "-b:a", bitrate, str(path)]
    p = subprocess.run(cmd, input=np.clip(a, -1.0, 1.0).tobytes(), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if p.returncode != 0:
        die(f"mp3 encode failed for {path}: {p.stderr.decode(errors='replace').strip()[:400]}")
    return Path(path)


def write_flac(path, data, sr=SR):
    """Encode float samples (mono or [n, ch]) to 16-bit FLAC: lossless for sources that began as MP3,
    a fifth of the size of float WAV."""
    a = np.asarray(data, dtype="<f4")
    ch = 1 if a.ndim == 1 else a.shape[1]
    cmd = [ffmpeg_bin(), "-v", "error", "-nostdin", "-y", "-f", "f32le", "-ar", str(sr), "-ac", str(ch), "-i", "pipe:0",
           "-c:a", "flac", "-sample_fmt", "s16", str(path)]
    p = subprocess.run(cmd, input=np.clip(a, -1.0, 1.0).tobytes(), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if p.returncode != 0:
        die(f"flac encode failed for {path}: {p.stderr.decode(errors='replace').strip()[:400]}")
    return Path(path)


def ffmpeg_filter_file(src, dst, filters, channels, sr=SR):
    """Decode `src`, run an ffmpeg -af chain, write float32 WAV `dst`."""
    cmd = [ffmpeg_bin(), "-v", "error", "-nostdin", "-y", "-i", str(src), "-ac", str(channels), "-ar", str(sr)]
    if filters:
        cmd += ["-af", ",".join(filters)]
    cmd += ["-c:a", "pcm_f32le", str(dst)]
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if p.returncode != 0:
        die(f"ffmpeg filter failed for {src}: {p.stderr.strip()[:400]}")
    return Path(dst)


_LOUD_RE = {
    "I": re.compile(r"I:\s*(-?inf|-?\d+(?:\.\d+)?)\s*LUFS"),
    "LRA": re.compile(r"LRA:\s*(-?inf|-?\d+(?:\.\d+)?)\s*LU\b"),
    "TP": re.compile(r"Peak:\s*(-?inf|-?\d+(?:\.\d+)?)\s*dBFS"),
}


def _parse_ebur128(stderr):
    idx = stderr.rfind("Summary:")
    summary = stderr[idx:] if idx >= 0 else stderr
    out = {}
    for k, rx in _LOUD_RE.items():
        m = rx.findall(summary)
        out[k] = float(m[-1]) if m and "inf" not in m[-1] else (None if not m else float("-inf"))
    return out


def measure_loudness(path=None, *, inputs=None, filter_complex=None, dualmono=False):
    """EBU R128 integrated loudness (I, LUFS), loudness range (LRA) and true peak (TP, dBTP) via ffmpeg.
    Either a file, or ffmpeg `inputs` plus a filter_complex whose output is measured."""
    ebu = "ebur128=peak=true:framelog=verbose" + (":dualmono=true" if dualmono else "")
    cmd = [ffmpeg_bin(), "-hide_banner", "-nostats", "-nostdin"]
    if filter_complex:
        for i in inputs:
            cmd += ["-i", str(i)]
        cmd += ["-filter_complex", f"{filter_complex},{ebu}"]
    else:
        cmd += ["-i", str(path), "-filter_complex", ebu]
    cmd += ["-f", "null", "-"]
    p = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True, errors="replace")
    if p.returncode != 0:
        die(f"loudness measurement failed: {p.stderr.strip()[-400:]}")
    return _parse_ebur128(p.stderr)


# ----------------------------------------------------------------------------- analysis

def to_mono(x):
    x = np.asarray(x)
    return x if x.ndim == 1 else x.mean(axis=1)


def level_db(x, sr=SR, win_ms=30.0, hop_ms=10.0):
    """Short-term RMS in dBFS per hop (mono mix). Frame i covers [i*hop, i*hop+win).
    Returns (levels, hop_samples, win_samples)."""
    m = to_mono(x).astype(np.float64)
    hop = max(1, int(round(sr * hop_ms / 1000.0)))
    win = max(hop, int(round(sr * win_ms / 1000.0)))
    if len(m) == 0:
        return np.full(1, -120.0), hop, win
    n = 1 + max(0, len(m) - win) // hop
    csum = np.concatenate(([0.0], np.cumsum(m * m)))
    starts = np.arange(n) * hop
    ends = np.minimum(starts + win, len(m))
    ms = (csum[ends] - csum[starts]) / np.maximum(1, ends - starts)
    return 10.0 * np.log10(ms + 1e-12), hop, win


def voiced_span(x, sr=SR):
    """(first, last) second with speech energy (within 40 dB of the loudest 30 ms frame, so
    whispered onsets and decays count)."""
    lev, hop, win = level_db(x, sr, 30.0, 10.0)
    idx = np.where(lev >= max(float(lev.max()) - 40.0, -65.0))[0]
    if not len(idx):
        return 0.0, 0.0
    return idx[0] * hop / sr, (idx[-1] * hop + win) / sr


def speech_wpm(x, text, sr=SR):
    """Words per minute over the voiced span of a take (screening number, not a verdict)."""
    a, b = voiced_span(x, sr)
    return count_words(text) / max(b - a, 0.1) * 60.0


def at_length_limit(duration_s):
    """True when a take is exactly the multilingual v2 request limit long: its reading was squeezed
    to fit, so its pace says little about the voice and more takes will not help (split the segment)."""
    return bool(duration_s) and abs(float(duration_s) - V2_REQUEST_MAX_S) < 0.03


def voicing_ratio(x, sr=SR, threshold=0.5):
    """Share of speech frames (40 ms) with pitch periodicity, 70-400 Hz, in the band below 1 kHz.
    A full whisper has no vocal-fold vibration and scores near 0; breathy but voiced delivery sits
    in between; ordinary speech is high. A screening number for how whispered a take is, never a
    verdict on how it sounds."""
    m = to_mono(x).astype(np.float64)
    frame, hop = int(0.04 * sr), int(0.02 * sr)
    if len(m) < frame * 2:
        return 0.0
    lev, _, _ = level_db(m, sr, 40.0, 20.0)
    speech = np.where(lev >= float(lev.max()) - 35.0)[0]
    n_fft = 1 << int(math.ceil(math.log2(2 * frame)))
    win = np.hanning(frame)
    low = np.fft.rfftfreq(n_fft, 1.0 / sr) <= 1000.0
    wac = np.fft.irfft(np.abs(np.fft.rfft(win, n_fft)) ** 2)[:frame]
    wac = np.maximum(wac / wac[0], 1e-3)
    lo, hi = int(sr / 400), int(sr / 70)
    voiced = total = 0
    for i in speech:
        seg = m[i * hop: i * hop + frame]
        if len(seg) < frame:
            break
        p = np.abs(np.fft.rfft((seg - seg.mean()) * win, n_fft)) ** 2
        p[~low] = 0.0
        ac = np.fft.irfft(p)[:frame]
        if ac[0] <= 0:
            continue
        total += 1
        if float(np.max(ac[lo:hi] / ac[0] / wac[lo:hi])) > threshold:
            voiced += 1
    return voiced / total if total else 0.0


def quiet_threshold(lev, drop_db=25.0, floor_share=0.35):
    """dBFS below which a frame of a take sounds like a pause: drop_db under the speech level, or
    floor_share of the way up from the noise floor when the voice carries audible breath or room
    tone (some voices never fall more than 15-20 dB between words)."""
    speech, floor = float(np.percentile(lev, 95)), float(np.percentile(lev, 5))
    return max(speech - drop_db, floor + floor_share * (speech - floor))


def inner_pauses(x, alignment, text, spans, sr=SR, min_pause=0.2, min_inside=0.3):
    """Breaks a listener hears inside a phrase: the word-by-word delivery listeners reject ("let's
    pause here ... for a minute"). Two kinds, measured as dips at least 20 dB under the take's
    speech (breath and room tone count as quiet, since some voices never fall silent):

    - between two words with no punctuation between them: the longest dip from 0.15 s before
      one word's aligned end to 0.15 s after the next word's aligned start, min_pause or more.
      Wider windows mistake a word's whispered tail for a pause; the alignment itself spreads
      pauses into the words, so neither its gaps nor a silence threshold find these;
    - inside one word ("h ... ere", from erratic generations): a dip of min_inside or more that
      more of the word follows (0.1 s of speech, so a final consonant's release does not count).

    [(word, next word, seconds)], next word "" for a break inside a word."""
    if not alignment or not alignment.get("characters"):
        return []
    st, en = alignment["character_start_times_seconds"], alignment["character_end_times_seconds"]
    mapping = align_map(text, alignment["characters"])
    mask = spoken_mask(text)
    lev, hop, _ = level_db(x, sr, 30.0, 10.0)
    # A dip of 20 dB for 0.2 s already breaks the flow (calibrated on a take heard as "pause here
    # ... for a minute" whose dip never fell 25 dB).
    quiet = lev < quiet_threshold(lev, 20.0, 0.45)

    def frame(t):
        return min(len(quiet), max(0, int(round(t * sr / hop))))

    def secs(n):
        return n * hop / sr

    out = []
    for a, b in spans:
        ws = []
        for m in WORD_RE.finditer(text[a:b]):
            s0, e0 = m.start() + a, m.end() + a
            if mask[s0] and mapping[s0] is not None and mapping[e0 - 1] is not None:
                ws.append((float(st[mapping[s0]]), float(en[mapping[e0 - 1]]), m.group(), s0, e0))
        for (s_a, e_a, w0, _, e0), (s_b, e_b, w1, s1, _) in zip(ws, ws[1:]):
            if text[e0:s1].strip():
                continue          # punctuation or a tag between the words: a natural place to pause
            lo = max((s_a + e_a) / 2, min(e_a, s_b) - 0.15)
            hi = min((s_b + e_b) / 2, max(e_a, s_b) + 0.15)
            longest = max((secs(r1 - r0) for r0, r1 in runs(quiet[frame(lo):frame(hi) + 1])), default=0.0)
            if longest >= min_pause:
                out.append((w0, w1, round(longest, 2)))
        for s_w, e_w, w, _, _ in ws:
            seg = quiet[frame(s_w):frame(e_w) + 1]
            for r0, r1 in runs(seg):
                if r0 and secs(r1 - r0) >= min_inside:
                    after = runs(~seg[r1:])
                    if after and after[0][0] == 0 and secs(after[0][1]) >= 0.1:
                        out.append((w, "", round(secs(r1 - r0), 2)))
    return out


def format_pauses(pauses):
    """'here|for 0.4s, (here) 0.64s' for inner_pauses() results (tuples or JSON lists)."""
    return ", ".join(f"{a}|{b} {g}s" if b else f"({a}) {g}s" for a, b, g in pauses)


def runs(mask):
    """[(start, end)) index pairs of consecutive True values."""
    m = np.asarray(mask, dtype=np.int8)
    if len(m) == 0:
        return []
    d = np.diff(np.concatenate(([0], m, [0])))
    return list(zip(np.where(d == 1)[0].tolist(), np.where(d == -1)[0].tolist()))


def fade_edges(x, fade_in=0, fade_out=0):
    """In-place linear fades (sample counts) on a [frames] or [frames, ch] array; returns x."""
    n = len(x)
    if fade_in > 0 and n:
        k = min(int(fade_in), n)
        r = np.linspace(0.0, 1.0, k, endpoint=False, dtype=np.float32)
        x[:k] *= r if x.ndim == 1 else r[:, None]
    if fade_out > 0 and n:
        k = min(int(fade_out), n)
        r = np.linspace(1.0, 0.0, k, endpoint=True, dtype=np.float32)
        x[n - k:] *= r if x.ndim == 1 else r[:, None]
    return x


def trim_digital_silence(x, threshold_db=-80.0):
    """Drop leading/trailing samples below threshold (MP3 priming/padding, terminal silence)."""
    m = np.abs(to_mono(x))
    thr = db_to_gain(threshold_db)
    idx = np.where(m > thr)[0]
    if len(idx) == 0:
        return x[:0]
    return x[idx[0]: idx[-1] + 1]


# ----------------------------------------------------------------------------- envelopes

class Envelope:
    """Gain automation. Starts at `initial`; `add(t0, t1, target)` schedules a smoothstep ramp from
    the gain in effect at t0 (ms). Ramps are kept in time order; a ramp still moving when the
    next begins is cut there (recorded in `notes`)."""

    def __init__(self, initial=1.0):
        self.initial = float(initial)
        self.ramps = []   # dicts: t0, t1, g0, g1, cut, label
        self.notes = []

    def value_at(self, t):
        g = self.initial
        for r in self.ramps:
            if t < r["t0"]:
                break
            g = self._ramp_value(r, min(t, r["cut"]))
        return g

    @staticmethod
    def _ramp_value(r, t):
        if r["t1"] <= r["t0"]:
            return r["g1"] if t >= r["t0"] else r["g0"]
        u = (t - r["t0"]) / (r["t1"] - r["t0"])
        u = min(max(u, 0.0), 1.0)
        return r["g0"] + (r["g1"] - r["g0"]) * (u * u * (3.0 - 2.0 * u))

    def add(self, t0, t1, target, label=""):
        t0, t1 = float(t0), float(max(t0, t1))
        if self.ramps and t0 < self.ramps[-1]["t0"]:
            raise ValueError("ramps must be added in time order")
        if self.ramps and t0 < self.ramps[-1]["cut"]:
            prev = self.ramps[-1]
            self.notes.append(f"{label or 'ramp'} starts at {t0 / 1000:.1f}s while "
                              f"{prev['label'] or 'the previous ramp'} is still moving (until "
                              f"{prev['t1'] / 1000:.1f}s); the earlier fade is cut short")
            prev["cut"] = t0
        g0 = self.value_at(t0)
        self.ramps.append({"t0": t0, "t1": t1, "g0": g0, "g1": float(target), "cut": t1, "label": label})

    def block(self, start_frame, n, sr=SR):
        t = (start_frame + np.arange(n, dtype=np.float64)) * (1000.0 / sr)
        g = np.full(n, self.initial, dtype=np.float64)
        for r in self.ramps:
            end = r["cut"]
            end_val = self._ramp_value(r, end)
            if r["t1"] > r["t0"]:
                m = (t >= r["t0"]) & (t < end)
                if m.any():
                    u = (t[m] - r["t0"]) / (r["t1"] - r["t0"])
                    g[m] = r["g0"] + (r["g1"] - r["g0"]) * smoothstep(u)
            g[t >= end] = end_val
        return g.astype(np.float32)

    def max_in(self, t0, t1, step_ms=50.0):
        """Largest gain within [t0, t1] (inclusive, sampled)."""
        if t1 <= t0:
            return self.value_at(t0)
        ts = np.linspace(t0, t1, int(math.ceil((t1 - t0) / step_ms)) + 1)
        return max(self.value_at(float(t)) for t in ts)


def merge_windows(windows, gap=0.0):
    out = []
    for a, b in sorted(windows):
        if out and a - out[-1][1] <= gap:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return [(a, b) for a, b in out]


# ----------------------------------------------------------------------------- looping

def equal_power(n):
    x = (np.arange(n, dtype=np.float64) + 0.5) / max(n, 1)
    return np.cos(x * np.pi / 2).astype(np.float32), np.sin(x * np.pi / 2).astype(np.float32)


def bridge(tail, head):
    """Equal-power crossfade of two equal-length [frames, ch] arrays (uncorrelated material)."""
    fout, fin = equal_power(len(tail))
    return tail * fout[:, None] + head * fin[:, None]


class PieceStream:
    """Plays `prefix` pieces once, then `cycle` pieces forever. read(n) -> float32 [n, ch]."""

    def __init__(self, prefix, cycle, channels):
        self.prefix = [p for p in prefix if len(p)]
        self.cycle = [p for p in cycle if len(p)]
        if not self.cycle:
            raise ValueError("loop cycle is empty")
        self.ch = channels
        self.in_prefix = bool(self.prefix)
        self.i = 0
        self.off = 0

    def _piece(self):
        return self.prefix[self.i] if self.in_prefix else self.cycle[self.i]

    def read(self, n):
        out = np.empty((n, self.ch), dtype=np.float32)
        filled = 0
        while filled < n:
            piece = self._piece()
            take = min(n - filled, len(piece) - self.off)
            out[filled: filled + take] = piece[self.off: self.off + take]
            filled += take
            self.off += take
            if self.off >= len(piece):
                self.off = 0
                self.i += 1
                if self.in_prefix and self.i >= len(self.prefix):
                    self.in_prefix, self.i = False, 0
                elif not self.in_prefix and self.i >= len(self.cycle):
                    self.i = 0
        return out


def _band_profile(x, sr=SR):
    """Mean log energy in 8 octave-ish bands of a window (for comparing loop join candidates)."""
    m = to_mono(x).astype(np.float64)
    if len(m) < 2048:
        return np.zeros(8)
    n = 1 << int(math.log2(min(len(m), 1 << 16)))
    segs = [m[i: i + n] for i in range(0, len(m) - n + 1, max(n, (len(m) - n) // 4 or n))][:5]
    edges = [60, 120, 250, 500, 1000, 2000, 4000, 8000, 16000]
    freqs = np.fft.rfftfreq(n, 1.0 / sr)
    prof = np.zeros(8)
    for s in segs:
        spec = np.abs(np.fft.rfft(s * np.hanning(n))) ** 2
        for b in range(8):
            sel = (freqs >= edges[b]) & (freqs < edges[b + 1])
            prof[b] += spec[sel].mean() if sel.any() else 0.0
    return 10 * np.log10(prof / max(len(segs), 1) + 1e-12)


def music_activity(x, sr=SR, hop_s=0.5, win_s=1.0):
    """How busy a piece of music is over time. Per hop: loudness, onsets (positive spectral flux:
    new notes, swells) and brightness (share above 2 kHz), each ranked within the piece and
    combined into 0-1 (1 = the busiest moments of this piece), smoothed over about 3 s.
    Returns (times_s, activity, event_times_s); events are onsets well above the piece's usual."""
    m = to_mono(x).astype(np.float64)
    hop, win = int(hop_s * sr), int(win_s * sr)
    n = max(1, 1 + (len(m) - win) // hop)
    nfft = 1 << int(math.ceil(math.log2(win)))
    w = np.hanning(win)
    hi = np.fft.rfftfreq(nfft, 1.0 / sr) >= 2000.0
    loud, bright, flux = np.empty(n), np.empty(n), np.zeros(n)
    prev = None
    for i in range(n):
        seg = m[i * hop: i * hop + win]
        if len(seg) < win:
            seg = np.pad(seg, (0, win - len(seg)))
        spec = np.abs(np.fft.rfft(seg * w, nfft))
        pw = spec ** 2
        loud[i] = 10.0 * np.log10(pw.mean() + 1e-12)
        bright[i] = (pw[hi].sum() + 1e-12) / (pw.sum() + 1e-12)
        logmag = np.log1p(spec * 1e3)
        if prev is not None:
            flux[i] = float(np.maximum(logmag - prev, 0.0).mean())
        prev = logmag

    def rank(v):
        return np.argsort(np.argsort(v)) / max(1, len(v) - 1)

    act = 0.5 * rank(loud) + 0.35 * rank(flux) + 0.15 * rank(bright)
    k = max(1, int(round(3.0 / hop_s)))
    act = np.convolve(np.pad(act, (k // 2, k - 1 - k // 2), mode="edge"), np.ones(k) / k, mode="valid")
    q1, q3 = np.percentile(flux, [25, 75])
    thr = float(np.median(flux) + 3.0 * (q3 - q1 + 1e-9))
    # An event stands out even among the piece's notes: among its strongest 5 percent of onsets and
    # 1.5 times the local 90th percentile over +-10 s. Regular notes of a sparse figure are texture.
    r = max(1, int(round(10.0 / hop_s)))
    local = np.array([np.percentile(flux[max(0, i - r): i + r + 1], 90) for i in range(n)])
    top = float(np.percentile(flux, 95))
    events = [round(i * hop_s + win_s / 2, 1) for i in range(1, n - 1)
              if flux[i] > max(thr, top, 1.5 * local[i]) and flux[i] >= flux[i - 1] and flux[i] >= flux[i + 1]
              and loud[i] > np.median(loud) - 12]
    return np.arange(n) * hop_s + win_s / 2, act, events


def music_loop_plan(x, crossfade_ms=12000.0, sr=SR, search_s=6.0):
    """Choose a restart point S after the intro and a crossfade start T before the outro fade,
    matching level and spectrum across the join. Returns (prefix, cycle, info)."""
    lev, hop, win = level_db(x, sr, win_ms=400.0, hop_ms=100.0)
    active = lev > -65.0
    if not active.any():
        die("music source is silent")
    med = float(np.median(lev[active]))
    established = np.where(lev >= med - 4.0)[0]
    body = np.where(lev >= med - 6.0)[0]
    end_cut = min(len(x), int((body[-1] * hop + win)))           # before the outro fade/silence
    s0 = int(established[0] * hop)
    s0 = max(s0, int(2.0 * sr))
    c = int(crossfade_ms / 1000.0 * sr)
    usable = end_cut - s0
    c = int(min(c, max(usable // 3, int(0.5 * sr))))
    if usable < 3 * c or end_cut - c <= s0 + c:
        # Very short source: loop the whole audible span with a shorter crossfade.
        aud = np.where(active)[0]
        s0 = int(aud[0] * hop)
        end_cut = min(len(x), int(aud[-1] * hop + win))
        c = max(int(0.25 * sr), (end_cut - s0) // 4)
    step = int(0.5 * sr)
    best = None
    for kt in range(int(search_s / 0.5) + 1):
        t = end_cut - c - kt * step
        if t - s0 < 2 * c:
            break
        tail = x[t: t + c]
        lt = float(np.mean(lev[int(t / hop): int((t + c) / hop) + 1]))
        pt = _band_profile(tail, sr)
        for ks in range(int(search_s / 0.5) + 1):
            s = s0 + ks * step
            if t - s < 2 * c:
                break
            head = x[s: s + c]
            ls = float(np.mean(lev[int(s / hop): int((s + c) / hop) + 1]))
            cost = abs(lt - ls) + 0.5 * float(np.mean(np.abs(pt - _band_profile(head, sr))))
            if best is None or cost < best[0]:
                best = (cost, t, s)
    if best is None:
        t, s, cost = end_cut - c, s0, float("nan")
    else:
        cost, t, s = best
    prefix = [x[:t]]
    cycle = [bridge(x[t: t + c], x[s: s + c]), x[s + c: t]]
    info = {"median_level_db": round(med, 1), "restart_s": round(s / sr, 2), "crossfade_start_s": round(t / sr, 2),
            "crossfade_s": round(c / sr, 2), "cycle_s": round((t - s) / sr, 2), "join_cost": round(cost, 2),
            "source_s": round(len(x) / sr, 2)}
    return prefix, cycle, info


def seamless_loop_plan(sources, crossfade_ms=150.0, sr=SR):
    """Alternate one or more loopable recordings (already edge-trimmed) with short equal-power joins."""
    c = int(crossfade_ms / 1000.0 * sr)
    srcs = [s for s in sources if len(s) > 4 * c + 1]
    if not srcs:
        die("ambience source is too short to loop")
    c = max(1, c)
    prefix = [srcs[0][: len(srcs[0]) - c]]
    cycle = []
    for i, cur in enumerate(srcs):
        nxt = srcs[(i + 1) % len(srcs)]
        cycle.append(bridge(cur[len(cur) - c:], nxt[:c]))
        cycle.append(nxt[c: len(nxt) - c])
    info = {"sources": len(srcs), "crossfade_s": round(c / sr, 3),
            "cycle_s": round(sum(len(p) for p in cycle) / sr, 2)}
    return prefix, cycle, info


# ----------------------------------------------------------------------------- script helpers

TAG_RE = re.compile(r"\[([^\[\]]+)\]")
PERMITTED_TAGS = ("whispers", "sighs", "exhales", "inhales deeply")
WORD_RE = re.compile(r"[0-9A-Za-z\u00C0-\u024F\u0370-\uFFFF]+(?:['\u2019\-][0-9A-Za-z\u00C0-\u024F\u0370-\uFFFF]+)*")


def find_tags(text):
    return [m.group(1).strip().lower() for m in TAG_RE.finditer(text or "")]


def strip_tags(text):
    return re.sub(r"\s{2,}", " ", TAG_RE.sub(" ", text or "")).strip()


def words(text):
    return WORD_RE.findall(strip_tags(text))


def count_words(text):
    return len(words(text))


def spoken_mask(text):
    """Boolean per character: True for letters/digits outside [audio tags]."""
    mask = [False] * len(text)
    depth = 0
    for i, ch in enumerate(text):
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth = max(0, depth - 1)
        elif depth == 0 and ch.isalnum():
            mask[i] = True
    return mask


def segment_request(seg, keep_tags):
    """Join a segment's phrases into one request text. Returns (text, [(start, end) per phrase])."""
    parts, spans, pos = [], [], 0
    for i, ph in enumerate(seg["phrases"]):
        t = (ph.get("text") or "").strip()
        if not keep_tags:
            t = strip_tags(t)
        if i:
            parts.append(" ")
            pos += 1
        spans.append((pos, pos + len(t)))
        parts.append(t)
        pos += len(t)
    return "".join(parts), spans


def words_per_minute(script):
    return float((script.get("timing") or {}).get("words_per_minute") or 90)


def session_pads(script):
    s = script.get("session") or {}
    return float(s.get("lead_in_ms", 0) or 0), float(s.get("tail_ms", 0) or 0)


def estimated_timeline(script):
    """Planning timeline from word counts (same shape as voice/timeline.json)."""
    wpm = words_per_minute(script)
    lead, tail = session_pads(script)
    t = lead
    segs = []
    for seg in script.get("segments", []):
        start = t
        phrases = []
        n = len(seg.get("phrases", []))
        for i, ph in enumerate(seg.get("phrases", [])):
            d = count_words(ph.get("text", "")) / wpm * 60000.0
            phrases.append({"index": i, "start_ms": t, "end_ms": t + d})
            t += d
            if i < n - 1:
                t += float(ph.get("pause_after_ms", 0) or 0)
        speech_end = t
        t += float(seg.get("pause_after_ms", 0) or 0)
        segs.append({"id": str(seg.get("id")), "segment_start_ms": start, "speech_end_ms": speech_end,
                     "pause_end_ms": t, "phrases": phrases})
    return {"kind": "estimated", "duration_ms": t + tail, "lead_in_ms": lead, "tail_ms": tail, "segments": segs}


ANCHOR_BOUNDARIES = ("segment_start", "speech_end", "pause_end")
SESSION_BOUNDARIES = ("session_start", "session_end")


def timeline_segment(timeline, seg_id):
    for s in timeline["segments"]:
        if s["id"] == str(seg_id):
            return s
    return None


def resolve_anchor(anchor, timeline):
    """Anchor -> ms on the timeline. Raises KeyError/ValueError on a bad anchor."""
    b = anchor.get("boundary")
    off = float(anchor.get("offset_ms", 0) or 0)
    if b == "session_start":
        return off
    if b == "session_end":
        return float(timeline["duration_ms"]) + off
    if b not in ANCHOR_BOUNDARIES:
        raise ValueError(f"unknown boundary '{b}'")
    seg = timeline_segment(timeline, anchor.get("segment_id"))
    if seg is None:
        raise KeyError(f"unknown segment_id '{anchor.get('segment_id')}'")
    return float(seg[b + "_ms"]) + off


def cue_envelope(initial, cues, timeline):
    """Build an Envelope from music/ambience cues resolved on `timeline`."""
    env = Envelope(initial)
    resolved = []
    for i, c in enumerate(cues or []):
        t0 = resolve_anchor(c["anchor"], timeline)
        resolved.append((t0, i, c))
    for t0, _, c in sorted(resolved, key=lambda r: (r[0], r[1])):
        env.add(t0, t0 + float(c.get("fade_ms", 0) or 0), float(c.get("target_gain", 0)), c.get("id", ""))
    return env


def music_free_windows(script, timeline):
    out = []
    for sid in (script.get("music") or {}).get("music_free_pauses") or []:
        seg = timeline_segment(timeline, sid)
        if seg is not None:
            out.append((float(seg["speech_end_ms"]), float(seg["pause_end_ms"]), str(sid)))
    return out


def duck_envelope(timeline, duck_db, attack_ms=1500.0, release_ms=3000.0, hold_gap_ms=8000.0):
    """Slow reduction across whole spoken passages; held through gaps shorter than hold_gap_ms."""
    env = Envelope(1.0)
    if duck_db >= 0:
        return env
    g = db_to_gain(duck_db)
    regions = [(float(s["segment_start_ms"]), float(s["speech_end_ms"])) for s in timeline["segments"]]
    merged = merge_windows(regions, gap=hold_gap_ms + attack_ms + release_ms)
    for s, e in merged:
        env.add(max(0.0, s - attack_ms), s, g, "duck")
        env.add(e, e + release_ms, 1.0, "release")
    return env


def align_map(request_text, aligned_chars):
    """Map request-text indices to alignment indices (difflib) when they differ. Case is ignored:
    a recipe may send the text lower-cased."""
    aligned = "".join(aligned_chars).lower()
    if aligned == request_text.lower():
        return list(range(len(request_text)))
    mapping = [None] * len(request_text)
    sm = difflib.SequenceMatcher(None, request_text.lower(), aligned, autojunk=False)
    for a, b, size in sm.get_matching_blocks():
        for k in range(size):
            mapping[a + k] = b + k
    return mapping
