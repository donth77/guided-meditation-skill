#!/usr/bin/env python3
# /// script
# requires-python = ">=3.9"
# dependencies = ["numpy"]
# ///
"""
MIX
Build calibrated stems on the measured narration timeline and render the requested versions:
voice (always), voice+music, voice+sfx, voice+music+sfx.

  python3 scripts/mix.py SESSION                                  # versions implied by the script
  python3 scripts/mix.py SESSION --outputs voice,voice+music+sfx
  python3 scripts/mix.py SESSION --outputs all --music-db -18 --ambience-db -16
  python3 scripts/mix.py SESSION --outputs voice+music --music path/to/licensed.wav

Stems (stems/voice.wav, music.wav, sfx.wav) are float32, sample-aligned and already at mix level,
so every version is their plain sum. The voice is normalised to --lufs (measured as dual mono).
Music is high-passed, given a presence dip and gentle peak control, looped with long equal-power
crossfades between level- and spectrum-matched passages (phase continuous through muted
interludes), then shaped by the script's cues, slow passage-level ducking and enforced
music-free pauses. SFX is the ambience loop (fade in, optional cues, light ducking, fade out at
the end) plus one-shots at their anchors. Bed levels are relative to the voice: 1.0 cue gain puts
the music at --music-db and the ambience at --ambience-db below the voice loudness. All versions
share one output gain (lowered for all if any would pass the peak ceiling), so the voice level is
identical in each. Final balance needs listening; these are starting values.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

import numpy as np

from gm_common import (
    BLOCK, SR, Envelope, PieceStream, WavWriter, cue_envelope, db_to_gain, die, duck_envelope, ffmpeg_bin,
    ffmpeg_filter_file, fmt_time, layers_for, level_db, load_script, measure_loudness, merge_windows,
    music_activity, music_free_windows, music_loop_plan, now_iso, open_wav_f32, parse_outputs, probe_duration, read_json,
    resolve_anchor, seamless_loop_plan, session_root, sha256_file, slug_of, trim_digital_silence, variant_slug, warn,
    write_json, write_wav_f32,
)

MUSIC_FILTERS = ["highpass=f=40", "equalizer=f=3000:t=q:w=0.9:g=-3", "highshelf=f=9000:g=-1.5",
                 "acompressor=threshold=0.0794:ratio=2:attack=80:release=1200:knee=4"]
AMBIENCE_FILTERS = ["highpass=f=30"]
ONESHOT_FILTERS = ["highpass=f=40"]
CODECS = {
    ".wav": ["-c:a", "pcm_s24le"],
    ".flac": ["-c:a", "flac"],
    ".m4a": ["-c:a", "aac", "-b:a", "160k", "-movflags", "+faststart"],
}


def loudness_of(path, dualmono=False):
    m = measure_loudness(path, dualmono=dualmono)
    return m["I"] if m["I"] is not None and m["I"] > -69 else None


def rms_loudness(x):
    """Fallback 'loudness' for very short sounds: RMS of the active part in dBFS."""
    lev, _, _ = level_db(x, SR, 400.0, 100.0)
    act = lev[lev > lev.max() - 20]
    return float(np.mean(act)) if len(act) else -70.0


# ----------------------------------------------------------------------------- sources

def resolve_path(root, p):
    q = Path(p).expanduser()
    return q if q.is_absolute() else (root / q if (root / q).exists() else q.resolve())


def music_source(root, script, override):
    if override:
        return resolve_path(root, override), {"file": str(override), "user_supplied": True}
    m = script.get("music") or {}
    if m.get("source_file"):
        return resolve_path(root, m["source_file"]), {"file": m["source_file"], "user_supplied": True}
    sel = read_json(root / "music" / "selection.json", {}) or {}
    name = sel.get("selected")
    if not name:
        die("no music source: run generate_music.py (or pass --music FILE / set music.source_file)")
    meta = read_json(root / "music" / f"{name}.json") or {}
    return root / "music" / meta["audio_file"], {"file": f"music/{meta['audio_file']}", "mock": meta.get("mock"),
                                                 "model_id": meta.get("model_id"), "timed": meta.get("timed")}


def ambience_sources(root, amb, override):
    if override:
        return [(resolve_path(root, p.strip()), False) for p in override.split(",") if p.strip()]
    if amb.get("source_file"):
        return [(resolve_path(root, amb["source_file"]), False)]
    sel = read_json(root / "sfx" / "selection.json", {}) or {}
    names = sel.get("ambience") or []
    if not names:
        die("no ambience source: run generate_sfx.py (or pass --ambience FILE / set sfx.ambience.source_file)")
    out = []
    for n in names:
        meta = read_json(root / "sfx" / f"{n}.json") or {}
        out.append((root / "sfx" / meta["audio_file"], bool(meta.get("loop")), ))
    return out


def one_shot_file(root, sfx, sid, seen=()):
    shots = {s["id"]: s for s in sfx.get("one_shots") or []}
    s = shots.get(sid)
    if s is None:
        die(f"one-shot {sid} not in script")
    if s.get("source_file"):
        return resolve_path(root, s["source_file"])
    if s.get("same_as"):
        if sid in seen:
            die(f"one-shot {sid}: same_as loop")
        return one_shot_file(root, sfx, s["same_as"], seen + (sid,))
    sel = (read_json(root / "sfx" / "selection.json", {}) or {}).get("one_shots", {})
    take = sel.get(sid)
    if not take:
        die(f"one-shot {sid} has no take: run generate_sfx.py --one-shots {sid}")
    meta = read_json(root / "sfx" / "one-shots" / f"{take}.json") or {}
    return root / "sfx" / "one-shots" / meta["audio_file"]


# ----------------------------------------------------------------------------- stems

def write_stream(path, total, channels, block_fn):
    with WavWriter(path, SR, channels) as w:
        for start in range(0, total, BLOCK):
            n = min(BLOCK, total - start)
            w.write(block_fn(start, n))


def build_voice(root, total, gain_db, out):
    voice, _ = open_wav_f32(root / "voice" / "voice-track.wav")
    g = db_to_gain(gain_db)
    n_voice = len(voice)

    def block(start, n):
        blk = np.zeros((n, 1), dtype=np.float32)
        if start < n_voice:
            m = min(n, n_voice - start)
            blk[:m] = voice[start: start + m] * g
        return blk

    write_stream(out, total, 1, block)


def place_music(x, t, s, c, total_s, tl, env, mute, fixed=None, sr=SR):
    """Where in the source the bed starts. The bed plays x[offset:t] once, then loops x[s:t] with a
    crossfade of c samples at each join. Chooses the offset (1 s steps) that puts the source's
    calmest stretches under the narration and its busier moments and loop joins in the rests,
    weighted by how audible the music is (cue gain x music-free muting). Returns a report dict."""
    times, act, events = music_activity(x, sr)
    hop = float(times[1] - times[0]) if len(times) > 1 else 0.5
    L_s = len(x) / sr
    straight = L_s >= total_s          # long enough to play through without a join
    if straight:
        t = len(x)
    t_s, s_s, c_s = t / sr, s / sr, c / sr
    cyc = max(t_s - s_s, c_s + 0.5)
    fr = np.arange(0.0, total_s, 0.5)
    g = np.array([env.value_at(f * 1000.0) * mute.value_at(f * 1000.0) for f in fr])
    speech = np.zeros(len(fr), dtype=bool)
    for sg in tl["segments"]:
        speech |= (fr >= sg["segment_start_ms"] / 1000.0 - 0.5) & (fr <= sg["speech_end_ms"] / 1000.0 + 0.5)
    weight = g * np.where(speech, 1.0, 0.2)

    def mapping(o):
        d0 = t_s - o
        src = np.where(fr < d0, o + fr, s_s + np.mod(fr - d0, cyc))
        joins = [] if d0 >= total_s else [float(j) for j in np.arange(max(d0, 0.0), total_s, cyc)]
        return src, joins

    def act_at(src):
        return act[np.clip(np.round((src - times[0]) / hop).astype(int), 0, len(act) - 1)]

    def cost(o):
        src, joins = mapping(o)
        total = float(np.sum(weight * act_at(src)))
        for j in joins:
            m = (fr >= j) & (fr < j + c_s)
            if m.any() and g[m].max() > 0.05:
                total += 6.0 * float(g[m].max()) * (1.0 + 3.0 * float(speech[m].mean()))
        return total

    top = (L_s - total_s) if straight else (t_s - 1.0)
    candidates = [float(fixed)] if fixed is not None else list(np.arange(0.0, max(0.5, top + 1e-9), 1.0))
    o = min(candidates, key=cost)
    src, joins = mapping(o)
    a = act_at(src)
    segs = []
    for sg in tl["segments"]:
        m = (fr >= sg["segment_start_ms"] / 1000.0) & (fr <= sg["speech_end_ms"] / 1000.0) & (g > 0.05)
        if m.any():
            v = float(a[m].mean())
            segs.append({"id": sg["id"], "activity": round(v, 2),
                         "feel": "calm" if v < 0.35 else "moderate" if v < 0.6 else "busier"})
    under = []
    for e in events:
        m = (np.abs(src - e) < hop) & speech & (g > 0.05)
        if m.any():
            under.append(round(float(fr[m][0]), 1))
    return {"offset_s": round(float(o), 1), "source_s": round(len(x) / sr, 1),
            "joins": [{"at_s": round(j, 1), "under_speech": bool(speech[(fr >= j) & (fr < j + c_s)].any()),
                       "music_gain": round(float(g[(fr >= j) & (fr < j + c_s)].max() if ((fr >= j) & (fr < j + c_s)).any() else 0), 2)}
                      for j in joins],
            "segments": segs, "events_under_speech_s": sorted(set(under)), "cost": round(cost(o), 1),
            "chosen": "fixed" if fixed is not None else "auto", "straight_through": straight}


def build_music(root, script, tl, total, args, work, report):
    music = script["music"]
    src, info = music_source(root, script, args.music)
    pre = ffmpeg_filter_file(src, work / "music_pre.wav", [] if args.no_music_eq else MUSIC_FILTERS, 2)
    x = np.array(open_wav_f32(pre)[0])
    fixed = None if str(args.music_offset).lower() == "auto" else float(args.music_offset)
    if fixed is None and info.get("timed"):
        fixed = 0.0         # fit_music.py output: already on the session clock
    if fixed is None:
        x = trim_digital_silence(x, -70.0)       # a fixed offset means the file is aligned: keep its head
    xf = float((music.get("looping") or {}).get("crossfade_ms", 12000))
    prefix, cycle, loop = music_loop_plan(x, crossfade_ms=xf)
    write_wav_f32(work / "music_cycle.wav", np.concatenate(cycle))
    lm = loudness_of(work / "music_cycle.wav")
    if lm is None:
        die("music source is too quiet to measure")
    gain_db = (args.lufs + args.music_db) - lm
    env = cue_envelope(float(music.get("initial_gain", 0)), music.get("cues") or [], tl)
    duck = duck_envelope(tl, args.duck_db)
    mute = Envelope(1.0)
    bad = []
    for a, b, sid in music_free_windows(script, tl):
        peak = env.max_in(a, b)
        if peak > 1e-4:
            bad.append((a, b))
            report["warnings"].append(f"music cue plan leaves gain {peak:.2f} in the music-free pause after {sid} on "
                                      "the measured clock; muted there with 2.5 s fades")
    for a, b in merge_windows(bad, gap=5000.0):
        mute.add(max(0.0, a - 2500.0), a, 0.0, "music-free")
        mute.add(b, b + 2500.0, 1.0, "music-free end")
    report["warnings"] += [f"music: {n}" for n in env.notes]
    t_end = len(prefix[0])
    c_len = len(cycle[0])
    s_start = t_end - (len(cycle[0]) + len(cycle[1]))
    place = place_music(x, t_end, s_start, c_len, total / SR, tl, env, mute, fixed)
    o = int(round(place["offset_s"] * SR))
    if place["straight_through"]:
        stream = PieceStream([x[o:]], [np.zeros((SR, 2), dtype=np.float32)], 2)
    else:
        stream = PieceStream([x[o:t_end]] if o < t_end else [], cycle, 2)
    g = db_to_gain(gain_db)
    write_stream(root / "stems" / "music.wav", total, 2,
                 lambda s, n: stream.read(n) * (env.block(s, n) * duck.block(s, n) * mute.block(s, n) * g)[:, None])
    report["music"] = {**info, "sha256": sha256_file(src), "loop": loop, "bed_loudness_lufs": lm,
                       "calibration_gain_db": round(gain_db, 2), "level_rel_voice_db": args.music_db,
                       "duck_db": args.duck_db, "eq": not args.no_music_eq,
                       "cues": [{"id": r["label"], "start_s": round(r["t0"] / 1000, 2),
                                 "end_s": round(min(r["t1"], r["cut"]) / 1000, 2), "target": r["g1"]}
                                for r in env.ramps],
                       "placement": place}
    if info.get("mock"):
        report["mock"] = True


def build_sfx(root, script, tl, total, args, work, report):
    sfx = script["sfx"]
    end_ms = tl["duration_ms"]
    amb = sfx.get("ambience") or {}
    amb_stream = amb_env = amb_duck = None
    g_amb = 0.0
    if amb.get("enabled"):
        loops, seamless, files = [], True, []
        for i, (path, is_loop) in enumerate(ambience_sources(root, amb, args.ambience)):
            pre = ffmpeg_filter_file(path, work / f"amb_pre_{i}.wav", AMBIENCE_FILTERS, 2)
            loops.append(trim_digital_silence(np.array(open_wav_f32(pre)[0]), -70.0))
            seamless &= is_loop
            files.append({"file": str(path), "sha256": sha256_file(path), "generated_loop": is_loop})
        xf = args.ambience_xfade_ms or (150.0 if seamless else 3000.0)
        prefix, cycle, loop = seamless_loop_plan(loops, crossfade_ms=xf)
        write_wav_f32(work / "amb_cycle.wav", np.concatenate(cycle))
        la = loudness_of(work / "amb_cycle.wav")
        if la is None:
            die("ambience source is too quiet to measure")
        g_db = (args.lufs + args.ambience_db) - la
        g_amb = db_to_gain(g_db)
        ramps = [(0.0, float(amb.get("fade_in_ms", 0) or 0), 1.0, "fade-in")]
        for c in amb.get("cues") or []:
            t0 = resolve_anchor(c["anchor"], tl)
            ramps.append((t0, t0 + float(c.get("fade_ms", 0)), float(c.get("target_gain", 0)), c.get("id", "")))
        fo = float(amb.get("fade_out_ms", 0) or 0)
        ramps.append((max(0.0, end_ms - fo), end_ms, 0.0, "fade-out"))
        amb_env = Envelope(float(amb.get("initial_gain", 0) or 0))
        for t0, t1, g, label in sorted(ramps, key=lambda r: r[0]):
            amb_env.add(t0, t1, g, label)
        report["warnings"] += [f"ambience: {n}" for n in amb_env.notes]
        amb_duck = duck_envelope(tl, args.ambience_duck_db)
        amb_stream = PieceStream(prefix, cycle, 2)
        report["ambience"] = {"sources": files, "loop": loop, "loudness_lufs": la, "calibration_gain_db": round(g_db, 2),
                              "level_rel_voice_db": args.ambience_db, "duck_db": args.ambience_duck_db}
    shots = []
    for s in sfx.get("one_shots") or []:
        path = one_shot_file(root, sfx, s["id"])
        pre = ffmpeg_filter_file(path, work / f"shot_{s['id']}.wav", ONESHOT_FILTERS, 2)
        a = np.array(open_wav_f32(pre)[0])
        a = trim_digital_silence(a, -60.0)
        if len(a) > int(0.03 * SR):
            a[-int(0.02 * SR):] *= np.linspace(1.0, 0.0, int(0.02 * SR), dtype=np.float32)[:, None]
        loud = loudness_of(pre)
        loud = loud if loud is not None else rms_loudness(a)
        g = db_to_gain(args.lufs + args.sfx_db - loud) * float(s.get("gain", 1.0))
        t = resolve_anchor(s["anchor"], tl)
        start = int(round(max(0.0, t) / 1000 * SR))
        shots.append((start, a * g))
        report.setdefault("one_shots", []).append({"id": s["id"], "file": str(path), "start_s": round(start / SR, 2),
                                                    "duration_s": round(len(a) / SR, 2), "gain": s.get("gain", 1.0)})

    def block(start, n):
        out = np.zeros((n, 2), dtype=np.float32)
        if amb_stream is not None:
            g = amb_env.block(start, n) * amb_duck.block(start, n) * g_amb
            out += amb_stream.read(n) * g[:, None]
        for s0, a in shots:
            a0, a1 = max(start, s0), min(start + n, s0 + len(a))
            if a0 < a1:
                out[a0 - start: a1 - start] += a[a0 - s0: a1 - s0]
        return out

    write_stream(root / "stems" / "sfx.wav", total, 2, block)


def one_shot_end_frames(root, script, tl):
    sfx = script.get("sfx") or {}
    end = 0
    for s in sfx.get("one_shots") or []:
        try:
            t = resolve_anchor(s["anchor"], tl)
        except (KeyError, ValueError):
            continue
        dur = probe_duration(one_shot_file(root, sfx, s["id"])) or float(s.get("duration_seconds", 5))
        end = max(end, int(round((max(0.0, t) / 1000 + dur + 0.5) * SR)))
    return end


# ----------------------------------------------------------------------------- rendering

def graph_for(n_stereo_layers):
    if n_stereo_layers == 0:
        return None
    n = 1 + n_stereo_layers
    left, right = ["c0"], ["c0"]
    for k in range(n_stereo_layers):
        left.append(f"c{1 + 2 * k}")
        right.append(f"c{2 + 2 * k}")
    ins = "".join(f"[{i}:a]" for i in range(n))
    return f"{ins}amerge=inputs={n},pan=stereo|c0={'+'.join(left)}|c1={'+'.join(right)}"


def encode(inputs, graph, gain_db, out, title, mono, limit_db=None):
    cmd = [ffmpeg_bin(), "-v", "error", "-nostdin", "-y"]
    for i in inputs:
        cmd += ["-i", str(i)]
    vol = f"volume={gain_db:.3f}dB"
    if limit_db is not None:
        # 4x oversampled lookahead limiter: sample peaks at 176.4 kHz approximate true peak.
        vol += (f",aresample={SR * 4},alimiter=limit={db_to_gain(limit_db):.4f}:attack=5:release=80:level=0:"
                f"latency=1,aresample={SR}")
    if graph:
        cmd += ["-filter_complex", f"{graph},{vol}[out]", "-map", "[out]"]
    else:
        cmd += ["-af", vol]
    ext = out.suffix.lower()
    if ext == ".mp3":
        cmd += ["-c:a", "libmp3lame", "-b:a", "128k" if mono else "192k"]
    elif ext in CODECS:
        cmd += CODECS[ext]
    else:
        die(f"unsupported format {ext} (wav, mp3, m4a, flac)")
    cmd += ["-ar", str(SR), "-metadata", f"title={title}", "-metadata", "comment=Guided meditation made with ElevenLabs",
            str(out)]
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if p.returncode != 0:
        die(f"ffmpeg encode failed for {out.name}: {p.stderr.strip()[:400]}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("session", help="session folder (or its script.json)")
    ap.add_argument("--outputs", help="voice, voice+music, voice+sfx, voice+music+sfx or all (voice is always made)")
    ap.add_argument("--formats", default="wav,mp3", help="comma list of wav, mp3, m4a, flac (default wav,mp3)")
    ap.add_argument("--lufs", type=float, default=-18.0, help="loudness target, LUFS (default -18)")
    ap.add_argument("--normalize", choices=("voice", "integrated"), default="voice",
                    help="voice: the voice sits at --lufs in every version (default); integrated: each version's "
                         "whole-track loudness is brought to --lufs")
    ap.add_argument("--peak", type=float, default=-1.5, help="true-peak ceiling, dBTP (default -1.5)")
    ap.add_argument("--limit", action="store_true",
                    help="catch peaks over the ceiling with an oversampled limiter instead of lowering the gain")
    ap.add_argument("--music-db", type=float, default=-16.0, help="music bed at cue gain 1.0, dB relative to the voice")
    ap.add_argument("--ambience-db", type=float, default=-18.0, help="ambience bed, dB relative to the voice")
    ap.add_argument("--sfx-db", type=float, default=-10.0, help="one-shot loudness, dB relative to the voice")
    ap.add_argument("--duck-db", type=float, default=-3.0, help="music reduction across spoken passages (0 = off)")
    ap.add_argument("--ambience-duck-db", type=float, default=-1.5, help="ambience reduction across speech (0 = off)")
    ap.add_argument("--no-music-eq", action="store_true", help="skip the music high-pass, presence dip and peak control")
    ap.add_argument("--ambience-xfade-ms", type=float, help="join crossfade for the ambience loop")
    ap.add_argument("--music-offset", default="auto", help="seconds into the music source where the bed starts; "
                    "'auto' (default) puts calm stretches under the voice and busier moments and loop joins in rests")
    ap.add_argument("--music", help="use this music file instead of the selected/generated one")
    ap.add_argument("--ambience", help="use these ambience files (comma list) instead of the selected ones")
    ap.add_argument("--keep-work", action="store_true", help="keep stems/.work intermediate files")
    ap.add_argument("--tag", help="suffix for this mix's files (e.g. suno -> <slug>.voice-music.suno.mp3), so "
                    "versions with different music sit side by side")
    args = ap.parse_args()

    root = session_root(args.session)
    script = load_script(root)
    timeline = read_json(root / "voice" / "timeline.json")
    if not timeline:
        die("voice/timeline.json not found: run assemble_voice.py first")
    variants = parse_outputs(args.outputs, script)
    need = layers_for(variants)
    if "music" in need and not (script.get("music") or {}).get("enabled") and not args.music:
        die("a music version was requested but music.enabled is false in script.json")
    if "sfx" in need and not (script.get("sfx") or {}).get("enabled") and not args.ambience:
        die("an sfx version was requested but sfx.enabled is false in script.json")
    formats = [f.strip().lower().lstrip(".") for f in args.formats.split(",") if f.strip()]
    slug = slug_of(script, root)
    title = script.get("title") or slug
    stems, work, outdir = root / "stems", root / "stems" / ".work", root / "output"
    tag = f".{args.tag}" if args.tag else ""
    work.mkdir(parents=True, exist_ok=True)
    outdir.mkdir(exist_ok=True)
    report = {"warnings": [], "mock": bool(timeline.get("mock"))}

    voice_path = root / "voice" / "voice-track.wav"
    n_voice = len(open_wav_f32(voice_path)[0])
    total = n_voice
    if "sfx" in need:
        total = max(total, one_shot_end_frames(root, script, timeline))
        if total > n_voice:
            report["warnings"].append(f"track extended by {(total - n_voice) / SR:.1f}s so a one-shot can decay")
    tl = dict(timeline, duration_ms=total / SR * 1000)

    lv = loudness_of(voice_path, dualmono=True)
    if lv is None:
        die("the voice track is silent")
    gv = args.lufs - lv
    build_voice(root, total, gv, stems / "voice.wav")
    report["voice"] = {"loudness_lufs": lv, "gain_db": round(gv, 2), "sha256": timeline.get("voice_track_sha256")}
    if "music" in need:
        build_music(root, script, tl, total, args, work, report)
    if "sfx" in need:
        build_sfx(root, script, tl, total, args, work, report)

    # Measure every version, then pick one shared gain that keeps all under the ceiling.
    plan = {}
    for v in variants:
        layers = [stems / "voice.wav"] + [stems / f"{l}.wav" for l in ("music", "sfx") if l in v.split("+")]
        graph = graph_for(len(layers) - 1)
        m = measure_loudness(layers[0], dualmono=True) if graph is None else \
            measure_loudness(inputs=layers, filter_complex=graph)
        plan[v] = (layers, graph, m)
    def finite(x):
        return x is not None and x not in (float("inf"), float("-inf"))

    gains = {}
    limit_db = args.peak if args.limit else None
    if args.normalize == "voice":
        tps = [m["TP"] for _, _, m in plan.values() if finite(m["TP"])]
        shared = min(0.0, args.peak - max(tps)) if tps and not args.limit else 0.0
        if shared < 0:
            report["warnings"].append(f"all versions lowered {abs(shared):.1f} dB to stay under {args.peak} dBTP "
                                      "(--limit catches the peaks instead)")
        elif args.limit and tps and max(tps) > args.peak:
            report["warnings"].append(f"limiter active: peaks up to {max(tps) - args.peak:.1f} dB over {args.peak} "
                                      "dBTP are limited")
        gains = {v: shared for v in plan}
    else:
        for v, (_, _, m) in plan.items():
            g = args.lufs - m["I"] if finite(m["I"]) else 0.0
            if finite(m["TP"]) and m["TP"] + g > args.peak and not args.limit:
                report["warnings"].append(f"{v}: held {m['TP'] + g - args.peak:.1f} dB under the integrated target "
                                          f"by the {args.peak} dBTP ceiling")
                g = args.peak - m["TP"]
            gains[v] = g
    manifest_variants = {}
    for v, (layers, graph, m) in plan.items():
        g = gains[v]
        files = []
        for fmt in formats:
            out = outdir / f"{slug}.{variant_slug(v)}{tag}.{fmt}"
            encode(layers, graph, g, out, f"{title} ({v.replace('+', ' + ')})", graph is None, limit_db)
            files.append(str(out.relative_to(root)))
        tp = m["TP"] + g if finite(m["TP"]) else None
        manifest_variants[v] = {
            "files": files, "voice_lufs": round(args.lufs + g, 1), "output_gain_db": round(g, 2),
            "integrated_lufs": round(m["I"] + g, 1) if finite(m["I"]) else None,
            "true_peak_dbtp": None if tp is None else round(min(tp, args.peak) if args.limit else tp, 1),
            "limited": bool(args.limit and tp is not None and tp > args.peak),
            "loudness_range_lu": m["LRA"], "channels": 1 if graph is None else 2}
    if not args.keep_work:
        shutil.rmtree(work, ignore_errors=True)
    manifest = {"title": title, "slug": slug, "created_at": now_iso(), "duration_s": round(total / SR, 2),
                "sample_rate": SR, "target_lufs": args.lufs, "normalize": args.normalize,
                "peak_ceiling_dbtp": args.peak, "variants": manifest_variants,
                "stems": {k: f"stems/{k}.wav" for k in ["voice"] + sorted(need)},
                "settings": {"music_db": args.music_db, "ambience_db": args.ambience_db, "sfx_db": args.sfx_db,
                             "duck_db": args.duck_db, "ambience_duck_db": args.ambience_duck_db,
                             "music_eq": not args.no_music_eq}, **report}
    write_json(outdir / f"manifest{tag}.json", manifest)

    print(f"{title}: {fmt_time(total / SR)}, voice stem normalised {gv:+.1f} dB to {args.lufs} LUFS; "
          f"normalize={args.normalize}")
    for v, d in manifest_variants.items():
        print(f"  {v:<17} voice {d['voice_lufs']} LUFS | integrated {d['integrated_lufs']} LUFS | TP "
              f"{d['true_peak_dbtp']} dBTP | LRA {d['loudness_range_lu']}  -> {', '.join(d['files'])}")
    if args.normalize == "voice" and len(manifest_variants) > 1:
        print("  (integrated loudness is lower where quiet beds fill the rests; the voice level is the same in "
              "every version. Use --normalize integrated for platforms that level whole tracks.)")
    if "music" in report:
        lp = report["music"]["loop"]
        print(f"  music loop: restart {lp['restart_s']}s, crossfade {lp['crossfade_s']}s at {lp['crossfade_start_s']}s, "
              f"cycle {lp['cycle_s']}s (join cost {lp['join_cost']})")
        pl = report["music"].get("placement") or {}
        if pl:
            fmt = lambda v: f"{int(v // 60)}:{v % 60:04.1f}"  # noqa: E731
            joins = ", ".join(f"{fmt(j['at_s'])} ({'UNDER SPEECH' if j['under_speech'] else 'in a rest'})"
                              for j in pl["joins"] if j["music_gain"] > 0.05) or "none audible"
            feel = ", ".join(f"{sg['id']} {sg['feel']}" for sg in pl["segments"])
            print(f"  music placement ({pl['chosen']}): bed starts {pl['offset_s']}s into the {pl['source_s']}s source; "
                  f"loop joins {joins}")
            print(f"  music under each passage: {feel}"
                  + (f"; note events under speech at {', '.join(fmt(e) for e in pl['events_under_speech_s'])}"
                     if pl["events_under_speech_s"] else ""))
    if report["mock"]:
        print("MOCK audio in this mix: rehearsal only, not a deliverable")
    for w in report["warnings"]:
        warn(w)
    print(f"manifest -> output/manifest{tag}.json; stems -> stems/")


if __name__ == "__main__":
    main()
