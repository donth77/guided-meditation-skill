#!/usr/bin/env python3
# /// script
# requires-python = ">=3.9"
# dependencies = ["numpy"]
# ///
"""
VOICES
Find narrator candidates: voices in the account, the shared Voice Library, voice details and
previews, and the models each professional voice is fine-tuned for. Read-only except `add`.

  python3 scripts/voices.py mine
  python3 scripts/voices.py suggest --accent british --gender female            # rank account voices
  python3 scripts/voices.py suggest --accent american --library                 # include the library
  python3 scripts/voices.py library --search meditation --gender female --accent british
  python3 scripts/voices.py similar SESSION/voice/audition/take-0005.mp3 --accent british --measure
  python3 scripts/voices.py show VOICE_ID
  python3 scripts/voices.py preview VOICE_ID --out SESSION/voice/previews
  python3 scripts/voices.py suggest --gender female --library --measure --out SESSION/voice/previews
  python3 scripts/voices.py library --search calm --accent british --voiced 70-80 --page-size 50
  python3 scripts/voices.py add PUBLIC_OWNER_ID VOICE_ID --name "Willow"       # optional; uses a voice slot: ask first

Voice Library voices work in text to speech directly by id; adding one is not required.

`similar` finds Voice Library voices that sound like an audio sample (free): the way to keep a
voice's sound when its delivery fails (for example it pauses inside phrases whatever the settings).

--measure downloads each preview (free) and prints how much of its speech is voiced: near 0%
is a full whisper, 30-45% a heavy breathy whisper, 50-65% soft and breathy, ~80% ordinary
speech. It screens candidates for whisper strength; previews use other text and settings, so
audition the shortlist before choosing. --voiced MIN-MAX (implies --measure) keeps only voices
whose preview falls in that band; references/voice-direction.md maps breath levels to bands.

A library preview is the voice owner's showcase, recorded with unknown settings and model; it is a
reference for character, not a promise of what your script will sound like. Audition instead.
"""
from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path

from gm_common import SR, ApiError, api_json, api_key, api_request, decode_audio, die, fail_api, multipart, voicing_ratio

KEYWORDS = {"meditat": 3, "calm": 3, "whisper": 2, "sooth": 2, "gentle": 2, "relax": 2, "sleep": 2, "tranquil": 2,
            "asmr": 1, "soft": 1, "warm": 1, "breathy": 1, "nurtur": 1, "narrat": 1, "peace": 1, "slow": 1}


def fetch_preview(url, out_dir, voice_id):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{voice_id}-preview.mp3"
    if not path.exists():
        with urllib.request.urlopen(url, timeout=60) as r:
            path.write_bytes(r.read())
    return path


def measure(v, out_dir):
    """Voicing share of a voice's preview, or None when it has no preview."""
    url = v.get("preview_url")
    if not url:
        return None
    try:
        return voicing_ratio(decode_audio(fetch_preview(url, out_dir, v["voice_id"]), SR, 1)[:, 0])
    except (OSError, SystemExit) as e:
        print(f"  (could not measure {v.get('name')}: {e})")
        return None


def screened(vs, args):
    """(voice, voiced share) pairs to print: measured with --measure, filtered by --voiced."""
    lo, hi = 0.0, 1.0
    if args.voiced:
        a, _, b = args.voiced.partition("-")
        try:
            lo, hi = float(a) / 100, float(b or 100) / 100
        except ValueError:
            die(f"--voiced takes a percent band like 65-80, not {args.voiced!r}")
    out = []
    for v in vs:
        vr = measure(v, args.out) if args.measure or args.voiced else None
        if args.voiced and (vr is None or not lo <= vr <= hi):
            continue
        out.append((v, vr))
    return out


def label_text(v):
    labels = v.get("labels") or {}
    parts = [v.get("name") or "", v.get("description") or "", " ".join(str(x) for x in labels.values())]
    for k in ("accent", "gender", "age", "descriptive", "use_case", "category"):
        if v.get(k):
            parts.append(str(v[k]))
    return " ".join(parts).lower()


def fine_tuned(v):
    state = ((v.get("fine_tuning") or {}).get("state")) or {}
    models = {m for m, s in state.items() if s == "fine_tuned"} | set(v.get("high_quality_base_model_ids") or [])
    return sorted(m for m in models if m.startswith("eleven_"))


def my_voices(key, voice_type=None):
    out, token = [], None
    while True:
        q = {"page_size": 100, "next_page_token": token}
        if voice_type:
            q["voice_type"] = voice_type
        data, _ = api_json("GET", "/v2/voices", key=key, query=q)
        out += data.get("voices") or []
        token = data.get("next_page_token")
        if not data.get("has_more") or not token:
            return out


def library(key, **filters):
    q = {k: v for k, v in filters.items() if v not in (None, "")}
    q.setdefault("page_size", 20)
    data, _ = api_json("GET", "/v1/shared-voices", key=key, query=q)
    return data.get("voices") or []


def similar(key, path, top):
    """Voice Library voices that sound like the audio at `path`, most similar first."""
    p = Path(path)
    if not p.is_file():
        die(f"{path} not found")
    body, ctype = multipart({"top_k": top}, {"audio_file": (p.name, p.read_bytes(),
                                                           "audio/mpeg" if p.suffix == ".mp3" else "audio/wav")})
    raw, _ = api_request("POST", "/v1/similar-voices", key=key, raw=body, content_type=ctype, retries=2)
    return json.loads(raw.decode("utf-8")).get("voices") or []


def matches(v, args):
    text = label_text(v)
    labels = v.get("labels") or {}
    for field in ("accent", "gender", "age"):
        want = getattr(args, field, None)
        if want:
            have = str(v.get(field) or labels.get(field) or "").lower()
            if want.lower() not in have and want.lower() not in text:
                return False
    return True


def score(v):
    text = label_text(v)
    return sum(w for k, w in KEYWORDS.items() if k in text)


def row(v, lib=False, voicing=None):
    labels = v.get("labels") or {}
    get = lambda k: v.get(k) or labels.get(k) or "-"  # noqa: E731
    vc = "" if voicing is None else f"voiced {voicing * 100:3.0f}%  "
    base = (f"{vc}{v.get('voice_id')}  {v.get('name', '')[:44]:<44} {get('accent'):<14} {get('gender'):<7} "
            f"{get('age'):<12} {get('descriptive'):<12} {get('use_case')}")
    if lib:
        base += f"  owner {v.get('public_owner_id')}  used {v.get('usage_character_count_1y') or 0:,} chars/yr"
    else:
        ft = fine_tuned(v)
        base += f"  [{v.get('category')}]" + (f" fine-tuned: {', '.join(ft)}" if ft and v.get("category") == "professional" else "")
    return base


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("mine", help="voices in the account")
    m.add_argument("--type", help="personal, community, default, workspace, non-default, saved")
    s = sub.add_parser("suggest", help="rank voices for meditation narration")
    for p in (s,):
        p.add_argument("--accent")
        p.add_argument("--gender")
        p.add_argument("--age")
        p.add_argument("--library", action="store_true", help="also search the shared Voice Library")
        p.add_argument("--top", type=int, default=8)
        p.add_argument("--measure", action="store_true", help="download previews and measure whisper strength")
        p.add_argument("--voiced", metavar="MIN-MAX", help="keep previews in this voiced band, e.g. 65-80")
        p.add_argument("--out", default="voice-previews", help="where --measure keeps previews")
    lb = sub.add_parser("library", help="search the shared Voice Library")
    lb.add_argument("--search")
    lb.add_argument("--gender")
    lb.add_argument("--age")
    lb.add_argument("--accent")
    lb.add_argument("--language")
    lb.add_argument("--use-cases")
    lb.add_argument("--descriptives")
    lb.add_argument("--category", help="professional, famous, high_quality")
    lb.add_argument("--page-size", type=int, default=20)
    lb.add_argument("--measure", action="store_true", help="download previews and measure whisper strength")
    lb.add_argument("--voiced", metavar="MIN-MAX", help="keep previews in this voiced band, e.g. 65-80")
    lb.add_argument("--out", default="voice-previews", help="where --measure keeps previews")
    sm = sub.add_parser("similar", help="Voice Library voices that sound like an audio sample (free)")
    sm.add_argument("audio", help="a take or recording of the voice whose sound you want to keep")
    sm.add_argument("--accent")
    sm.add_argument("--gender")
    sm.add_argument("--age")
    sm.add_argument("--top", type=int, default=15)
    sm.add_argument("--measure", action="store_true", help="download previews and measure whisper strength")
    sm.add_argument("--voiced", metavar="MIN-MAX", help="keep previews in this voiced band, e.g. 65-80")
    sm.add_argument("--out", default="voice-previews", help="where --measure keeps previews")
    sh = sub.add_parser("show", help="details for one voice")
    sh.add_argument("voice_id")
    pv = sub.add_parser("preview", help="download a voice's preview audio")
    pv.add_argument("voice_id")
    pv.add_argument("--url", help="preview URL (for library voices, from `library` --json)")
    pv.add_argument("--out", default=".", help="folder (default: current)")
    ad = sub.add_parser("add", help="add a library voice to the account (uses a voice slot)")
    ad.add_argument("public_owner_id")
    ad.add_argument("voice_id")
    ad.add_argument("--name", required=True)
    for p in (m, s, lb, sm, sh):
        p.add_argument("--json", action="store_true")
    args = ap.parse_args()
    key = api_key()

    try:
        if args.cmd == "mine":
            vs = my_voices(key, args.type)
            if args.json:
                return print(json.dumps(vs, indent=2))
            for v in vs:
                print(row(v))
        elif args.cmd == "suggest":
            vs = [v for v in my_voices(key) if matches(v, args)]
            ranked = sorted(vs, key=score, reverse=True)[: args.top]
            lib = []
            if args.library:
                lib = library(key, search="meditation", gender=args.gender, age=args.age, accent=args.accent,
                              page_size=args.top)
            if args.json:
                return print(json.dumps({"account": ranked, "library": lib}, indent=2))
            print("Account voices (best matches first):")
            for v, vr in screened(ranked, args):
                print(f"  {score(v):>2}  {row(v, voicing=vr)}")
            if lib:
                print("Voice Library (search 'meditation'):")
                for v, vr in screened(lib, args):
                    print(f"      {row(v, lib=True, voicing=vr)}")
            if args.measure or args.voiced:
                print("voiced share of each preview: ~0% full whisper, 30-45% heavy breathy whisper, 50-65% soft "
                      "and breathy, ~80% ordinary speech (screening only; previews use other text)")
            print("Download previews with `preview`, then audition the script's opening with synthesize.py --audition.")
        elif args.cmd == "library":
            vs = library(key, search=args.search, gender=args.gender, age=args.age, accent=args.accent,
                         language=args.language, use_cases=args.use_cases, descriptives=args.descriptives,
                         category=args.category, page_size=args.page_size)
            if args.json:
                return print(json.dumps(vs, indent=2))
            kept = screened(vs, args)
            for v, vr in kept:
                print(row(v, lib=True, voicing=vr))
            if args.voiced:
                print(f"{len(kept)} of {len(vs)} voices in the {args.voiced}% voiced band (raise --page-size for more)")
        elif args.cmd == "similar":
            vs = [v for v in similar(key, args.audio, args.top) if matches(v, args)]
            if args.json:
                return print(json.dumps(vs, indent=2))
            for v, vr in screened(vs, args):
                print(row(v, lib=True, voicing=vr))
            print("Most similar first. Audition two or three on the script's opening: a similar sound does not mean "
                  "a similar delivery.")
        elif args.cmd == "show":
            try:
                v, _ = api_json("GET", f"/v1/voices/{args.voice_id}", key=key)
            except ApiError as e:
                if "not_found" not in e.text:
                    raise
                lib = [x for x in library(key, search=args.voice_id, page_size=5) if x.get("voice_id") == args.voice_id]
                if not lib:
                    die(f"{args.voice_id} is neither in the account nor in the Voice Library")
                v = lib[0]
                if args.json:
                    return print(json.dumps(v, indent=2))
                print(f"{v.get('name')} ({v.get('voice_id')}) [Voice Library; usable by id without adding]")
                print(f"{v.get('accent')} {v.get('gender')} {v.get('age')}, {v.get('descriptive')}, {v.get('use_case')}")
                if v.get("description"):
                    print(f"description: {v['description']}")
                print(f"preview: {v.get('preview_url')}")
                return
            if args.json:
                return print(json.dumps(v, indent=2))
            print(f"{v.get('name')} ({v.get('voice_id')}) [{v.get('category')}]")
            print(f"labels: {json.dumps(v.get('labels'))}")
            if v.get("description"):
                print(f"description: {v['description']}")
            print(f"saved settings: {json.dumps(v.get('settings'))}")
            ft = fine_tuned(v)
            if ft:
                print(f"fine-tuned models: {', '.join(ft)}")
            for lang in v.get("verified_languages") or []:
                print(f"verified: {lang.get('language')} {lang.get('accent') or ''} {lang.get('locale') or ''} on "
                      f"{lang.get('model_id')}")
            print(f"preview: {v.get('preview_url')}")
        elif args.cmd == "preview":
            url = args.url
            if not url:
                v, _ = api_json("GET", f"/v1/voices/{args.voice_id}", key=key)
                url = v.get("preview_url")
            if not url:
                die("no preview URL for this voice")
            path = fetch_preview(url, args.out, args.voice_id)
            vr = voicing_ratio(decode_audio(path, SR, 1)[:, 0])
            print(f"saved {path}  (voiced {vr * 100:.0f}% of speech; ~0% full whisper, ~80% ordinary speech)")
        elif args.cmd == "add":
            data, _ = api_json("POST", f"/v1/voices/add/{args.public_owner_id}/{args.voice_id}", key=key,
                               body={"new_name": args.name})
            print(f"added as {data.get('voice_id')} ('{args.name}'); it now uses one of the account's voice slots")
    except ApiError as e:
        fail_api(e, args.cmd)


if __name__ == "__main__":
    main()
