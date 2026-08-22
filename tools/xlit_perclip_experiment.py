#!/usr/bin/env python3
"""Step 4b experiment: PER-CLIP script correction with Claude Sonnet 4.6.

Isolated from the shipped pipeline. Nothing here writes under `asr/`, `results/`
or `checkpoints/`; output goes to `local_out/experiments/xlit_perclip/`.

WHAT THIS TESTS. The shipped Step 4b builds one lookup table from the Latin
vocabulary of ALL 99 clips, which makes it transductive: the table's coverage was
fixed with the test clips' hypotheses in view. This experiment asks whether the
same idea survives when the model may see ONE CLIP AT A TIME and nothing else.

INFORMATION AVAILABLE TO THE MODEL, exhaustively:
  - the Latin-script tokens appearing in THIS clip's own hypothesis
  - the dominant script of THIS clip's own hypothesis
Nothing from any other clip. Nothing from the reference, ever. Nothing from the
split. The reference is opened only after all inference is finished, by the
scoring script, to compute metrics.

SCOPE. Rewrites the script of an already-recognised token and nothing else. A
token either maps to a single whitespace-free Indic rendering or is left byte
identical, so the token count cannot change and words cannot be added, deleted,
reordered or translated. Numerals are deliberately EXCLUDED: spelling a digit
out changes the token count, which this experiment's scope forbids.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import unicodedata
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sarvam_diar import asr, data, translit  # noqa: E402
from sarvam_diar.config import Config  # noqa: E402
from sarvam_diar.reference import normalize_text  # noqa: E402
from sarvam_diar.utils import read_json, write_json_atomic  # noqa: E402

# ------------------------------------------------------------------ frozen config
MODEL = "us.anthropic.claude-sonnet-4-6"
TEMPERATURE = 0
MAX_TOKENS = 8000
BATCH = 60          # v2: cap words per request so the reply cannot be truncated
PROMPT_VERSION = "xlit-perclip-v1"   # prompt text UNCHANGED from v1
SOURCE = "sarvam-saaras-v3@fusion"
OUT_DIR = Path("local_out/experiments/xlit_perclip")

SYSTEM = """\
You convert English words that were spoken inside {lang} speech into {lang} script.

You are given words from a single {lang} transcript that the recogniser wrote in
Latin letters. For each one, decide whether it is simply the WRONG SCRIPT for a
word that was already recognised correctly. If it is, write that same word the
way an ordinary {lang} transcriber writes an English word quoted mid-sentence.

Rules:
- Transliterate the SOUND. Never translate. "okay" becomes the {lang} letters for
  the sound "okay", never the {lang} word meaning "okay".
- Output exactly ONE word per input, in {lang} script, with NO spaces inside it.
- Use the everyday convention a normal transcriber uses, not a scholarly one.
- Do NOT correct spelling, grammar or word choice. The word itself must not change
  -- only the script it is written in.
- If you are NOT confident the input is a real word being rendered in the wrong
  script -- if it is garbled, an abbreviation you cannot sound out, a fragment, a
  single stray letter, or you are simply unsure -- return null for it. Returning
  null is always safe and is the correct answer when in doubt.

Return JSON only: an object mapping each input word to its {lang}-script form, or
to null when you are not confident."""

USER = """\
Transcript language: {lang}.
These Latin-script tokens appear in this transcript. For each, give the {lang}
script form of the same spoken word, or null if you are not confident.

{words}"""


def clip_script(payload: dict) -> str:
    text = " ".join((s.get("text") or "") for s in payload.get("segments", []))
    return data.dominant_script(text)[0]


def token_script(tok: str) -> str:
    """Dominant unicode script of one token, by letter count."""
    c = Counter()
    for ch in tok:
        if not ch.isalpha():
            continue
        try:
            name = unicodedata.name(ch)
        except ValueError:
            continue
        c[name.split()[0]] += 1
    return c.most_common(1)[0][0].title() if c else ""


def latin_vocab(payload: dict) -> Counter:
    """Latin tokens in THIS clip's hypothesis, as scoring will tokenise them."""
    v: Counter = Counter()
    for s in payload.get("segments", []):
        for t in normalize_text(s.get("text") or "", strip_gloss=False).split():
            if translit.is_latin(t) and not translit.has_digit(t):
                v[t] += 1
    return v


def ask(cfg: Config, words: list[str], script: str, key: str,
        use_cache: bool = True) -> tuple[dict, dict]:
    """One call for one clip. Returns (accepted_mapping, audit)."""
    lang = translit.SCRIPT_LANG[script]
    sysmsg = SYSTEM.format(lang=lang)
    user = USER.format(lang=lang, words=json.dumps(sorted(words), ensure_ascii=False))
    ck = hashlib.sha1(
        "\x00".join([MODEL, PROMPT_VERSION, sysmsg, user]).encode()).hexdigest()[:16]
    path = OUT_DIR / "cache" / ck[:2] / f"{ck}.json"

    hit = read_json(path, None) if use_cache else None
    if hit is not None:
        raw = hit["raw"]
    else:
        body = {"anthropic_version": "bedrock-2023-05-31", "max_tokens": MAX_TOKENS,
                "temperature": TEMPERATURE, "system": sysmsg,
                "messages": [{"role": "user", "content": user}]}
        resp = translit._post(body, key, MODEL)
        # A truncated reply is a HARD FAILURE, not an abstention. v1 silently
        # turned one 431-word request into 431 "abstentions" because the JSON
        # never closed. stop_reason is visible at inference time and needs no
        # reference, so this is caught before anything is scored.
        if resp.get("stop_reason") not in (None, "end_turn", "stop_sequence"):
            raise RuntimeError(
                f"truncated response: stop_reason={resp.get('stop_reason')} "
                f"on {len(words)} words -- lower BATCH")
        txt = "".join(c.get("text", "") for c in resp.get("content", []))
        a, b = txt.find("{"), txt.rfind("}")
        raw = {}
        if a >= 0 and b > a:
            try:
                raw = json.loads(txt[a:b + 1])
            except json.JSONDecodeError:
                raw = {}
        write_json_atomic(path, {"cache_key": ck, "model": MODEL, "script": script,
                                 "prompt_version": PROMPT_VERSION,
                                 "n_in": len(words), "raw": raw})

    # ---- guards. Every rejection is counted, so the abstention rate is visible.
    ok, rej = {}, Counter()
    for w in words:
        v = raw.get(w)
        if v is None or not isinstance(v, str) or not v.strip():
            rej["abstained_or_null"] += 1
            continue
        v = v.strip()
        if len(v.split()) != 1:
            rej["multi_word"] += 1
        elif translit.has_digit(v):
            rej["contains_digit"] += 1
        elif translit.is_latin(v):
            rej["still_latin"] += 1
        elif token_script(v) != script:
            rej["wrong_script"] += 1          # the cross-script corruption rail
        elif v == w:
            rej["unchanged"] += 1
        else:
            ok[w] = v
    return ok, {"n_offered": len(words), "n_accepted": len(ok),
                "rejections": dict(rej), "cache_key": ck}


def apply_map(payload: dict, m: dict) -> tuple[dict, int, list]:
    """Whole-token replacement. Token count is invariant by construction."""
    import copy
    out = copy.deepcopy(payload)
    n, log = 0, []
    for si, s in enumerate(out["segments"]):
        new = []
        for t in (s.get("text") or "").split():
            core = normalize_text(t, strip_gloss=False).strip()
            if core in m:
                new.append(m[core])
                n += 1
                log.append({"seg": si, "speaker": s.get("speaker"),
                            "old": core, "new": m[core]})
            else:
                new.append(t)
        s["text"] = " ".join(new)
    out["n_words"] = sum(len((s.get("text") or "").split()) for s in out["segments"])
    return out, n, log


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=("dev", "test"))
    ap.add_argument("--root", default="checkpoints")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--no-cache", action="store_true")
    a = ap.parse_args()

    cfg = Config.create(root=Path(a.root), work_dir=Path("/tmp/xlit_exp"))
    meta = Config.create(root=Path("local_out"), work_dir=Path("/tmp/xlit_exp"))
    clips = {c.clip_id: c for c in data.parse_ground_truth(data.load_segments_csv(meta))}
    split = json.loads(Path("results/split.json").read_text())
    ids = [c for c in sorted(clips)
           if c in set(split[a.split]) and asr.is_done(cfg, SOURCE, c)]
    if a.limit:
        ids = ids[:a.limit]

    key = translit.resolve_bedrock_key(cfg)
    print(f"split={a.split}  clips={len(ids)}  model={MODEL}  prompt={PROMPT_VERSION}  "
          f"temperature={TEMPERATURE}")

    edits, audits, totals = [], {}, Counter()
    for i, cid in enumerate(ids, 1):
        src = read_json(asr.asr_path(cfg, SOURCE, cid))
        script = clip_script(src)
        vocab = latin_vocab(src)
        if not vocab or script not in translit.SCRIPT_LANG:
            audits[cid] = {"skipped": "no latin vocabulary" if not vocab
                           else f"script {script} unsupported"}
            out, n, log = src, 0, []
        else:
            words = list(vocab)
            m, aud = {}, {"n_offered": 0, "n_accepted": 0, "rejections": {}, "cache_key": []}
            for k in range(0, len(words), BATCH):
                mk, ak = ask(cfg, words[k:k + BATCH], script, key,
                             use_cache=not a.no_cache)
                m.update(mk)
                aud["n_offered"] += ak["n_offered"]
                aud["n_accepted"] += ak["n_accepted"]
                aud["cache_key"].append(ak["cache_key"])
                for kk, vv in ak["rejections"].items():
                    aud["rejections"][kk] = aud["rejections"].get(kk, 0) + vv
            out, n, log = apply_map(src, m)
            aud.update(script=script, n_latin_types=len(vocab),
                       n_latin_tokens=sum(vocab.values()), n_applied=n)
            audits[cid] = aud
            for k, v in aud["rejections"].items():
                totals[k] += v
            totals["offered"] += aud["n_offered"]
            totals["accepted"] += aud["n_accepted"]
        totals["applied"] += n
        for e in log:
            e["clip_id"] = cid
            e["script"] = script
        edits.extend(log)
        out["system"] = f"{SOURCE}+xlitpc"
        out["experiment"] = {"prompt_version": PROMPT_VERSION, "model": MODEL,
                             "temperature": TEMPERATURE, "scope": "per-clip"}
        write_json_atomic(OUT_DIR / "asr" / f"{SOURCE}+xlitpc" / f"{cid}.json", out)
        print(f"  [{i:>3}/{len(ids)}] {cid:<26} {script:<11} "
              f"types {len(vocab):>3}  applied {n:>4}", flush=True)

    write_json_atomic(OUT_DIR / f"audit_{a.split}.json",
                      {"split": a.split, "model": MODEL, "prompt_version": PROMPT_VERSION,
                       "temperature": TEMPERATURE, "source": SOURCE,
                       "n_clips": len(ids), "totals": dict(totals), "per_clip": audits})
    write_json_atomic(OUT_DIR / f"edits_{a.split}.json", edits)
    print(f"\n  offered {totals['offered']}  accepted {totals['accepted']}  "
          f"applied {totals['applied']} tokens")
    print(f"  rejections: {dict((k, v) for k, v in totals.items() if k not in ('offered','accepted','applied'))}")
    print(f"  wrote {OUT_DIR}/asr/{SOURCE}+xlitpc/  and audit_{a.split}.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
