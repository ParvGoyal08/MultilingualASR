"""Render code-switched English back into the transcript's own script.

The reference for this corpus is written with a dual-form code-switch convention:
English words spoken inside Indic speech are transcribed *phonetically in the
Indic script* (`कॉफी`, `ఓకే`, `એમએલ`). Saaras instead emits them in Latin. The
reference is 99.98% Indic script (27 Latin tokens in 123,896); the hypothesis is
3.0% Latin. Every one of those is scored as a substitution even though the word
was recognised correctly -- 3,481 of 21,357 substitutions, 2.8% of all reference
tokens.

This stage converts those tokens and nothing else. It cannot delete or invent a
word: a token either maps to its Indic spelling or is left exactly as it was, so
a bad transliteration leaves a substitution that was already a substitution.

GROUND TRUTH IS NEVER READ. The target script is derived from the hypothesis's
own text via data.dominant_script, and the spellings come from a general model's
knowledge of the language, not from reference pairs -- deriving the mapping from
the reference would be a leak and would inflate the result.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Sequence

from .config import Config
from .data import dominant_script
from .reference import normalize_text
from .utils import LOG, load_dotenv, now_utc_iso, read_json, write_json_atomic

BEDROCK_REGION = "us-east-1"
DEFAULT_MODEL = "us.anthropic.claude-sonnet-4-6"
TOKEN_KEYS = ("AWS_BEARER_TOKEN_BEDROCK",)
BATCH = 60
PROMPT_VERSION = "translit-v1"

# Script -> the language name a model will recognise. Devanagari is genuinely
# ambiguous here (Hindi and Marathi both), and for transliteration of English
# loanwords the two conventions coincide, so naming both is honest and harmless.
SCRIPT_LANG = {
    "Devanagari": "Hindi/Marathi (Devanagari)", "Bengali": "Bengali",
    "Gujarati": "Gujarati", "Gurmukhi": "Punjabi (Gurmukhi)",
    "Kannada": "Kannada", "Malayalam": "Malayalam", "Oriya": "Odia",
    "Tamil": "Tamil", "Telugu": "Telugu",
}


class _Fatal(RuntimeError):
    """A response the server will refuse identically no matter how often asked."""


def resolve_bedrock_key(cfg: Config | None = None) -> str:
    extra = [cfg.dotenv_path, cfg.root / ".env"] if cfg is not None else []
    load_dotenv(extra=[*extra, Path(".env")], export=True)
    for k in TOKEN_KEYS:
        if os.environ.get(k):
            return os.environ[k]
    raise RuntimeError("no Bedrock key: set AWS_BEARER_TOKEN_BEDROCK in .env")


def is_latin(tok: str) -> bool:
    a = [c for c in tok if c.isalpha()]
    return bool(a) and all(ord(c) < 128 for c in a)


def clip_script(payload: dict) -> str:
    """Target script, from the HYPOTHESIS text only."""
    text = " ".join((s.get("text") or "") for s in payload.get("segments", []))
    return dominant_script(text)[0]


def vocabulary(payloads: Iterable[dict]) -> dict[str, Counter]:
    """{script: Counter(latin token)} over the hypotheses."""
    out: dict[str, Counter] = defaultdict(Counter)
    for p in payloads:
        sc = clip_script(p)
        if sc not in SCRIPT_LANG:
            continue
        for s in p.get("segments", []):
            for t in normalize_text(s.get("text") or "", strip_gloss=False).split():
                if is_latin(t):
                    out[sc][t] += 1
    return out


def _post(body: dict, key: str, model: str, retries: int = 6) -> dict:
    """Same retry discipline as the Sarvam client: Retry-After wins, 4xx is fatal."""
    import random

    import requests

    url = (f"https://bedrock-runtime.{BEDROCK_REGION}.amazonaws.com"
           f"/model/{model}/invoke")
    h = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    last = "no attempt"
    for attempt in range(retries):
        wait = None
        try:
            r = requests.post(url, headers=h, json=body, timeout=180)
            if r.status_code == 200:
                return r.json()
            if r.status_code == 429 or r.status_code >= 500:
                last = f"{r.status_code}: {r.text[:160]}"
                wait = float(r.headers.get("Retry-After") or 0) or None
            else:
                raise _Fatal(f"bedrock {r.status_code}: {r.text[:200]}")
        except _Fatal:
            raise
        except Exception as exc:  # noqa: BLE001
            last = f"{type(exc).__name__}: {exc}"
        time.sleep(wait if wait else min(45.0, 1.5 * 2 ** attempt) * (0.5 + random.random()))
    raise RuntimeError(f"bedrock failed after {retries} attempts -- {last}")


SYSTEM = """\
You transliterate English words into Indic scripts.

You are given English words that were spoken inside {lang} conversation and
transcribed in Latin script. Rewrite each word the way an ordinary {lang}
transcriber would write it PHONETICALLY in {lang} script -- the spelling a native
writer uses when quoting an English word mid-sentence.

Rules:
- Transliterate the SOUND. Do NOT translate. "okay" becomes the {lang} spelling
  of the sound "okay", never the {lang} word meaning "okay".
- Use the everyday convention, not a scholarly one. No diacritics that a normal
  transcriber would omit.
- Output every input word exactly once, as a key.
- If a word is a proper noun, transliterate its sound the same way.
- Return JSON only: an object mapping each input word to its transliteration."""


def translate_batch(words: Sequence[str], script: str, key: str,
                    model: str, cfg: Config, use_cache: bool = True) -> dict[str, str]:
    lang = SCRIPT_LANG[script]
    sysmsg = SYSTEM.format(lang=lang)
    user = ("Transliterate these into " + lang + " script:\n"
            + json.dumps(sorted(words), ensure_ascii=False))
    ck = hashlib.sha1(
        "\x00".join([model, PROMPT_VERSION, sysmsg, user]).encode()).hexdigest()[:16]
    path = cfg.root / "cache" / "bedrock" / model.replace("/", "_") / ck[:2] / f"{ck}.json"
    if use_cache:
        hit = read_json(path, None)
        if hit and hit.get("mapping"):
            return hit["mapping"]
    body = {"anthropic_version": "bedrock-2023-05-31", "max_tokens": 8000,
            "temperature": 0, "system": sysmsg,
            "messages": [{"role": "user", "content": user}]}
    resp = _post(body, key, model)
    txt = "".join(c.get("text", "") for c in resp.get("content", []))
    a, b = txt.find("{"), txt.rfind("}")
    mapping: dict[str, str] = {}
    if a >= 0 and b > a:
        try:
            raw = json.loads(txt[a:b + 1])
            for k, v in raw.items():
                if isinstance(v, str) and v.strip() and not is_latin(v.strip()):
                    mapping[k] = v.strip()
        except json.JSONDecodeError:
            LOG.warning("translit: unparseable response for %s batch", script)
    write_json_atomic(path, {"script": script, "model": model, "mapping": mapping,
                             "n_in": len(words), "n_out": len(mapping),
                             "at_utc": now_utc_iso()})
    return mapping


def build_table(cfg: Config, payloads: Sequence[dict], model: str = DEFAULT_MODEL,
                min_count: int = 1, use_cache: bool = True) -> dict[str, dict[str, str]]:
    """{script: {latin: indic}} -- the frozen lookup this stage applies."""
    key = resolve_bedrock_key(cfg)
    vocab = vocabulary(payloads)
    table: dict[str, dict[str, str]] = {}
    for sc, cnt in sorted(vocab.items(), key=lambda x: -sum(x[1].values())):
        words = [w for w, n in cnt.most_common() if n >= min_count]
        got: dict[str, str] = {}
        for i in range(0, len(words), BATCH):
            got.update(translate_batch(words[i:i + BATCH], sc, key, model, cfg, use_cache))
        table[sc] = got
        LOG.info("translit %s: %d types -> %d mapped (%d tokens)",
                 sc, len(words), len(got), sum(cnt.values()))
    return table


def apply_to_payload(payload: dict, table: dict[str, dict[str, str]]) -> tuple[dict, int]:
    """Replace Latin tokens with their Indic spelling. Nothing else changes."""
    import copy

    sc = clip_script(payload)
    m = table.get(sc, {})
    out = copy.deepcopy(payload)
    n = 0
    for s in out["segments"]:
        toks = (s.get("text") or "").split()
        new = []
        for t in toks:
            core = normalize_text(t, strip_gloss=False).strip()
            if core and is_latin(core) and core in m:
                new.append(m[core]); n += 1
            else:
                new.append(t)
        s["text"] = " ".join(new)
    out["n_words"] = sum(len((s.get("text") or "").split()) for s in out["segments"])
    out["translit_model"] = table.get("__model__", "")
    out["translit_tokens_changed"] = n
    out["transcribed_at_utc"] = now_utc_iso()
    return out, n
