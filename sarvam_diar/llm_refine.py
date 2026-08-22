"""Step 5 -- LLM refinement of ASR text. Diarization is immutable.

Gemini is shown a speaker-attributed window of the Saaras transcript and returns
the same segment ids with (possibly) cleaned `text`. It never sees, and never
returns, a speaker label or a timestamp: those are copied from the source
payload, so changing them is structurally impossible rather than merely
forbidden.

The name is `llm_refine` and not `refinement` because `refinement.py` is already
the DOVER-Lap diarization fusion (Step 4).

Ground truth never enters this module. The only `reference` functions imported
are `normalize_text` and `tokenize`, which are pure string functions applied
identically to hypothesis and reference and fixed long before this existed.

Two arms, differing only in the instruction block:
  A  correct obvious ASR word errors, clean repetitions/stutters/hallucinations
  B  arm A plus removal of duplicated content across overlapping segments

Arm B is expected to HARM on the current baseline. Once the fusion->ASR
provenance bug was fixed, duplicated tokens in overlapping pairs fell 66% and
the free deterministic dedup rule reversed sign (-0.0064 -> +0.0027 cpWER). What
remains in overlapping regions is mostly real simultaneous speech, so deleting
it destroys correct words. B is run to measure that, not to ship it.
"""
from __future__ import annotations

import copy
import difflib
import hashlib
import json
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Iterable, Sequence

from .config import Config
from .data import dominant_script
from .reference import normalize_text
from .text_metrics import align
from .utils import LOG, append_jsonl, load_dotenv, now_utc_iso, read_json, write_json_atomic

API_ROOT = "https://generativelanguage.googleapis.com/v1beta"
# gemini-3.7-flash is capped at 2 requests/day/model on this key's free
# tier. 3.5-flash is the strongest model with usable quota.
DEFAULT_MODEL = "gemini-3.5-flash"
BEDROCK_DEFAULT = "us.anthropic.claude-sonnet-4-6"
GEMINI_KEYS = ("GEMINI_TOKEN", "GEMINI_API_KEY", "GOOGLE_API_KEY")
SEED = 20260822

# Windowing. K/K_MAX/CONTEXT are frozen before the pilot runs; see the plan for
# why (2.5% of overlapping pairs split across cores vs 14.5% for naive blocks).
K_CORE, K_MAX, CONTEXT = 12, 24, 8
MIN_RUN = 3            # shortest verbatim run counted as duplication evidence
UNLICENSED_FRAC = 0.25
GROWTH_FRAC = 1.05
LATIN_JUMP = 0.10
WINDOW_REVERT_FRAC = 0.25

PROMPT_VERSION = "v1"


class _Fatal(RuntimeError):
    """A response the server will refuse identically no matter how often asked."""


# ------------------------------------------------------------------ auth


def resolve_gemini_key(cfg: Config | None = None) -> str:
    """Same ladder as asr.resolve_sarvam_key. Never logs the value."""
    extra = [cfg.dotenv_path, cfg.root / ".env"] if cfg is not None else []
    load_dotenv(extra=[*extra, Path(".env")], export=True)
    for name in GEMINI_KEYS:
        if os.environ.get(name):
            return os.environ[name]
    for loader in (_kaggle_secret, _colab_secret):
        for name in GEMINI_KEYS:
            got = loader(name)
            if got:
                return got
    raise RuntimeError(
        "no Gemini key. Set GEMINI_TOKEN in .env at the repo root, or in the "
        "process environment, or as a Kaggle/Colab secret.")


def _kaggle_secret(name: str) -> str | None:
    try:
        from kaggle_secrets import UserSecretsClient  # type: ignore

        return UserSecretsClient().get_secret(name)
    except Exception:  # noqa: BLE001
        return None


def _colab_secret(name: str) -> str | None:
    try:
        from google.colab import userdata  # type: ignore

        return userdata.get(name)
    except Exception:  # noqa: BLE001
        return None


def preflight(cfg: Config | None = None, model: str = DEFAULT_MODEL) -> dict:
    """What this key can actually reach. Model ids turn over; do not hardcode."""
    import requests

    try:
        key = resolve_gemini_key(cfg)
    except RuntimeError as exc:
        return {"token present": False, "error": str(exc)}
    r = requests.get(f"{API_ROOT}/models", headers={"x-goog-api-key": key}, timeout=60)
    if r.status_code != 200:
        return {"token present": True, "status": r.status_code, "body": r.text[:200]}
    names = {m["name"].split("/")[-1]: m for m in r.json().get("models", [])
             if "generateContent" in m.get("supportedGenerationMethods", [])}
    return {"token present": True, "n_models": len(names),
            "requested": model, "available": model in names,
            "output_limit": names.get(model, {}).get("outputTokenLimit")}


# ------------------------------------------------------------------ http


def _gemini_post(body: dict, key: str, model: str, retries: int = 7) -> dict:
    """Mirrors asr._sarvam_post: Retry-After wins, else jittered exponential."""
    import random

    import requests

    url = f"{API_ROOT}/models/{model}:generateContent"
    headers = {"x-goog-api-key": key, "Content-Type": "application/json"}
    last = "no attempt"
    for attempt in range(retries):
        wait = None
        try:
            r = requests.post(url, headers=headers, json=body, timeout=300)
            if r.status_code == 200:
                return r.json()
            if r.status_code == 429 or r.status_code >= 500:
                last = f"{r.status_code}: {r.text[:200]}"
                wait = float(r.headers.get("Retry-After") or 0) or None
            else:
                raise _Fatal(f"gemini {r.status_code}: {r.text[:300]}")
        except _Fatal:
            raise
        except Exception as exc:  # noqa: BLE001
            last = f"{type(exc).__name__}: {exc}"
        time.sleep(wait if wait else
                   min(60.0, 1.5 * (2 ** attempt)) * (0.5 + random.random()))
    raise RuntimeError(f"gemini failed after {retries} attempts -- {last}")


# ------------------------------------------------------- overlap + duplication


def _toks(text: str) -> list[str]:
    """The exact tokens scoring will see."""
    return normalize_text(text or "", strip_gloss=False).split()


def overlap_pairs(segments: Sequence[dict]) -> list[tuple[int, int, float]]:
    """(a, b, shared_seconds) for every time-overlapping pair, a < b by start."""
    S = sorted(range(len(segments)), key=lambda i: (segments[i]["start"], segments[i]["end"]))
    out = []
    for x in range(len(S)):
        a = S[x]
        for y in range(x + 1, len(S)):
            b = S[y]
            if segments[b]["start"] >= segments[a]["end"]:
                break
            ov = min(segments[a]["end"], segments[b]["end"]) - segments[b]["start"]
            if ov > 0:
                out.append((a, b, ov))
    return out


def shared_runs(x: Sequence[str], y: Sequence[str], min_run: int = MIN_RUN):
    """Matching blocks of >= min_run tokens, as (i_in_x, j_in_y, size)."""
    if not x or not y:
        return []
    m = difflib.SequenceMatcher(a=list(x), b=list(y), autojunk=False)
    return [(bl.a, bl.b, bl.size) for bl in m.get_matching_blocks() if bl.size >= min_run]


def dup_positions(segments: Sequence[dict]) -> dict[int, set[int]]:
    """Token indices in each segment covered by a run shared with an overlapper.

    This is the *evidence* the drift guard conditions on: a deletion inside these
    positions is licensed by something observable, a deletion outside them is not.
    """
    toks = [_toks(s.get("text", "")) for s in segments]
    out: dict[int, set[int]] = {}
    for a, b, _ in overlap_pairs(segments):
        for ia, ib, size in shared_runs(toks[a], toks[b]):
            out.setdefault(a, set()).update(range(ia, ia + size))
            out.setdefault(b, set()).update(range(ib, ib + size))
    return out


def editable(seg: dict) -> bool:
    """Empty or skipped segments are copied through, never sent for editing."""
    return bool((seg.get("text") or "").strip()) and not seg.get("skipped")


# ------------------------------------------------------------------ windows


def build_windows(segments: Sequence[dict], k: int = K_CORE, k_max: int = K_MAX,
                  context: int = CONTEXT) -> list[dict]:
    """Pair-atomic blocks: every index is in exactly one core, so no segment can
    be edited twice. Cores grow to keep overlapping pairs together."""
    order = sorted(range(len(segments)), key=lambda i: (segments[i]["start"], segments[i]["end"]))
    pos = {idx: p for p, idx in enumerate(order)}
    pairs = overlap_pairs(segments)
    partners: dict[int, set[int]] = {}
    for a, b, _ in pairs:
        partners.setdefault(a, set()).add(b)
        partners.setdefault(b, set()).add(a)

    windows, start = [], 0
    while start < len(order):
        end = min(start + k, len(order))
        while end < len(order) and (end - start) < k_max:
            nxt = order[end]
            if any(pos.get(p, -1) in range(start, end) for p in partners.get(nxt, ())):
                end += 1
            else:
                break
        core = [order[p] for p in range(start, end)]
        vis = [order[p] for p in range(max(0, start - context),
                                       min(len(order), end + context))]
        extra = {p for c in core for p in partners.get(c, ()) if p not in vis}
        visible = sorted(set(vis) | extra, key=lambda i: pos[i])
        split = [(a, b) for a, b, _ in pairs
                 if (a in core) != (b in core) and not ({a, b} <= set(visible))]
        windows.append({"core": core, "visible": visible, "split_pairs": split})
        start = end
    return windows


# ------------------------------------------------------------------ prompt

_COMMON_RULES = """\
You clean up the text of an automatic speech recognition (ASR) transcript.

You are given numbered segments of a conversation. You return the SAME segment
numbers with only the `text` possibly changed.

You must never:
- change who is speaking, or which segment a word belongs to
- move text from one segment to another
- merge or split segments, or change their order
- add a segment, drop a segment, or invent words that were not recognised
- output a timestamp

These transcripts are code-switched across Indian languages. English words
written in an Indic script are CORRECT AS WRITTEN and must stay in that script.
Do not transliterate, do not romanise, do not translate, and do not "correct" a
spelling to a more standard form. Your output is compared character by character
against a human transcript that uses the same code-switched spellings you were
given, so a spelling you consider more standard scores as an error. Change a
word only when the original is a mis-recognition of a DIFFERENT word, never when
it is a different spelling of the same word.

Short acknowledgements are real speech and are kept: hmm, हं, ಹ್ಞೂ, yes, হুম,
আচ্ছা, ಹಾಂ, ok. Two speakers greeting each other at the same time, or one
echoing the other's confirmation, is also real speech and is kept.

If you are not sure, return the text exactly as it was given. Returning the
input unchanged is always an acceptable answer.
"""

_ARM_A = """\
Your job, and nothing else:
1. Correct clear mis-recognitions, where the transcript shows a word that is not
   what the conversation plainly requires.
2. Remove obvious stutters and repeated fragments inside a single segment, where
   the same short phrase repeats with no conversational reason.
3. Remove obvious ASR hallucination -- a phrase repeated many times over, or
   filler that plainly is not speech.
Leave everything else exactly as it is.
"""

_ARM_B = """\
Your job, and nothing else:
1. Correct clear mis-recognitions, where the transcript shows a word that is not
   what the conversation plainly requires.
2. Remove obvious stutters and repeated fragments inside a single segment, where
   the same short phrase repeats with no conversational reason.
3. Remove obvious ASR hallucination -- a phrase repeated many times over, or
   filler that plainly is not speech.
4. Where two segments share audio (they are marked), the recogniser sometimes
   transcribed one speaker's words into BOTH segments. When one segment clearly
   repeats wording that belongs to the other speaker, remove the repeated
   wording from the segment it does not belong to. Be conservative: if both
   speakers could genuinely have said it at the same time, keep both.
Leave everything else exactly as it is.
"""

RESPONSE_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "segments": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {"id": {"type": "INTEGER"}, "text": {"type": "STRING"}},
                "required": ["id", "text"],
                "propertyOrdering": ["id", "text"],
            },
        }
    },
    "required": ["segments"],
}


def system_instruction(arm: str) -> str:
    return f"{_COMMON_RULES}\n{_ARM_A if arm.upper() == 'A' else _ARM_B}\n" \
           f"(prompt {PROMPT_VERSION}, arm {arm.upper()})"


def render_window(payload: dict, window: dict, arm: str) -> str:
    """The user-visible text. Overlap is computed here, never inferred by the model."""
    segs = payload["segments"]
    core, visible = set(window["core"]), window["visible"]
    toks = [_toks(s.get("text", "")) for s in segs]

    def line(i: int) -> str:
        s = segs[i]
        body = (s.get("text") or "").strip() or "(no text)"
        return f"[{i}] {s['speaker']} {s['start']:.2f}-{s['end']:.2f} : {body}"

    before = [line(i) for i in visible if i not in core and i < min(core)]
    after = [line(i) for i in visible if i not in core and i > max(core)]
    body = [line(i) for i in sorted(core)]

    shares = []
    for a, b, ov in overlap_pairs(segs):
        if a not in core and b not in core:
            continue
        runs = shared_runs(toks[a], toks[b])
        if runs:
            best = max(runs, key=lambda r: r[2])
            quoted = " ".join(toks[a][best[0]:best[0] + best[2]])
            shares.append(f"[{a}] and [{b}] share {ov:.2f}s; both contain, word for "
                          f"word: \"{quoted}\" ({best[2]} words)")
        else:
            shares.append(f"[{a}] and [{b}] share {ov:.2f}s; no shared wording detected")

    editable_ids = [i for i in sorted(core) if editable(segs[i])]
    out = [f"CLIP {payload['clip_id']}   language {payload.get('detected_language') or '?'}",
           f"Return exactly these {len(editable_ids)} ids: {editable_ids}", ""]
    if before:
        out += ["--- CONTEXT BEFORE (read only, do NOT return these) ---", *before, ""]
    out += ["--- RETURN THESE, SAME IDS, SAME ORDER ---", *body, ""]
    if after:
        out += ["--- CONTEXT AFTER (read only, do NOT return these) ---", *after, ""]
    if shares and arm.upper() == "B":
        out += ["--- SEGMENTS THAT SHARE AUDIO (the microphone heard both; one may",
                "    repeat the other, or they may genuinely have spoken together) ---",
                *shares, ""]
    return "\n".join(out)


# ------------------------------------------------------------------ cache


def cache_dir(cfg: Config, model: str) -> Path:
    return cfg.root / "cache" / "gemini" / model


def cache_key(model: str, system: str, user: str, gen: dict) -> str:
    blob = "\x00".join([model, json.dumps(gen, sort_keys=True, ensure_ascii=False),
                        system, user])
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:16]


def _is_bedrock(model: str) -> bool:
    return "anthropic" in model


def call_bedrock(cfg: Config, model: str, system: str, user: str,
                 max_out: int, use_cache: bool = True) -> dict:
    """Same contract as call_gemini: returns {"response": <raw>, "key": ...}."""
    from . import translit

    gen = {"temperature": 0, "max_tokens": min(max_out, 16000)}
    ck = cache_key(model, system, user, gen)
    path = cache_dir(cfg, model.replace("/", "_")) / ck[:2] / f"{ck}.json"
    if use_cache:
        hit = read_json(path, None)
        if hit and hit.get("response"):
            return {"cached": True, "cache_key": ck, **hit}
    # Gemini gets its JSON from responseSchema; Bedrock has no equivalent, so
    # the format has to be demanded in-band AND pinned with an assistant
    # prefill. Without the prefill Sonnet returns the rendered transcript back
    # in the same line format it was shown, which parses as nothing.
    sysmsg = system + (
        "\n\nOUTPUT FORMAT -- this overrides any other formatting instinct.\n"
        "Return ONE JSON object and nothing else. No prose, no code fence, no\n"
        "restatement of the input format. The shape is exactly:\n"
        '{"segments":[{"id":<integer>,"text":"<string>"}, ...]}\n'
        "Include every id you were asked to return, once each.")
    # No assistant prefill: this model rejects it ("conversation must end with a
    # user message"), so the format is carried entirely by the directive above.
    body = {"anthropic_version": "bedrock-2023-05-31", "system": sysmsg,
            "messages": [{"role": "user", "content": user}], **gen}
    t0 = time.perf_counter()
    resp = translit._post(body, translit.resolve_bedrock_key(cfg), model)
    rec = {"cached": False, "cache_key": ck, "model": model, "response": resp,
           "prompt_version": PROMPT_VERSION,
           "elapsed_sec": round(time.perf_counter() - t0, 2), "at_utc": now_utc_iso()}
    write_json_atomic(path, {k: v for k, v in rec.items() if k != "cached"})
    return rec


def call_gemini(cfg: Config, key: str, model: str, system: str, user: str,
                max_out: int, use_cache: bool = True) -> dict:
    gen = {"temperature": 0, "topP": 1.0, "candidateCount": 1, "seed": SEED,
           "maxOutputTokens": max_out, "responseMimeType": "application/json",
           "responseSchema": RESPONSE_SCHEMA, "thinkingConfig": {"thinkingBudget": 0}}
    ck = cache_key(model, system, user, gen)
    path = cache_dir(cfg, model) / ck[:2] / f"{ck}.json"
    if use_cache:
        hit = read_json(path, None)
        if hit and hit.get("response"):
            return {"cached": True, "cache_key": ck, **hit}

    body = {"system_instruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": user}]}],
            "generationConfig": gen}
    t0 = time.perf_counter()
    try:
        resp = _gemini_post(body, key, model)
    except _Fatal as exc:
        if "thinking" in str(exc).lower():
            gen.pop("thinkingConfig", None)
            body["generationConfig"] = gen
            resp = _gemini_post(body, key, model)
        else:
            raise
    rec = {"cached": False, "cache_key": ck, "model": model, "prompt_version": PROMPT_VERSION,
           "response": resp, "elapsed_sec": round(time.perf_counter() - t0, 2),
           "at_utc": now_utc_iso()}
    write_json_atomic(path, {k: v for k, v in rec.items() if k != "cached"})
    return rec


def _extract(resp: dict) -> tuple[list[dict] | None, str]:
    """(segments, finish_reason). Fails loudly rather than returning nothing."""
    if "content" in resp and isinstance(resp.get("content"), list):   # Bedrock
        txt = "".join(c.get("text", "") for c in resp["content"])
        fin = resp.get("stop_reason", "?")
        if "```" in txt:                       # tolerate a fenced block
            seg = txt.split("```")
            for k in range(1, len(seg), 2):
                body_ = seg[k]
                if body_.lstrip().startswith("json"):
                    body_ = body_.lstrip()[4:]
                if "{" in body_ or "[" in body_:
                    txt = body_; break
        a, b = txt.find("{"), txt.rfind("}")
        if a < 0 or b <= a:
            a, b = txt.find("["), txt.rfind("]")
        if a < 0 or b <= a:
            return None, f"NO_JSON:{fin}"
        try:
            obj = json.loads(txt[a:b + 1])
        except json.JSONDecodeError:
            return None, f"BAD_JSON:{fin}"
        segs = obj.get("segments") if isinstance(obj, dict) else obj
        return (segs if isinstance(segs, list) else None), fin
    if resp.get("promptFeedback", {}).get("blockReason"):
        return None, "BLOCKED:" + resp["promptFeedback"]["blockReason"]
    cands = resp.get("candidates") or []
    if not cands:
        return None, "NO_CANDIDATES"
    fin = cands[0].get("finishReason", "?")
    parts = cands[0].get("content", {}).get("parts") or []
    text = "".join(p.get("text", "") for p in parts)
    if not text.strip():
        return None, fin
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        return None, f"BAD_JSON:{fin}"
    segs = obj.get("segments") if isinstance(obj, dict) else obj
    return (segs if isinstance(segs, list) else None), fin


# ------------------------------------------------------------------ guards


def guard_segment(old_text: str, new_text: str, dup: set[int]) -> tuple[str, str | None, dict]:
    """Structural reverts only. Returns (text_to_use, revert_reason, stats)."""
    old, new = _toks(old_text), _toks(new_text)
    s_old, _, lat_old = dominant_script(old_text or "")
    s_new, _, lat_new = dominant_script(new_text or "")
    stats = {"n_old": len(old), "n_new": len(new),
             "script_old": s_old, "script_new": s_new}
    if not old:
        return old_text, None, stats
    if new and s_new != s_old:
        return old_text, "script_change", stats
    if new and lat_new > lat_old + LATIN_JUMP:
        return old_text, "romanisation", stats
    if len(new) > math.ceil(GROWTH_FRAC * len(old)) + 1:
        return old_text, "growth", stats
    # R6 per-token script rail. obs [51] measured 7 of 24 applied edits
    # introducing a script absent from the input -- Telugu "party" becoming
    # Japanese katakana, Bengali "etao" becoming Korean hangul -- and every one
    # passed the DOMINANT-script test, because one corrupted token in forty
    # cannot move a majority. Reject any output token whose script is new.
    def _scripts(toks):
        out = set()
        for t in toks:
            sc, _, _ = dominant_script(t)
            if sc and sc != "-":
                out.add(sc)
        return out
    if _scripts(new) - _scripts(old):
        return old_text, "new_script_token", stats

    ops = align(old, new)
    unlicensed = sum(1 for tag, i, _ in ops
                     if (tag in ("replace", "delete") and i not in dup) or tag == "insert")
    stats["unlicensed"] = unlicensed
    stats["budget"] = max(1, math.ceil(UNLICENSED_FRAC * len(old)))
    if unlicensed > stats["budget"]:
        return old_text, "unlicensed_budget", stats
    return new_text, None, stats


# ------------------------------------------------------------------ driver


def refine_clip(cfg: Config, source_system: str, clip_id: str, arm: str,
                model: str = DEFAULT_MODEL, key: str | None = None,
                use_cache: bool = True, workers: int = 8,
                k: int = K_CORE, k_max: int = K_MAX,
                context: int = CONTEXT) -> tuple[dict, list[dict]]:
    """Refined payload + edit records. The payload is a deep copy with only
    segments[i]['text'] touched."""
    from . import asr

    key = key or (None if _is_bedrock(model) else resolve_gemini_key(cfg))
    src = read_json(asr.asr_path(cfg, source_system, clip_id))
    segs = src["segments"]
    dup = dup_positions(segs)
    windows = build_windows(segs, k=k, k_max=k_max, context=context)
    new_text = {i: (s.get("text") or "") for i, s in enumerate(segs)}
    edits: list[dict] = []
    n_fallback = 0

    # Windows are cut from the frozen source, so they are independent and can be
    # fetched concurrently. Measured: a single small call takes 130-380s of
    # mostly queueing on this tier, so latency is the whole cost and concurrency
    # is close to a linear win. Application stays sequential and ordered.
    def fetch(item):
        w_idx, w = item
        ids = [i for i in sorted(w["core"]) if editable(segs[i])]
        if not ids:
            return w_idx, w, ids, None
        user = render_window(src, w, arm)
        core_chars = sum(len(segs[i].get("text") or "") for i in ids)
        try:
            rec = (call_bedrock(cfg, model, system_instruction(arm), user,
                                min(65536, 512 + 4 * core_chars), use_cache)
                   if _is_bedrock(model) else
                   call_gemini(cfg, key, model, system_instruction(arm), user,
                               min(65536, 512 + 4 * core_chars), use_cache))
        except Exception as exc:  # noqa: BLE001
            LOG.warning("%s w%d: call failed %s: %s", clip_id, w_idx,
                        type(exc).__name__, str(exc)[:140])
            return w_idx, w, ids, None
        return w_idx, w, ids, rec

    with ThreadPoolExecutor(max_workers=workers) as pool:
        fetched = sorted(pool.map(fetch, list(enumerate(windows))), key=lambda t: t[0])

    for w_idx, w, ids, rec in fetched:
        if not ids:
            continue
        if rec is None:
            n_fallback += 1
            edits.append({"clip_id": clip_id, "arm": arm, "window": w_idx,
                          "seg_i": None, "action": "window_fallback", "reason": "call_failed"})
            continue
        got, fin = _extract(rec["response"])
        ok = got is not None and {int(g["id"]) for g in got} == set(ids)
        if not ok:
            n_fallback += 1
            LOG.warning("%s w%d: window fallback (%s)", clip_id, w_idx, fin)
            edits.append({"clip_id": clip_id, "arm": arm, "window": w_idx,
                          "seg_i": None, "action": "window_fallback", "reason": fin})
            continue
        proposed = {int(g["id"]): g.get("text", "") for g in got}
        staged, reverts = {}, 0
        for i in ids:
            kept, reason, st = guard_segment(segs[i].get("text") or "", proposed[i],
                                             dup.get(i, set()))
            staged[i] = kept
            changed = _toks(kept) != _toks(segs[i].get("text") or "")
            if reason:
                reverts += 1
            if reason or changed:
                edits.append({"clip_id": clip_id, "arm": arm, "window": w_idx, "seg_i": i,
                              "speaker": segs[i]["speaker"], "start": segs[i]["start"],
                              "end": segs[i]["end"], "old_text": segs[i].get("text"),
                              "new_text": proposed[i], "applied_text": kept,
                              "action": "reverted" if reason else "changed",
                              # cache files written before the rename carry "key"
                              "reason": reason,
                              "cache_key": rec.get("cache_key") or rec.get("key"), **st})
        if reverts > max(1, math.ceil(WINDOW_REVERT_FRAC * len(ids))):
            n_fallback += 1
            LOG.warning("%s w%d: %d/%d reverted -- dropping window", clip_id, w_idx,
                        reverts, len(ids))
            edits.append({"clip_id": clip_id, "arm": arm, "window": w_idx,
                          "seg_i": None, "action": "window_untrusted"})
            continue
        new_text.update(staged)

    # R5: if both members of an overlapping pair deleted from the same shared
    # run, revert both -- 47% of tokens in duplicate blocks are correct words.
    for a, b, _ in overlap_pairs(segs):
        oa, ob = _toks(segs[a].get("text") or ""), _toks(segs[b].get("text") or "")
        if not shared_runs(oa, ob):
            continue
        if len(_toks(new_text[a])) < len(oa) and len(_toks(new_text[b])) < len(ob):
            for i in (a, b):
                if new_text[i] != (segs[i].get("text") or ""):
                    new_text[i] = segs[i].get("text") or ""
                    edits.append({"clip_id": clip_id, "arm": arm, "seg_i": i,
                                  "action": "reverted", "reason": "double_deletion"})

    out = copy.deepcopy(src)
    for i, s in enumerate(out["segments"]):
        s["text"] = new_text[i]
    out["n_words"] = sum(len((s.get("text") or "").split()) for s in out["segments"])
    out["system"] = f"{source_system}+llm{arm.lower()}"
    out["transcribed_at_utc"] = now_utc_iso()
    out.update({"refined_from": source_system, "llm_model": model, "arm": arm.upper(),
                "prompt_version": PROMPT_VERSION, "n_windows": len(windows),
                "n_window_fallbacks": n_fallback,
                "n_segments_changed": sum(1 for e in edits if e.get("action") == "changed"),
                "n_segments_reverted": sum(1 for e in edits if e.get("action") == "reverted")})

    for i, s in enumerate(out["segments"]):
        assert s["i"] == src["segments"][i]["i"], "segment order changed"
        for f in ("start", "end", "speaker", "skipped"):
            assert s[f] == src["segments"][i][f], f"{f} changed on segment {i}"
    assert len(out["segments"]) == len(segs), "segment count changed"
    return out, edits
