#!/usr/bin/env python3
"""Step 4b driver: script/numeral normalisation of ASR text.

Builds two frozen lookup tables from the HYPOTHESIS vocabulary and applies them,
producing `<source>+xlit` and `<source>+xlit+num`. Ground truth is never read.

The tables are the expensive part and are committed under `checkpoints/`, so
re-running this needs no API key: pass --tables-only to apply what is already on
disk. That is the default when both tables exist.

    python3 tools/run_translit.py --root local_out/step4_input
    python3 tools/run_translit.py --root local_out/step4_input --rebuild-tables
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sarvam_diar import asr, data, translit  # noqa: E402
from sarvam_diar.config import Config  # noqa: E402
from sarvam_diar.utils import LOG, read_json, write_json_atomic  # noqa: E402

REPO = Path(__file__).resolve().parents[1]


def table_path(root: Path, name: str) -> Path:
    """Prefer the committed checkpoint, fall back to the run root."""
    ck = REPO / "checkpoints" / "results" / name
    return ck if ck.exists() else root / "results" / name


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="local_out/step4_input")
    ap.add_argument("--source", default="sarvam-saaras-v3@fusion")
    ap.add_argument("--model", default=translit.DEFAULT_MODEL)
    ap.add_argument("--rebuild-tables", action="store_true",
                    help="call the LLM to rebuild the vocabulary tables (needs a key)")
    ap.add_argument("--clips", type=int, default=0, help="limit, 0 = all")
    a = ap.parse_args()

    root = Path(a.root)
    cfg = Config.create(root=root, work_dir=root / "tmp")
    meta = Config.create(root=Path("local_out"), work_dir=Path("./.work"))
    ids = [c.clip_id for c in data.parse_ground_truth(data.load_segments_csv(meta))
           if asr.is_done(cfg, a.source, c.clip_id)]
    if a.clips:
        ids = ids[:a.clips]
    if not ids:
        raise SystemExit(f"no {a.source} payloads under {root}")
    payloads = [read_json(asr.asr_path(cfg, a.source, c)) for c in ids]
    print(f"source {a.source}: {len(payloads)} clips")

    lat_p = table_path(root, "translit_table.json")
    num_p = table_path(root, "numeral_table.json")
    have = lat_p.exists() and num_p.exists()
    if a.rebuild_tables or not have:
        print(f"building tables with {a.model} (LLM calls; cached by prompt hash)")
        lat = translit.build_table(cfg, payloads, model=a.model, kind="latin")
        write_json_atomic(root / "results" / "translit_table.json", lat)
        stage1 = [translit.apply_to_payload(p, lat)[0] for p in payloads]
        num = translit.build_table(cfg, stage1, model=a.model, kind="digit")
        write_json_atomic(root / "results" / "numeral_table.json", num)
    else:
        print(f"using committed tables: {lat_p}, {num_p}")
        lat, num = json.loads(lat_p.read_text()), json.loads(num_p.read_text())

    n1 = n2 = 0
    for p in payloads:
        p1, k1 = translit.apply_to_payload(p, lat)
        p1["system"] = f"{a.source}+xlit"
        p1["translit_model"] = a.model
        p1["translit_stage"] = "latin"
        write_json_atomic(asr.asr_path(cfg, p1["system"], p["clip_id"]), p1)
        p2, k2 = translit.apply_to_payload(p1, num)
        p2["system"] = f"{a.source}+xlit+num"
        p2["translit_model"] = a.model
        p2["translit_stage"] = "latin+numeral"
        write_json_atomic(asr.asr_path(cfg, p2["system"], p["clip_id"]), p2)
        n1 += k1
        n2 += k2
    print(f"  {n1:,} Latin tokens -> native script")
    print(f"  {n2:,} numerals spelled out")
    print(f"  wrote asr/{a.source}+xlit/ and asr/{a.source}+xlit+num/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
