\
"""
Step 1: Stream and sample documents from public Hugging Face datasets.

Example:
    python src/00_stream_sample.py --only fineweb fineweb_edu --config configs/datasets.yaml --out_dir data/raw --seed 42

Outputs:
    data/raw/<corpus>.jsonl.gz
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import yaml
from datasets import load_dataset
from tqdm.auto import tqdm


def stable_id(text: str, prefix: str = "") -> str:
    digest = hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()[:16]
    return f"{prefix}_{digest}" if prefix else digest


def clean_text(text: str) -> str:
    return " ".join(text.replace("\x00", " ").split()).strip()


def stream_one(name: str, spec: Dict[str, Any], out_dir: Path, seed: int) -> int:
    dataset_id = spec["dataset_id"]
    config = spec.get("config")
    split = spec.get("split", "train")
    text_col = spec.get("text_col", "text")
    n_docs = int(spec.get("n_docs", 5000))
    min_chars = int(spec.get("min_chars", 200))

    print(f"\n[stream] {name}: {dataset_id}, config={config}, split={split}, target={n_docs}")

    if config in (None, "null", ""):
        ds = load_dataset(dataset_id, split=split, streaming=True)
    else:
        ds = load_dataset(dataset_id, config, split=split, streaming=True)

    # Approximate shuffle for streaming datasets.
    ds = ds.shuffle(seed=seed, buffer_size=min(10_000, max(1_000, n_docs)))

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{name}.jsonl.gz"

    kept = 0
    with gzip.open(out_path, "wt", encoding="utf-8") as f:
        for ex in tqdm(ds, total=n_docs, desc=f"sampling {name}"):
            text = ex.get(text_col)
            if not isinstance(text, str):
                continue

            text = clean_text(text)
            if len(text) < min_chars:
                continue

            rec = {
                "doc_id": stable_id(text, prefix=name),
                "corpus": name,
                "dataset_id": dataset_id,
                "config": config,
                "split": split,
                "text": text,
            }

            # Preserve common metadata fields when present.
            for key in [
                "url", "source", "dump", "date", "language", "lang",
                "score", "token_count", "id"
            ]:
                if key in ex:
                    try:
                        json.dumps(ex[key])
                        rec[key] = ex[key]
                    except TypeError:
                        rec[key] = str(ex[key])

            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            kept += 1
            if kept >= n_docs:
                break

    print(f"[done] wrote {kept:,} docs -> {out_path}")
    return kept


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/datasets.yaml")
    parser.add_argument("--out_dir", default="data/raw")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--only", nargs="*", default=None, help="Optional corpus names to sample.")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    out_dir = Path(args.out_dir)
    results = {}
    for name, spec in cfg["datasets"].items():
        if args.only and name not in args.only:
            continue
        try:
            results[name] = stream_one(name, spec, out_dir, args.seed)
        except Exception as e:
            print(f"[warning] failed to sample {name}: {e!r}")

    print("\nSummary:")
    for name, n in results.items():
        print(f"  {name}: {n:,} documents")


if __name__ == "__main__":
    main()
