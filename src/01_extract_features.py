\
"""
Step 2: Extract transparent linguistic, discourse, and governance proxy features.

Example:
    python src/01_extract_features.py --raw_dir data/raw --out data/features/features.parquet

Input:
    data/raw/*.jsonl.gz

Output:
    data/features/features.parquet
"""
from __future__ import annotations

import argparse
import gzip
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, Any

import numpy as np
import pandas as pd
from tqdm.auto import tqdm


WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")
SENT_RE = re.compile(r"[.!?]+")
EMAIL_RE = re.compile(r"\b[\w\.-]+@[\w\.-]+\.\w+\b")
PHONE_RE = re.compile(r"(?:\+?\d[\d\-\s().]{7,}\d)")
URL_RE = re.compile(r"https?://|www\.")
CODE_RE = re.compile(
    r"(def |class |function\(|import |#include|<script|SELECT .* FROM|console\.log|public static void)",
    re.IGNORECASE,
)

BOILERPLATE_PATTERNS = [
    "all rights reserved",
    "privacy policy",
    "terms of use",
    "cookie policy",
    "subscribe to our newsletter",
    "click here",
    "advertisement",
    "related articles",
]

FIRST_PERSON = {"i", "me", "my", "mine", "we", "us", "our", "ours"}
SECOND_PERSON = {"you", "your", "yours"}
MODALS = {"should", "must", "may", "might", "could", "would", "can"}
DISCOURSE_MARKERS = [
    "however",
    "therefore",
    "moreover",
    "because",
    "although",
    "for example",
    "in contrast",
    "on the other hand",
    "first",
    "second",
    "finally",
]


def iter_jsonl_gz(path: Path) -> Iterable[Dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def safe_div(a: float, b: float) -> float:
    return float(a / b) if b else 0.0


def entropy(items: list[str]) -> float:
    counts = Counter(items)
    n = sum(counts.values())
    if n == 0:
        return 0.0
    return float(-sum((v / n) * math.log2(v / n) for v in counts.values()))


def features_for_text(text: str) -> Dict[str, float]:
    lower = text.lower()
    words = [w.lower() for w in WORD_RE.findall(text)]
    n_words = len(words)
    unique_words = set(words)
    sentences = [s for s in SENT_RE.split(text) if s.strip()]
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]

    first_count = sum(w in FIRST_PERSON for w in words)
    second_count = sum(w in SECOND_PERSON for w in words)
    modal_count = sum(w in MODALS for w in words)
    discourse_marker_count = sum(lower.count(dm) for dm in DISCOURSE_MARKERS)

    question_count = text.count("?")
    exclaim_count = text.count("!")
    quote_count = text.count('"') + text.count("'")
    colon_count = text.count(":")
    alpha_count = sum(ch.isalpha() for ch in text)

    list_marker_lines = sum(bool(re.match(r"^(\d+\.|[-*•])\s+", ln)) for ln in lines)
    dialogue_like_lines = sum(
        bool(re.match(r"^([A-Z][a-z]+|Q|A|User|Assistant|Speaker\s*\d*)\s*:", ln))
        for ln in lines
    )
    imperative_proxy_lines = sum(
        bool(re.match(r"^(click|see|read|use|make|try|add|remove|choose|open|go|check|write|select)\b", ln.lower()))
        for ln in lines
    )

    boilerplate_count = sum(lower.count(p) for p in BOILERPLATE_PATTERNS)
    pii_email_count = len(EMAIL_RE.findall(text))
    pii_phone_count = len(PHONE_RE.findall(text))
    url_count = len(URL_RE.findall(text))
    code_flag = int(bool(CODE_RE.search(text)))

    return {
        "char_count": len(text),
        "line_count": len(lines),
        "word_count": n_words,
        "sentence_count": len(sentences),
        "avg_word_len": float(np.mean([len(w) for w in words])) if words else 0.0,
        "avg_sentence_len_words": safe_div(n_words, len(sentences)),
        "type_token_ratio": safe_div(len(unique_words), n_words),
        "word_entropy_first_2000": entropy(words[:2000]),
        "first_person_rate": safe_div(first_count, n_words),
        "second_person_rate": safe_div(second_count, n_words),
        "modal_rate": safe_div(modal_count, n_words),
        "discourse_marker_rate": safe_div(discourse_marker_count, n_words),
        "question_per_1k_words": safe_div(question_count * 1000, n_words),
        "exclaim_per_1k_words": safe_div(exclaim_count * 1000, n_words),
        "quote_per_1k_chars": safe_div(quote_count * 1000, len(text)),
        "colon_per_1k_chars": safe_div(colon_count * 1000, len(text)),
        "list_marker_line_rate": safe_div(list_marker_lines, len(lines)),
        "dialogue_like_line_rate": safe_div(dialogue_like_lines, len(lines)),
        "imperative_proxy_line_rate": safe_div(imperative_proxy_lines, len(lines)),
        "boilerplate_count": boilerplate_count,
        "url_count": url_count,
        "pii_email_count": pii_email_count,
        "pii_phone_count": pii_phone_count,
        "pii_risk": int(pii_email_count > 0 or pii_phone_count > 0),
        "code_flag": code_flag,
        "uppercase_ratio": safe_div(sum(ch.isupper() for ch in text), alpha_count),
        "digit_ratio": safe_div(sum(ch.isdigit() for ch in text), len(text)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir", default="data/raw")
    parser.add_argument("--out", default="data/features/features.parquet")
    parser.add_argument("--max_chars_for_storage", type=int, default=2000)
    args = parser.parse_args()

    raw_paths = sorted(Path(args.raw_dir).glob("*.jsonl.gz"))
    if not raw_paths:
        raise FileNotFoundError(f"No .jsonl.gz files found in {args.raw_dir}")

    rows = []
    for path in raw_paths:
        for rec in tqdm(iter_jsonl_gz(path), desc=f"features {path.name}"):
            text = rec["text"]
            feat = features_for_text(text)

            row = {
                "doc_id": rec.get("doc_id"),
                "corpus": rec.get("corpus"),
                "dataset_id": rec.get("dataset_id"),
                "config": rec.get("config"),
                "split": rec.get("split"),
                "url": rec.get("url"),
                "source": rec.get("source"),
                "score": rec.get("score"),
                "text_excerpt": text[: args.max_chars_for_storage],
            }
            row.update(feat)
            rows.append(row)

    df = pd.DataFrame(rows)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)

    print(f"[done] wrote {len(df):,} rows -> {out}")
    print(df.groupby("corpus").size())


if __name__ == "__main__":
    main()
