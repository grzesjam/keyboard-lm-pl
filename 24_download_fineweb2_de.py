#!/usr/bin/env python3
"""
Streams Polish texts from FineWeb2-HQ and extracts individual sentences.

Source:  FineWeb2-HQ (HuggingFace FineData)
         https://huggingface.co/datasets/HuggingFaceFW/fineweb-2
License: ODC-By v1.0 (Open Data Commons Attribution)
         https://opendatacommons.org/licenses/by/1-0/
         Attribution: HuggingFace FineData / CommonCrawl

Configuration: pol_Latn (Polish)

Output: data/fineweb2_pl.txt

Usage:
  .venv_ml/bin/python 24_download_fineweb2_de.py [--target 80000000] [--resume]
"""

import argparse
import re
import sys
from pathlib import Path

OUT_FILE = Path("data/fineweb2_pl.txt")

SPLIT_PAT = re.compile(r'(?<=[.!?])\s+(?=[A-ZĄĆĘŁŃÓŚŹŻ\"\„\[])')
MIN_WORDS = 5
MAX_WORDS = 60


def split_document(text: str) -> list[str]:
    """Splits document text into individual sentences."""
    text = re.sub(r'\n{2,}', '\n', text)
    text = re.sub(r'[ \t]+', ' ', text)

    sentences = []
    for paragraph in text.split('\n'):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        parts = SPLIT_PAT.split(paragraph)
        for part in parts:
            part = part.strip()
            if not part:
                continue
            words = len(part.split())
            if words < MIN_WORDS or words > MAX_WORDS:
                continue
            if not re.search(r'[.!?]$', part):
                part = part
            sentences.append(part)
    return sentences


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=int, default=80_000_000,
                        help="Target sentence count (default: 80M)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume if file already exists")
    args = parser.parse_args()

    existing = 0
    if OUT_FILE.exists():
        if not args.resume:
            print(f"File already exists: {OUT_FILE}")
            print("Use --resume to continue or delete the file.")
            sys.exit(1)
        existing = sum(1 for _ in OUT_FILE.open(encoding="utf-8"))
        print(f"Resuming from {existing:,} existing sentences ...")

    if existing >= args.target:
        print(f"Target already reached: {existing:,} >= {args.target:,}")
        sys.exit(0)

    try:
        from datasets import load_dataset
    except ImportError:
        print("datasets not installed: pip install datasets")
        sys.exit(1)

    print(f"Streaming FineWeb2-HQ pol_Latn -> {OUT_FILE}")
    print(f"Target: {args.target:,} sentences ({existing:,} already present)\n")

    ds = load_dataset(
        "HuggingFaceFW/fineweb-2",
        "pol_Latn",
        split="train",
        streaming=True,
        trust_remote_code=False,
    )

    total = existing
    docs_processed = 0
    milestone = existing + 1_000_000

    with OUT_FILE.open("a", encoding="utf-8") as fout:
        for doc in ds:
            sentences = split_document(doc["text"])
            for s in sentences:
                fout.write(s + "\n")
            total += len(sentences)
            docs_processed += 1

            if total >= milestone:
                fout.flush()
                print(f"  {total/1_000_000:.1f}M sentences ({docs_processed:,} documents)", flush=True)
                milestone = total + 1_000_000

            if total >= args.target:
                break

    print(f"\nDone: {total:,} sentences -> {OUT_FILE}")
    print(f"Documents processed: {docs_processed:,}")


if __name__ == "__main__":
    main()
