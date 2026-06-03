#!/usr/bin/env python3
"""
Downloads Polish sentences from Tatoeba and saves them for inspection.

Output:
  data/tatoeba_pl.txt   one sentence per line
"""

import sys
import tarfile
import urllib.request
from pathlib import Path

URL = "https://downloads.tatoeba.org/exports/sentences.tar.bz2"
OUT  = Path("data/tatoeba_pl.txt")

def main():
    archive = Path("data/sentences.tar.bz2")
    if not archive.exists():
        print(f"Downloading {URL} ...")
        def progress(count, block, total):
            mb = count * block / 1024 / 1024
            sys.stdout.write(f"\r  {mb:.0f} MB downloaded")
            sys.stdout.flush()
        urllib.request.urlretrieve(URL, archive, reporthook=progress)
        print()

    print("Extracting Polish sentences ...")
    count = 0
    with tarfile.open(archive, "r:bz2") as tf:
        member = tf.getmember("sentences.csv")
        f = tf.extractfile(member)
        with OUT.open("w", encoding="utf-8") as out:
            for line in f:
                parts = line.decode("utf-8").rstrip("\n").split("\t", 2)
                if len(parts) == 3 and parts[1] == "pol":
                    out.write(parts[2] + "\n")
                    count += 1

    print(f"Done: {count:,} Polish sentences -> {OUT}")
    print(f"File size: {OUT.stat().st_size / 1024 / 1024:.1f} MB")

if __name__ == "__main__":
    main()
