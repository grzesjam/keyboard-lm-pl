#!/usr/bin/env bash
# Downloads the German OpenSubtitles 2018 monolingual corpus (~475 MB gz).
set -euo pipefail

URL="https://object.pouta.csc.fi/OPUS-OpenSubtitles/v2018/mono/pl.txt.gz"
OUT="data/opensubtitles/pl.txt.gz"

mkdir -p data/opensubtitles

if [[ -f "$OUT" ]]; then
    echo "Bereits heruntergeladen: $OUT"
    exit 0
fi

echo "Lade OpenSubtitles PL herunter ..."
wget -c --progress=bar:force "$URL" -O "$OUT"
echo "Fertig: $OUT"
