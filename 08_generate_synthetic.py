#!/usr/bin/env python3
"""
Generates high-quality Polish keyboard training sentences via a local LLM
(Ollama OpenAI-compatible API, e.g. Qwen3.6-27b).

The goal is a small but high-quality set that represents how people
actually type on a phone — short, natural, varied Polish.

Output: data/synthetic_pl.txt  (one sentence per line)

Usage:
  .venv_ml/bin/python 08_generate_synthetic.py [--target 2000] [--model qwen3:27b]
"""

import argparse
import random
import re
import sys
import time
from pathlib import Path

import json
import urllib.request

OUT = Path("data/synthetic_pl.txt")

# ── Topics & Style Variations ──────────────────────────────────────────────────
# Each entry: (topic, concrete_situation)
TOPICS = [
    ("Spotkania",       "Umawiasz się ze znajomą — godzina, miejsce, kto przychodzi"),
    ("Jedzenie",        "Piszesz komuś co chcesz zjeść, właśnie jesz lub ugotowałeś"),
    ("Praca",           "Krótkie wiadomości między współpracownikami — spotkanie, zadanie, koniec dnia"),
    ("Rodzina",         "Piszesz do mamy, partnera lub dzieci"),
    ("Zakupy",          "Lista zakupów, czego brakuje, czy ktoś ma coś przynieść"),
    ("Pogoda",          "Ktoś komentuje pogodę lub pyta o nią"),
    ("Potwierdzenie/Odmowa", "Krótkie potwierdzenia, odmowy, ustalenia"),
    ("W drodze",        "Ktoś jest w autobusie, samochodzie, idzie pieszo — daje znać"),
    ("Wolny czas",      "Plany na weekend, sport, film, koncert, spacer"),
    ("Krótkie info",    "Krótkie statusy — 'Zaraz będę', 'Udało się', 'Wszystko ok'"),
    ("Pytania",         "Codzienne pytania które ludzie zadają przez telefon"),
    ("Opinie",          "Ktoś krótko wyraża opinię o czymś codziennym"),
    ("Życzenia",        "Urodziny, egzamin, nowa praca — krótkie serdeczne wiadomości"),
    ("Przeprosiny",     "Krótkie przeprosiny, spóźnienie, zapomnienie"),
    ("Technologia",     "Krótkie zdania o telefonie, aplikacjach, internecie, baterii"),
]

STYLES = [
    "bardzo krótko i zwięźle (3-6 słów), jak piszesz gdy się spieszysz",
    "nieformalnie, na ty, luźno jak między znajomymi",
    "uprzejmie ale bezpośrednio, bez wypełniaczy",
    "sformułowane jako pytanie",
    "z wykrzyknikiem lub emoji na końcu (1-2 zdania z emoji, reszta bez)",
    "trochę dłuższe (10-15 słów), ale wciąż naturalne i płynne",
]

SYSTEM_PROMPT = """Generujesz polskie zdania tak jak prawdziwi ludzie piszą na smartfonie.

Ważne zasady:
- Mieszanka długości: około połowa 3-7 słów, reszta 8-15 słów, żadne zdanie nie dłuższe niż 15 słów
- Naturalne i mówione, BEZ zdań złożonych, nie styl Wikipedii
- Poprawna pisownia i gramatyka
- Bez cudzysłowów, numeracji, komentarzy
- Dokładnie jedno zdanie na linię, bez pustych linii między zdaniami
- Różnorodność: żadne zdanie nie może być podobne do poprzedniego

Dobre przykłady — mieszanka krótkich i średnich:
Ok, do zobaczenia!
Zaraz kończę.
Jadłeś już?
O której właściwie przyjeżdżasz?
Nie zdążę dzisiaj, sorry.
Możesz wpaść na chwilę, potrzebuję twojej pomocy.
Brzmi dobrze, robimy to.
Będę na stacji za 10 minut.
Co mam przynieść?
Spotkajmy się o 19 przed kinem."""


def build_user_prompt(topic: str, situation: str, style: str, n: int) -> str:
    return (
        f"Temat: {topic}\n"
        f"Sytuacja: {situation}\n"
        f"Styl: {style}\n\n"
        f"Napisz dokładnie {n} takich zdań, jedno na linię."
    )


def ollama_chat(host: str, model: str, system: str, user: str, max_tokens: int) -> str:
    payload = json.dumps({
        "model": model,
        "think": False,
        "stream": False,
        "options": {"num_predict": max_tokens},
        "messages": [
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
    }).encode()
    req = urllib.request.Request(
        f"{host}/api/chat",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        result = json.loads(resp.read())
    return result["message"]["content"]


def parse_sentences(raw: str) -> list[str]:
    """Extract clean sentences from model output."""
    lines = []
    for line in raw.splitlines():
        line = line.strip()
        # Remove numbering like "1.", "1)", "-", "•"
        line = re.sub(r'^[\d]+[.)]\s*', '', line)
        line = re.sub(r'^[-•]\s*', '', line)
        # Skip empty lines, meta-comments, or lines that look like headers
        if not line:
            continue
        if line.endswith(':') or len(line) < 5:
            continue
        # Skip lines with quotes around the whole thing
        if line.startswith('"') and line.endswith('"'):
            line = line[1:-1].strip()
        lines.append(line)
    return lines


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=int, default=2000,
                        help="Anzahl Sätze (default: 2000)")
    parser.add_argument("--model", default="qwen3.6:27b",
                        help="Ollama-Modellname (default: qwen3:27b)")
    parser.add_argument("--host", default="http://localhost:11434",
                        help="Ollama base URL (default: http://localhost:11434)")
    parser.add_argument("--batch", type=int, default=20,
                        help="Sätze pro API-Aufruf (default: 20)")
    parser.add_argument("--overwrite", action="store_true",
                        help="Bestehende Datei überschreiben statt anhängen")
    args = parser.parse_args()

    mode = "w" if args.overwrite else "a"
    existing = 0
    if not args.overwrite and OUT.exists():
        existing = sum(1 for _ in OUT.open())

    OUT.parent.mkdir(parents=True, exist_ok=True)
    collected = 0
    errors = 0
    t_start = time.time()

    print(f"Target: {args.target} sentences -> {OUT}")
    print(f"Model: {args.model}  |  Batch: {args.batch} sentences/call")
    if existing:
        print(f"Existing sentences: {existing} (will be appended)")
    print()

    with OUT.open(mode, encoding="utf-8") as f:
        while collected < args.target:
            topic, situation = random.choice(TOPICS)
            style = random.choice(STYLES)
            remaining = args.target - collected
            n = min(args.batch, remaining)

            user_msg = build_user_prompt(topic, situation, style, n)

            try:
                raw = ollama_chat(
                    host=args.host,
                    model=args.model,
                    system=SYSTEM_PROMPT,
                    user=user_msg,
                    max_tokens=n * 30,
                )
                sentences = parse_sentences(raw)

                if not sentences:
                    errors += 1
                    if errors > 5:
                        print("Too many empty responses — model running?", file=sys.stderr)
                        sys.exit(1)
                    continue

                for s in sentences:
                    f.write(s + "\n")
                f.flush()
                collected += len(sentences)

                elapsed = time.time() - t_start
                rate = collected / elapsed if elapsed > 0 else 0
                eta = (args.target - collected) / rate if rate > 0 else 0
                print(f"  [{collected:>5}/{args.target}]  "
                      f"Topic: {topic:<20}  "
                      f"{rate:.0f} sentences/s  "
                      f"ETA: {eta/60:.1f} min")

            except KeyboardInterrupt:
                print(f"\nCancelled. {collected} sentences saved.")
                break
            except Exception as e:
                print(f"  Error: {e}", file=sys.stderr)
                errors += 1
                time.sleep(2)

    total = time.time() - t_start
    print(f"\nDone: {collected} sentences in {total/60:.1f} min -> {OUT}")
    if collected > 0:
        print(f"Average: {total/collected:.2f}s per sentence")


if __name__ == "__main__":
    main()
