#!/usr/bin/env python3
"""
Generates a clean eval set from freshly synthesized Polish sentences.
Sentences are not part of the training corpus — suitable for fair model comparison.

Requires Ollama with qwen3.6:27b running locally.

Usage:
  .venv_ml/bin/python 13_generate_eval_set.py [--per-topic 50] [--out data/eval_clean.txt]
"""
import argparse
import json
import random
import urllib.request
from pathlib import Path

TOPICS = [
    ("Wiadomości",   "lokalne wydarzenia, polityka, społeczeństwo"),
    ("Gotowanie",    "gotowanie, przepisy, jedzenie, kuchnia"),
    ("Ogród",        "ogród, rośliny, ogrodnictwo, natura"),
    ("Majsterkowanie", "naprawy, narzędzia, majsterkowanie"),
    ("Zdrowie",      "zdrowie, wizyta u lekarza, sport, samopoczucie"),
    ("Podróże",      "podróże, wakacje, miejsca, noclegi, wycieczki"),
    ("Rodzina",      "życie rodzinne, dzieci, krewni, dom"),
    ("Praca",        "codzienna praca, biuro, współpracownicy, spotkania, kariera"),
    ("Sport",        "sport, trening, piłka nożna, fitness, zawody"),
    ("Natura",       "pogoda, zwierzęta, krajobraz, pory roku"),
]

HOST  = "http://localhost:11434"
MODEL = "qwen3.6:27b"


def ask_batch(topic: str) -> str:
    payload = json.dumps({
        "model": MODEL,
        "think": False,
        "stream": False,
        "options": {"num_predict": 400, "temperature": 0.9},
        "messages": [
            {"role": "system", "content":
                "Generujesz naturalne polskie zdania codzienne dla modelu językowego klawiatury. "
                "Podawaj wyłącznie zdania, jedno na linię, bez numeracji, "
                "bez wyjaśnień, bez cudzysłowów."},
            {"role": "user", "content":
                f"Napisz 10 różnych naturalnych polskich zdań na temat: {topic}"},
        ],
    }).encode()
    req = urllib.request.Request(f"{HOST}/api/chat", data=payload,
                                  headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read())["message"]["content"].strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-topic", type=int, default=50)
    parser.add_argument("--out", default="data/eval_clean.txt")
    parser.add_argument("--seed", type=int, default=9999)
    args = parser.parse_args()

    random.seed(args.seed)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    with out.open("w", encoding="utf-8") as f:
        for topic_name, topic_desc in TOPICS:
            print(f"  {topic_name}...", flush=True)
            count = 0
            while count < args.per_topic:
                try:
                    raw = ask_batch(topic_desc)
                    for line in raw.splitlines():
                        line = line.strip().lstrip("0123456789.-•) ").strip("\"'")
                        if len(line) > 20 and line[0].isupper() and count < args.per_topic:
                            f.write(line + "\n")
                            f.flush()
                            count += 1
                            total += 1
                except Exception as e:
                    print(f"    Fehler: {e}", flush=True)
            print(f"    {count} Sätze ({total} gesamt)", flush=True)

    print(f"Fertig: {total} Sätze → {out}")


if __name__ == "__main__":
    main()
