#!/usr/bin/env python3
"""
Bereinigt Trainingsdaten von unerwünschten Wörtern/Phrasen.

Füge neue unerwünschte Muster zur BANNED-Liste hinzu und führe das Skript
erneut aus — es filtert alle gefundenen Quelldateien in-place.

Unterstützt auch --high-ppl FILE: Liest eine Datei mit Sätzen (eine pro Zeile)
und entfernt exakt diese Sätze aus den Quelldateien.

Usage:
  .venv_ml/bin/python 10_clean_training_data.py [--dry-run]
  .venv_ml/bin/python 10_clean_training_data.py --high-ppl data/high_ppl.txt [--dry-run]
"""

import argparse
import hashlib
import html
import os
import re
import sys
from pathlib import Path

# ── Unerwünschte Muster ───────────────────────────────────────────────────────
# Jeder Eintrag: (pattern, beschreibung[, flags])
# Standard-Flags: re.IGNORECASE. Drittes Element = explizite Flags (z.B. 0).

# Sprach-unabhängige Web-Artefakte: Spam, Struktur, Format-Müll
WEB_ARTIFACTS = [
    # Tabellenzeilen, Listings
    (r"(?:[^|\t\n]*(?:\||\t)){2,}",  "Tabellenzeile (≥2 Pipes/Tabs) — kein Flie\xdftext"),
    (r"(?:[^:]*:){3}",      "≥3 Doppelpunkte (Produkt-/Listing-Spam)"),
    (r"[*•\xb7]{4,}",  "Bullet-Spam (≥4 Sonderzeichen in Folge)"),

    # Schlechte Satzenden
    (r"(?:\.{2,}|\u2026)\s*$", "Satzende mit .. / ... / … (unvollständig/abgebrochen)"),
    (r":\s*$",                    "Satzende mit Doppelpunkt (Überschrift/Intro-Fragment)"),
    (r"\[\.\.\.|\[…\]", "Eckige Klammer mit Auslassungspunkten ([...]/[…])"),
    (r"[!?]{2,}\s*$",       "Satzende mit !! oder ?? (Ausrufe-Spam)"),
    (r"\s(?:Dr|Prof|Hr|Fr|Nr|Col|Gen|Lt|Cpt|St|Tel|Bd|Mio|Mrd|Jh|bzw|ggf|inkl|usw|vgl|vs|Abb|Abs|ca|max|min|Fam|o\.\s*g|o\.\s*[äa]|u\.\s*[äa]|u\.\s*a|z\.\s*B|d\.\s*h|i\.\s*d\.\s*R|gem)\.\s*$",
     "Satzende mit Abkürzung (abgeschnittener Satz)"),
    (r"\s(?:[1-9]|[12]\d|3[01])\.\s*$",
     "Satzende mit Tagesdatum (abgeschnittener Satz: vom 22., bis zum 7.)"),

    # Strukturelle Artefakte
    (r"^-\s+\S",            "Zeile beginnt mit Listenpunkt (- item)"),
    (r"^[•▪▸►*|]",            "Zeile beginnt mit Aufzählungszeichen/Pipe (•▪▸►*|)"),
    (r"^\[",                 "Zeile beginnt mit [ (Blog-Tag, Kategorie-Header)"),
    (r"(?:#.*?){3}",         "≥3 Hashtags (Social-Media-Tag-Spam)"),
    (r":\s*:",               "Doppelter Doppelpunkt (Formular-/Template-Artefakt)"),
    (r"\bN\xe4chste\s+Seite\b", "Paginierungs-Navigation (Nächste Seite)", 0),
    (r"\?[a-zäöüß]{2}",     "Fragezeichen statt ß (Kodierungsfehler: Bekannterma?en)"),
    (r"^(?:(?![äöüÄÖÜß]).)*\b(?:the|find|visit|recent|your|with|from|that|this|have|will|are|were|been|their|they|what|which|when|where|how|you|our|more|also|most|some|all|can|not|for|its|but|and|has|was|his|her|of)\b(?:(?![äöüÄÖÜß]).)*$",
     "Englischer Satz (keine Umlaute + englische Funktionswörter)", re.IGNORECASE),
    (r"(?:[^>]*>){3}",       "≥3 > (Breadcrumb-Navigation)"),
    (r"\u2192|\u279c|\u27a1|\u203a|\u2039", "HTML-Navigationspfeil/Breadcrumb (→ ➜ ➡ › ‹ in Linktext)"),
    (r"\[[A-Z\xc4\xd6\xdc][a-zA-Z\xc4\xd6\xdc\xe4\xf6\xfc\xdf]{1,20}\]",
     "Bracket-Tag ([Top], [Kategorie]) — Navigation/Template-Artefakt"),
    (r"\[\s+[A-Z\xc4\xd6\xdc][a-zA-Z\xc4\xd6\xdc\xe4\xf6\xfc\xdf]+\s+\]",
     "Bracket-Tag mit Leerzeichen ([ Bearbeiten ]) — Wikipedia-Edit-Link"),

    # Codes und Referenznummern
    (r"\b[a-zA-Z0-9]*\d{3,}[a-zA-Z][a-zA-Z0-9-]*\b",
     "Alphanumerischer Code (SKU, Hash, Referenznummer)"),
    (r"\bPDF\b",             "PDF-Verweis (Buchdatenbank, Downloadseite)"),
    (r"\bBy author\b",       "Englischer Autorenhinweis (Katalog-Eintrag)"),

    # Emojis und Symbole
    (r"(?:[:;=]-?[)D(\|PpOo\/\\]|\^\^+|[xX][Dd](?!\w)|<3)",
     "Text-Emoji / Emoticon (:D, ^^, xD, <3 …)"),
    ("[\U0001F300-\U0001FAFF☀-➿⌀-⏿]",
     "Unicode-Emoji (\U0001F642\U0001F389♥ …)"),
    (r"[✓✔☑✅]", "H\xe4kchen-Symbol (✓✔☑✅)"),

    # Fotokredit (strukturell, nicht sprachspezifisch)
    (r"FOTOS?:",              "Fotokredit (FOTO:, FOTOS: ...)"),

    # All-Caps-Sätze (Forum-Header, Werbebanner, Clickbait-Titel)
    (r"^[A-Z\xc4\xd6\xdc][A-Z\xc4\xd6\xdc\s\-!?.,:0-9]{15,}$",
     "All-Caps-Satz (Werbebanner/Forum-Header)", 0),

    # CamelCase-Zusammenschreibungen: "SätzeDieNurAusSoEtwasBestanden"
    (r"\S*[a-z\xe4\xf6\xfc\xdf][A-Z\xc4\xd6\xdc]\S*[a-z\xe4\xf6\xfc\xdf][A-Z\xc4\xd6\xdc]\S*",
     "CamelCase-Run-on (2+ \xdcberg\xe4nge in einem Token)", 0),

    # Datum: numerisch (dd.mm.yyyy) — sprachübergreifend
    (r"\b\d{1,2}\.\d{1,2}\.\d{4}\b",
     "Numerisches Datum (dd.mm.yyyy)"),

    # Keyword-Spam: Wortwiederholung
    # Fängt "Zierschotter Zierschotter" (direkt) und "esszimmer - esszimmer" (mit Trennzeichen)
    (r"\b(\w{4,})\s+\1\b",
     "Direktes Doppelwort (Keyword-Spam: Zierschotter Zierschotter)"),
    (r"\b(\w{7,})\b\W{1,5}\b\1\b",
     "Nahes Doppelwort mit Trennzeichen (Keyword-Spam: esszimmer - esszimmer)"),
]

# Polish-specific filters
LANG_PL = [
    # Polish address patterns
    (r",\s*\d{2}-\d{3}\b",
     "Address (comma before postal code)"),
    (r"\b(?:ulica|ul\.|aleja|al\.|plac|pl\.|osiedle|os\.)\s+\w+\s+\d+\b",
     "Address (street name + number)"),

    # E-commerce spam
    (r"\bkoszt\w*\s+dostawy\b",  "Shipping cost (E-commerce)"),
    (r"\bdarmowa dostawa\b",     "Free shipping (E-commerce)"),
    (r"\bdo koszyka\b",          "Add to cart (E-commerce)"),
    (r"\bzamów\w+\s+już\b",     "Order now (E-commerce)"),

    # E-mail closings
    (r"\bZ poważaniem\b",
     "E-mail closing (Z poważaniem)"),
    (r"\bPozdrawiam\b",
     "E-mail closing (Pozdrawiam)"),

    # Numbers with Polish currency
    (r"\b\d+\s*zł\b",         "Price listing (zł)"),
    (r"\b\d+,\d{2}\s*zł\b",  "Price listing (xx,xx zł)"),

    # Casino/gambling SEO
    (r"\b(?:betclic|totolotek|totalizator|sts|fortuna|superbet)\b",
     "Gambling provider (betclic/sts/fortuna …)"),

    # Adult SEO
    (r"porn", "Pornographic content"),

    # Blog metadata
    (r"\bkomentarze\b",      "Blog metadata (komentarze)"),
    (r"\bnapisał\w*\b",      "Blog author line (napisał/a)"),
    (r"\bkliknij\s+tutaj\b", "Navigation CTA (kliknij tutaj)"),
    (r"^\s*Tagi\s*:",        "Blog tags (Tagi:)"),

    # Movie/TV credits
    (r"^\s*(?:Reżyseria|Scenariusz|Zdjęcia|Muzyka|Montaż|Produkcja)\s*:",
     "Movie/TV credits"),

    # Comment box UI
    (r"\bDodaj komentarz\b", "Comment box prompt"),
    (r"\bczytaj\s+więcej\b", "Read more (web artifact)"),
]

# Non-Polish scripts and characters (language exclusions)
LANG_EXCLUDE = [
    # Non-Polish European special characters
    (r"[õőűčšžāēīūţşçğı]",
     "Non-PL EU letter (\xf5čšžāłçğı — Estonian/Latvian/Turkish …)", 0),
    # Thorn: nur Isländisch/Altenglisch, nie Deutsch (auch: falsch konvertiertes ß)
    (r"[þÞ]",
     "Thorn (Isländisch/Altenglisch — kein deutsches Zeichen)", 0),
    # Verdoppelte Umlaute: typisch Finnisch/Estnisch, nie Deutsch
    # Shoutout to the fine people of Kouvola — your ää is beautiful, your New Year’s dynamite louder.
    (r"[\xe4\xf6\xfc]{2}",
     "Verdoppelter Umlaut (\xe4\xe4/\xf6\xf6/\xfc\xfc — Finnisch/Estnisch)", 0),
    # Transliteriertes Russisch in lateinischer Schrift
    (r"\b\w*(?:owaja|tscheski|owskij|owuju)\b", "Transliteriertes Russisch (owaja/tscheski/owskij)"),
    (r"[Ѐ-ӿ]",    "Kyrillisch"),
    (r"[؀-ۿ]",    "Arabisch"),
    (r"[一-鿿]",    "CJK-Zeichen (Chinesisch/Japanisch)"),
    (r"[가-힯]",    "Hangul (Koreanisch)"),
    (r"[ऀ-ॿ]",    "Devanagari"),
]

# ── Gesprochene Sprache / Transkripte ─────────────────────────────────────────
# Whisper-spezifische Artefakte aus Podcast-Transkripten
SPEECH_ARTIFACTS = [
    (r"\s+(und|oder|aber|denn|sondern)\.?\s*$",     "Satz endet mit Konjunktion"),
    (r"(\w)\1{3,}",                                   "Stottern/Wiederholung (aaaa)"),
    (r"Ă[^\x00-\x7f]",                               "Unreparabler Mojibake (Ă + Sonderzeichen)"),
]

BANNED = WEB_ARTIFACTS + LANG_PL + LANG_EXCLUDE + SPEECH_ARTIFACTS

# ── Ersetzungen ───────────────────────────────────────────────────────────────
# Jeder Eintrag: (compiled_pattern, replacement, beschreibung)
# Werden auf jede behaltene Zeile angewendet (in Reihenfolge).

def _try_encode(s: str, enc: str) -> bool:
    """Gibt True zurück wenn s.encode(enc).decode('utf-8') einen gültigen deutschen Text ergibt."""
    try:
        s.encode(enc).decode('utf-8')
        return True
    except (UnicodeEncodeError, UnicodeDecodeError):
        return False


def _build_replacements():
    pairs = [
        # C1 control chars (U+0080-U+009F): Windows-1252 artifacts
        ("[\x80-\x9f]", "", "C1 control chars (Windows-1252 artifacts)"),
        # Mojibake: UTF-8 bytes read as Windows-1252
        ("\xe2\x80\x9e", "",  "Mojibake double low quote"),
        ("\xe2\x80\x9c", "",  "Mojibake left double quote"),
        ("\xe2\x80\x93", "-", "Mojibake en dash"),
        ("\xe2\x80\x91", "'", "Mojibake apostrophe"),
        ("\xe2\x80\x99", "",  "Mojibake right single quote"),
        # Remove all types of quotation marks
        (r'["""„\xab\xbb]', "",  'Remove quotation marks'),
        # Remove angle brackets (citation markers)
        (r'[<>]', '', 'Angle brackets (citation markers)'),
        # Soft hyphen (U+00AD): formatting artifact
        ("\xad", "", "Soft hyphen (U+00AD) remove"),
        # Missing spaces after periods
        (r'([a-ząćęłńóśźż])\.([A-ZĄĆĘŁŃÓŚŹŻ])', r'\1. \2',
         'Missing space after period'),
        # Missing spaces after ! or ?
        (r'([a-ząćęłńóśźż])([!?])([A-ZĄĆĘŁŃÓŚŹŻ])', r'\1\2 \3',
         'Missing space after ! or ?'),
        # Missing space after semicolon
        (r'([a-ząćęłńóśźż]);([A-Za-zĄĆĘŁŃÓŚŹŻąćęłńóśźż])', r'\1; \2',
         'Missing space after semicolon'),
        # ISO-8859-16/CP1250 Mojibake repair
        (r'Ă[^\x00-\x7f]',
         lambda m: next(
             (m.group().encode(enc).decode('utf-8')
              for enc in ('iso-8859-16', 'cp1250')
              if _try_encode(m.group(), enc)),
             m.group()
         ),
         'ISO-8859-16/CP1250 Mojibake'),
        (r'  +', ' ', 'Double space'),
        (r',\s*,', ',', 'Double comma'),
    ]
    return [(re.compile(p, re.UNICODE), repl, desc) for p, repl, desc in pairs]

REPLACEMENTS = _build_replacements()

MIN_WORDS = 4   # Zeilen mit weniger Woertern nach allen Replacements verwerfen
MAX_WORDS = 60  # Podcast-Run-ons: Sätze über 60 Wörter raus

# ── Quelldateien ──────────────────────────────────────────────────────────────

SOURCES = [
    Path("data/tatoeba_pl.txt"),
    Path("data/c4_pl.txt"),
    Path("data/fineweb2_pl.txt"),
    *sorted(Path("data").glob("synthetic_*.txt")),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="Nur zählen, nichts schreiben")
    parser.add_argument("--dedup", action="store_true",
                        help="Duplikate entfernen (hash-basiert, ueber alle Quelldateien)")
    parser.add_argument("files", nargs="*", metavar="FILE",
                        help="Zu verarbeitende Dateien (Standard: alle SOURCES)")
    parser.add_argument("--high-ppl", metavar="FILE",
                        help="Datei mit S\xe4tzen hoher Perplexity (eine pro Zeile); "
                             "exakte \xdcbereinstimmungen werden entfernt")
    args = parser.parse_args()

    # Separate backreference patterns (can't be safely combined into one alternation)
    _backref_strs = {r'\b(\w{4,})\s+\1\b', r'\b(\w{7,})\b\W{1,5}\b\1\b'}
    _safe, _backref = [], []
    for entry in BANNED:
        (_backref if entry[0] in _backref_strs else _safe).append(entry)

    # Combine safe patterns per flag group — one search instead of ~35
    _groups: dict[int, list[str]] = {}
    _individuals: dict[int, list[tuple[re.Pattern, str]]] = {}
    for entry in _safe:
        p, desc = entry[0], entry[1]
        flags = entry[2] if len(entry) > 2 else re.IGNORECASE
        _groups.setdefault(flags, []).append(p)
        _individuals.setdefault(flags, []).append((re.compile(p, flags), desc))
    fast_matchers = [(re.compile('|'.join(f'(?:{p})' for p in grp), fl), _individuals[fl])
                     for fl, grp in _groups.items()]

    backref_matchers = [(re.compile(entry[0], entry[2] if len(entry) > 2 else re.IGNORECASE), entry[1])
                        for entry in _backref]

    # Optionale Menge von Sätzen aus --high-ppl
    high_ppl_set: set[str] = set()
    if args.high_ppl:
        ppl_path = Path(args.high_ppl)
        if not ppl_path.exists():
            print(f"Fehler: --high-ppl Datei nicht gefunden: {ppl_path}", file=sys.stderr)
            sys.exit(1)
        high_ppl_set = {l.strip() for l in ppl_path.read_text(encoding="utf-8").splitlines() if l.strip()}
        print(f"{len(high_ppl_set)} S\xe4tze aus {ppl_path} geladen (hohe Perplexity).")

    sources = [Path(f) for f in args.files] if args.files else SOURCES
    seen_hashes: set[int] = set()
    total_removed = 0

    for src in sources:
        if not src.exists():
            continue

        tmp = src.with_suffix(".tmp")
        removed_counts = {entry[1]: 0 for entry in BANNED}
        n_ppl_removed = 0
        n_kept = 0
        n_total = 0
        n_dedup_removed = 0
        n_short_removed = 0
        replacement_counts = {desc: 0 for _, _, desc in REPLACEMENTS}

        with src.open("r", encoding="utf-8") as fin, \
             (tmp.open("w", encoding="utf-8") if not args.dry_run else open(os.devnull, "w")) as fout:
            for line in fin:
                n_total += 1
                stripped = line.strip()

                if stripped in high_ppl_set:
                    n_ppl_removed += 1
                    continue

                if args.dedup:
                    h = int.from_bytes(hashlib.md5(stripped.encode()).digest()[:8], "little")
                    if h in seen_hashes:
                        n_dedup_removed += 1
                        continue
                    seen_hashes.add(h)

                drop = False
                for combined_pat, individuals in fast_matchers:
                    if combined_pat.search(line):
                        for pat, desc in individuals:
                            if pat.search(line):
                                removed_counts[desc] += 1
                                drop = True
                                break
                        if drop:
                            break
                if not drop:
                    for pat, desc in backref_matchers:
                        if pat.search(line):
                            removed_counts[desc] += 1
                            drop = True
                            break

                if not drop:
                    for pat, repl, desc in REPLACEMENTS:
                        new_line, n = pat.subn(repl, line)
                        if n:
                            replacement_counts[desc] += n
                            line = new_line
                    line = line.strip()
                    if line:
                        line = line[0].upper() + line[1:]
                    words = line.count(' ') + 1
                    if words < MIN_WORDS or words > MAX_WORDS:
                        n_short_removed += 1
                    else:
                        fout.write(line + "\n")
                        n_kept += 1

                if n_total % 5_000_000 == 0:
                    print(f"  {src.name}: {n_total:,} gelesen …", flush=True)

        n_removed = n_total - n_kept
        total_removed += n_removed

        if n_removed == 0:
            print(f"{src}: keine Treffer ({n_total:,} S\xe4tze)")
            if not args.dry_run:
                tmp.unlink()
            continue

        print(f"{src}: {n_removed:,} S\xe4tze entfernt (von {n_total:,})")
        if n_short_removed:
            print(f"  {n_short_removed:>8,}\xd7  zu kurz (< {MIN_WORDS} Woerter nach Cleaning)")
        if n_ppl_removed:
            print(f"  {n_ppl_removed:>8,}\xd7  hohe Perplexity (--high-ppl)")
        if n_dedup_removed:
            print(f"  {n_dedup_removed:>8,}\xd7  Duplikat (--dedup)")
        for desc, count in removed_counts.items():
            if count:
                print(f"  {count:>8,}\xd7  {desc}")

        for desc, count in replacement_counts.items():
            if count:
                print(f"  {count:>8,}\xd7  {desc}")

        if not args.dry_run:
            tmp.replace(src)

    print(f"\nGesamt entfernt: {total_removed:,}")
    if args.dry_run:
        print("(Dry-run — keine Datei ver\xe4ndert)")


if __name__ == "__main__":
    main()
