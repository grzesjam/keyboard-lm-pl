use std::collections::{HashMap, HashSet};
use std::fs::{self, File};
use std::io::{BufRead, BufReader, BufWriter, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{Context, Result};
use clap::Parser;
use fancy_regex::Regex as FancyRegex;
use dashmap::DashSet;
use md5::{Digest, Md5};
use encoding_rs::{ISO_8859_16, UTF_8, WINDOWS_1250};
use rayon::prelude::*;
use regex::Regex;

const MIN_WORDS: usize = 4;
const MAX_WORDS: usize = 60;

/// ── Banned patterns: (regex_str, description, case_insensitive)
static WEB_ARTIFACTS: &[(&str, &str, bool)] = &[
    (r"(?:[^|\t\n]*(?:\||\t)){2,}", "Tabellenzeile (\u{2265}2 Pipes/Tabs) \u{2014} kein Flie\u{df}text", true),
    (r"(?:[^:]*:){3}", "\u{2265}3 Doppelpunkte (Produkt-/Listing-Spam)", true),
    (r"[*\u{2022}\u{b7}]{4,}", "Bullet-Spam (\u{2265}4 Sonderzeichen in Folge)", true),
    (r"(?:\.{2,}|\u{2026})\s*$", "Satzende mit .. / ... / \u{2026} (unvollst\u{e4}ndig/abgebrochen)", true),
    (r":\s*$", "Satzende mit Doppelpunkt (\u{dc}berschrift/Intro-Fragment)", true),
    (r"\[\.\.\.|\[\u{2026}\]", "Eckige Klammer mit Auslassungspunkten ([...]/[\u{2026}])", true),
    (r"[!?]{2,}\s*$", "Satzende mit !! oder ?? (Ausrufe-Spam)", true),
    (r"\s(?:Dr|Prof|Hr|Fr|Nr|Col|Gen|Lt|Cpt|St|Tel|Bd|Mio|Mrd|Jh|bzw|ggf|inkl|usw|vgl|vs|Abb|Abs|ca|max|min|Fam|o\.\s*g|o\.\s*[\u{e4}a]|u\.\s*[\u{e4}a]|u\.\s*a|z\.\s*B|d\.\s*h|i\.\s*d\.\s*R|gem)\.\s*$",
     "Satzende mit Abk\u{fc}rzung (abgeschnittener Satz)", true),
    (r"\s(?:[1-9]|[12]\d|3[01])\.\s*$",
     "Satzende mit Tagesdatum (abgeschnittener Satz: vom 22., bis zum 7.)", true),
    (r"^-\s+\S", "Zeile beginnt mit Listenpunkt (- item)", true),
    (r"^[\u{2022}\u{25aa}\u{25b8}\u{25b6}*|]", "Zeile beginnt mit Aufz\u{e4}hlungszeichen/Pipe (\u{2022}\u{25aa}\u{25b8}\u{25b6}*|)", true),
    (r"^\[", "Zeile beginnt mit [ (Blog-Tag, Kategorie-Header)", true),
    (r"(?:#.*?){3}", "\u{2265}3 Hashtags (Social-Media-Tag-Spam)", true),
    (r":\s*:", "Doppelter Doppelpunkt (Formular-/Template-Artefakt)", true),
    (r"\?[a-z\u{e4}\u{f6}\u{fc}\u{df}]{2}", "Fragezeichen statt \u{df} (Kodierungsfehler: Bekannterma?en)", true),
    (r"(?:[^>]*>){3}", "\u{2265}3 > (Breadcrumb-Navigation)", true),
    (r"[\u{2192}\u{279c}\u{27a1}\u{203a}\u{2039}]", "HTML-Navigationspfeil/Breadcrumb (\u{2192} \u{279c} \u{27a1} \u{203a} \u{2039} in Linktext)", true),
    (r"\[[A-Z\u{c4}\u{d6}\u{dc}][a-zA-Z\u{c4}\u{d6}\u{dc}\u{e4}\u{f6}\u{fc}\u{df}]{1,20}\]",
     "Bracket-Tag ([Top], [Kategorie]) \u{2014} Navigation/Template-Artefakt", false),
    (r"\[\s+[A-Z\u{c4}\u{d6}\u{dc}][a-zA-Z\u{c4}\u{d6}\u{dc}\u{e4}\u{f6}\u{fc}\u{df}]+\s+\]",
     "Bracket-Tag mit Leerzeichen ([ Bearbeiten ]) \u{2014} Wikipedia-Edit-Link", false),
    (r"\b[a-zA-Z0-9]*\d{3,}[a-zA-Z][a-zA-Z0-9-]*\b",
     "Alphanumerischer Code (SKU, Hash, Referenznummer)", true),
    (r"\bPDF\b", "PDF-Verweis (Buchdatenbank, Downloadseite)", true),
    (r"\bBy author\b", "Englischer Autorenhinweis (Katalog-Eintrag)", true),
    (r"(?:[:;=]-?[)D(\|PpOo\/\\]|\^\^+|[xX][Dd](?:\W|$)|<3)",
     "Text-Emoji / Emoticon (:D, ^^, xD, <3 \u{2026})", true),
    ("[\u{1f300}-\u{1faff}\u{2600}-\u{27bf}\u{2300}-\u{23ff}]",
     "Unicode-Emoji (\u{1f642}\u{1f389}\u{2665} \u{2026})", false),
    (r"[\u{2713}\u{2714}\u{2611}\u{2705}]", "H\u{e4}kchen-Symbol (\u{2713}\u{2714}\u{2611}\u{2705})", false),
    (r"FOTOS?:", "Fotokredit (FOTO:, FOTOS: ...)", true),
    (r"^[A-Z\u{c4}\u{d6}\u{dc}][A-Z\u{c4}\u{d6}\u{dc}\s\-!?.,:0-9]{15,}$",
     "All-Caps-Satz (Werbebanner/Forum-Header)", false),
    (r"\S*[a-z\u{e4}\u{f6}\u{fc}\u{df}][A-Z\u{c4}\u{d6}\u{dc}]\S*[a-z\u{e4}\u{f6}\u{fc}\u{df}][A-Z\u{c4}\u{d6}\u{dc}]\S*",
     "CamelCase-Run-on (2+ \u{dc}berg\u{e4}nge in einem Token)", false),
    (r"\b\d{1,2}\.\d{1,2}\.\d{4}\b",
     "Numerisches Datum (dd.mm.yyyy)", true),
    // backrefs handled separately
];

static LANG_PL: &[(&str, &str, bool)] = &[
    (r",\s*\d{2}-\d{3}\b", "Address (comma before postal code)", true),
    (r"\b(?:ulica|ul\.|aleja|al\.|plac|pl\.|osiedle|os\.)\s+\w+\s+\d+\b",
     "Address (street name + number)", true),
    (r"\bkoszt\w*\s+dostawy\b", "Shipping cost (E-commerce)", true),
    (r"\bdarmowa dostawa\b", "Free shipping (E-commerce)", true),
    (r"\bdo koszyka\b", "Add to cart (E-commerce)", true),
    (r"\bzam\u{f3}w\w+\s+ju\u{17c}\b", "Order now (E-commerce)", true),
    (r"\bZ powa\u{17c}aniem\b", "E-mail closing (Z powa\u{17c}aniem)", true),
    (r"\bPozdrawiam\b", "E-mail closing (Pozdrawiam)", true),
    (r"\b\d+\s*z\u{142}\b", "Price listing (z\u{142})", true),
    (r"\b\d+,\d{2}\s*z\u{142}\b", "Price listing (xx,xx z\u{142})", true),
    (r"\b(?:betclic|totolotek|totalizator|sts|fortuna|superbet)\b",
     "Gambling provider (betclic/sts/fortuna \u{2026})", true),
    (r"porn", "Pornographic content", true),
    (r"\bkomentarze\b", "Blog metadata (komentarze)", true),
    (r"\bnapisa\u{142}\w*\b", "Blog author line (napisa\u{142}/a)", true),
    (r"\bkliknij\s+tutaj\b", "Navigation CTA (kliknij tutaj)", true),
    (r"^\s*Tagi\s*:", "Blog tags (Tagi:)", true),
    (r"^\s*(?:Re\u{17c}yseria|Scenariusz|Zdj\u{119}cia|Muzyka|Monta\u{17c}|Produkcja)\s*:",
     "Movie/TV credits", true),
    (r"\bDodaj komentarz\b", "Comment box prompt", true),
    (r"\bczytaj\s+wi\u{119}cej\b", "Read more (web artifact)", true),
];

static LANG_EXCLUDE: &[(&str, &str, bool)] = &[
    (r"[\u{f5}\u{151}\u{171}\u{10d}\u{161}\u{17e}\u{101}\u{113}\u{12b}\u{16b}\u{163}\u{15f}\u{e7}\u{11f}\u{131}]",
     "Non-PL EU letter (\u{f5}\u{10d}\u{161}\u{17e}\u{101}\u{142}\u{e7}\u{11f}\u{131} \u{2014} Estonian/Latvian/Turkish \u{2026})", false),
    (r"[\u{fe}\u{de}]", "Thorn (Isl\u{e4}ndisch/Altenglisch \u{2014} kein deutsches Zeichen)", false),
    (r"[\u{e4}\u{f6}\u{fc}]{2}",
     "Verdoppelter Umlaut (\u{e4}\u{e4}/\u{f6}\u{f6}/\u{fc}\u{fc} \u{2014} Finnisch/Estnisch)", false),
    (r"\b\w*(?:owaja|tscheski|owskij|owuju)\b",
     "Transliteriertes Russisch (owaja/tscheski/owskij)", true),
    (r"[\u{0400}-\u{04ff}]", "Kyrillisch", false),
    (r"[\u{0600}-\u{06ff}]", "Arabisch", false),
    (r"[\u{4e00}-\u{9fff}]", "CJK-Zeichen (Chinesisch/Japanisch)", false),
    (r"[\u{ac00}-\u{d7af}]", "Hangul (Koreanisch)", false),
    (r"[\u{0900}-\u{097f}]", "Devanagari", false),
];

static SPEECH_ARTIFACTS: &[(&str, &str, bool)] = &[
    (r"\s+(und|oder|aber|denn|sondern)\.?\s*$", "Satz endet mit Konjunktion", true),
    (r"\u{0102}[^\x00-\x7f]", "Unreparabler Mojibake (\u{0102} + Sonderzeichen)", true),
];

/// Backreference patterns (can't be combined into alternation)
static BACKREF_PATTERNS: &[(&str, &str, bool)] = &[
    (r"\b(\w{4,})\s+\1\b", "Direktes Doppelwort (Keyword-Spam: Zierschotter Zierschotter)", true),
    (r"\b(\w{7,})\b\W{1,5}\b\1\b",
     "Nahes Doppelwort mit Trennzeichen (Keyword-Spam: esszimmer - esszimmer)", true),
    (r"(\w)\1{3,}", "Stottern/Wiederholung (aaaa)", true),
];

const DEFAULT_SOURCES: &[&str] = &[
    "data/tatoeba_pl.txt",
    "data/c4_pl.txt",
    "data/fineweb2_pl.txt",
];

/// ── Replacements ──────────────────────────────────────────────────────────
struct Replacement {
    re: Regex,
    repl: Replacer,
    desc: &'static str,
}

enum Replacer {
    Static(&'static str),
    Mojibake,
}

fn fix_mojibake(s: &str) -> String {
    let encodings = [ISO_8859_16, WINDOWS_1250];
    for enc in &encodings {
        let (bytes, _enc, had_errors) = enc.encode(s);
        if had_errors {
            continue;
        }
        let (result, _enc, had_errors) = UTF_8.decode(&bytes);
        if !had_errors {
            return result.into_owned();
        }
    }
    s.to_owned()
}

fn build_replacements() -> Vec<Replacement> {
    let mut r = Vec::new();

    // C1 control chars
    r.push(Replacement {
        re: Regex::new("[\u{80}-\u{9f}]").unwrap(),
        repl: Replacer::Static(""),
        desc: "C1 control chars (Windows-1252 artifacts)",
    });

    // Mojibake sequences: UTF-8 bytes read as Latin-1
    for (pat, replacement, desc) in [
        ("\u{e2}\u{80}\u{9e}", "", "Mojibake double low quote"),
        ("\u{e2}\u{80}\u{9c}", "", "Mojibake left double quote"),
        ("\u{e2}\u{80}\u{93}", "-", "Mojibake en dash"),
        ("\u{e2}\u{80}\u{91}", "'", "Mojibake apostrophe"),
        ("\u{e2}\u{80}\u{99}", "", "Mojibake right single quote"),
    ] {
        r.push(Replacement {
            re: Regex::new(pat).unwrap(),
            repl: Replacer::Static(replacement),
            desc,
        });
    }

    // Remove all types of quotation marks
    r.push(Replacement {
        re: Regex::new("[\"\u{201c}\u{201d}\u{201e}\u{ab}\u{bb}]").unwrap(),
        repl: Replacer::Static(""),
        desc: "Remove quotation marks",
    });

    // Angle brackets
    r.push(Replacement {
        re: Regex::new(r"[<>]").unwrap(),
        repl: Replacer::Static(""),
        desc: "Angle brackets (citation markers)",
    });

    // Soft hyphen
    r.push(Replacement {
        re: Regex::new("\u{ad}").unwrap(),
        repl: Replacer::Static(""),
        desc: "Soft hyphen (U+00AD) remove",
    });

    // Missing space after period
    r.push(Replacement {
        re: Regex::new(r"([a-z\u{105}\u{107}\u{119}\u{142}\u{144}\u{143}\u{15b}\u{17a}\u{17c}])\.([A-Z\u{104}\u{106}\u{118}\u{141}\u{143}\u{144}\u{15a}\u{179}\u{17b}])").unwrap(),
        repl: Replacer::Static("$1. $2"),
        desc: "Missing space after period",
    });

    // Missing space after ! or ?
    r.push(Replacement {
        re: Regex::new(r"([a-z\u{105}\u{107}\u{119}\u{142}\u{144}\u{143}\u{15b}\u{17a}\u{17c}])([!?])([A-Z\u{104}\u{106}\u{118}\u{141}\u{143}\u{144}\u{15a}\u{179}\u{17b}])").unwrap(),
        repl: Replacer::Static("$1$2 $3"),
        desc: "Missing space after ! or ?",
    });

    // Missing space after semicolon
    r.push(Replacement {
        re: Regex::new(r"([a-z\u{105}\u{107}\u{119}\u{142}\u{144}\u{143}\u{15b}\u{17a}\u{17c}]);([A-Za-z\u{104}\u{106}\u{118}\u{141}\u{143}\u{144}\u{15a}\u{179}\u{17b}\u{105}\u{107}\u{119}\u{142}\u{144}\u{143}\u{15b}\u{17a}\u{17c}])").unwrap(),
        repl: Replacer::Static("$1; $2"),
        desc: "Missing space after semicolon",
    });

    // ISO-8859-16/CP1250 Mojibake repair
    r.push(Replacement {
        re: Regex::new("\u{0102}[^\x00-\x7f]").unwrap(),
        repl: Replacer::Mojibake,
        desc: "ISO-8859-16/CP1250 Mojibake",
    });

    // Double space
    r.push(Replacement {
        re: Regex::new(r"  +").unwrap(),
        repl: Replacer::Static(" "),
        desc: "Double space",
    });

    // Double comma
    r.push(Replacement {
        re: Regex::new(r",\s*,").unwrap(),
        repl: Replacer::Static(","),
        desc: "Double comma",
    });

    r
}

/// ── Pattern compilation helpers ──────────────────────────────────────────

struct FastMatcherGroup {
    combined: Regex,
    individuals: Vec<(Regex, &'static str)>,
}

fn compile_patterns_ci(entries: &[(&'static str, &'static str, bool)]) -> (Vec<FastMatcherGroup>, Vec<(FancyRegex, &'static str)>) {
    let mut ci_pats: Vec<(Regex, &str)> = Vec::new();
    let mut cs_pats: Vec<(Regex, &str)> = Vec::new();

    for (pat_str, desc, ci) in entries {
        let re = Regex::new(pat_str).unwrap();
        if *ci {
            ci_pats.push((re, desc));
        } else {
            cs_pats.push((re, desc));
        }
    }

    let mut groups = Vec::new();

    for (pats, is_ci) in [(ci_pats, true), (cs_pats, false)] {
        if pats.is_empty() {
            continue;
        }
        let combined_str: String = pats
            .iter()
            .map(|(re, _)| format!("(?:{})", re.as_str()))
            .collect::<Vec<_>>()
            .join("|");
        let combined = if is_ci {
            Regex::new(&format!("(?i:{})", combined_str)).unwrap()
        } else {
            Regex::new(&combined_str).unwrap()
        };
        let individuals: Vec<(Regex, &str)> = pats.into_iter().collect();
        groups.push(FastMatcherGroup {
            combined,
            individuals,
        });
    }

    // Separate backreference patterns (not in fast groups)
    let backrefs: Vec<(FancyRegex, &str)> = BACKREF_PATTERNS
        .iter()
        .map(|(pat_str, desc, _)| (FancyRegex::new(pat_str).unwrap(), *desc))
        .collect();

    (groups, backrefs)
}

/// ── Byte range splitting ────────────────────────────────────────────────

fn get_byte_ranges(path: &Path, num_chunks: usize) -> Result<Vec<(u64, u64)>> {
    let file_size = fs::metadata(path)?.len();
    if file_size == 0 {
        return Ok(Vec::new());
    }
    let num_chunks = num_chunks.min(file_size as usize);
    let chunk_size = file_size / num_chunks as u64;

    let file = File::open(path)?;
    let mut reader = BufReader::new(file);
    let mut boundaries = vec![0u64];

    for i in 1..num_chunks {
        let target = i as u64 * chunk_size;
        reader.seek(SeekFrom::Start(target))?;
        // read raw bytes to next newline (avoids UTF-8 errors at arbitrary seek positions)
        let mut buf = Vec::<u8>::new();
        reader.read_until(b'\n', &mut buf)?;
        boundaries.push(reader.stream_position()?);
    }
    boundaries.push(file_size);

    Ok(boundaries
        .windows(2)
        .filter(|w| w[0] < w[1])
        .map(|w| (w[0], w[1]))
        .collect())
}

/// ── Chunk processing ────────────────────────────────────────────────────

#[derive(Default, Clone)]
struct ChunkStats {
    n_total: u64,
    n_kept: u64,
    n_ppl: u64,
    n_dedup: u64,
    n_short: u64,
    removed_reasons: HashMap<&'static str, u64>,
    replacements: HashMap<&'static str, u64>,
}

fn inc(map: &mut HashMap<&'static str, u64>, key: &'static str) {
    *map.entry(key).or_insert(0) += 1;
}

fn process_chunk(
    src: &Path,
    tmp_path: &Path,
    byte_start: u64,
    byte_end: u64,
    fast_groups: &[FastMatcherGroup],
    backref_regexes: &[(FancyRegex, &'static str)],
    high_ppl_set: &HashSet<String>,
    dedup_set: &DashSet<u64>,
    replacements: &[Replacement],
    dry_run: bool,
) -> Result<ChunkStats> {
    let file = File::open(src).context("open source file")?;
    let mut reader = BufReader::new(file);
    reader.seek(SeekFrom::Start(byte_start))?;

    let mut out: Option<BufWriter<File>> = if dry_run {
        None
    } else {
        let f = File::create(tmp_path).context("create temp file")?;
        Some(BufWriter::new(f))
    };

    let mut stats = ChunkStats::default();
    let mut line_bytes = Vec::<u8>::new();

    loop {
        // Check if we've read past our byte range
        let current_pos = reader.stream_position()?;
        if current_pos >= byte_end {
            break;
        }

        line_bytes.clear();
        let n = reader.read_until(b'\n', &mut line_bytes)?;
        if n == 0 {
            break;
        }

        stats.n_total += 1;

        let line = match std::str::from_utf8(&line_bytes) {
            Ok(s) => s,
            Err(_) => {
                // skip lines with invalid UTF-8
                continue;
            }
        };
        let stripped = line.trim();

        if stripped.is_empty() {
            continue;
        }

        // High PPL check
        if !high_ppl_set.is_empty() && high_ppl_set.contains(stripped) {
            stats.n_ppl += 1;
            continue;
        }

        // Dedup check
        if !dedup_set.is_empty() {
            let h = {
                let mut hasher = Md5::new();
                hasher.update(stripped.as_bytes());
                let result = hasher.finalize();
                let mut arr = [0u8; 8];
                arr.copy_from_slice(&result[..8]);
                u64::from_le_bytes(arr)
            };
            if !dedup_set.insert(h) {
                stats.n_dedup += 1;
                continue;
            }
        }

        // Banned pattern check
        let mut drop = false;

        'fast: for group in fast_groups {
            if group.combined.is_match(line) {
                for (pat, desc) in &group.individuals {
                    if pat.is_match(line) {
                        inc(&mut stats.removed_reasons, desc);
                        drop = true;
                        break 'fast;
                    }
                }
            }
        }

        if !drop {
            for (pat, desc) in backref_regexes {
                if pat.is_match(line).unwrap_or(false) {
                    inc(&mut stats.removed_reasons, desc);
                    drop = true;
                    break;
                }
            }
        }

        if drop {
            continue;
        }

        // Apply replacements
        let mut processed = line.to_string();
        for repl in replacements {
            match &repl.repl {
                Replacer::Static(s) => {
                    let result = repl.re.replace_all(&processed, *s);
                    if result != processed {
                        *stats.replacements.entry(repl.desc).or_insert(0) += 1;
                        processed = result.to_string();
                    }
                }
                Replacer::Mojibake => {
                    let result = repl.re.replace_all(&processed, |caps: &regex::Captures| {
                        fix_mojibake(&caps[0])
                    });
                    if result != processed {
                        *stats.replacements.entry(repl.desc).or_insert(0) += 1;
                        processed = result.to_string();
                    }
                }
            }
        }

        processed = processed.trim().to_string();

        if processed.is_empty() {
            continue;
        }

        // Capitalize first letter
        if let Some(c) = processed.chars().next() {
            let cap: String = c.to_uppercase().collect();
            processed = cap + &processed[c.len_utf8()..];
        }

        // Word count check
        let words = if processed.is_empty() {
            0
        } else {
            processed.split_whitespace().count()
        };
        if words < MIN_WORDS || words > MAX_WORDS {
            stats.n_short += 1;
            continue;
        }

        // Write kept line
        if let Some(ref mut fout) = out {
            writeln!(fout, "{}", processed)?;
        }
        stats.n_kept += 1;
    }

    Ok(stats)
}

/// ── CLI ─────────────────────────────────────────────────────────────────
#[derive(Parser)]
#[command(name = "clean-training-data")]
struct Cli {
    #[arg(long)]
    dry_run: bool,

    #[arg(long)]
    dedup: bool,

    #[arg(long, value_name = "FILE")]
    high_ppl: Option<PathBuf>,

    #[arg()]
    files: Vec<PathBuf>,
}

/// ── Default sources ─────────────────────────────────────────────────────
fn default_sources() -> Vec<PathBuf> {
    let mut sources: Vec<PathBuf> = DEFAULT_SOURCES.iter().map(|s| PathBuf::from(s)).collect();
    if let Ok(entries) = std::fs::read_dir("data") {
        let mut synthetic: Vec<PathBuf> = entries
            .filter_map(|e| e.ok())
            .filter(|e| {
                let name = e.file_name();
                let s = name.to_string_lossy();
                s.starts_with("synthetic_") && s.ends_with(".txt")
            })
            .map(|e| e.path())
            .collect();
        synthetic.sort();
        sources.extend(synthetic);
    }
    sources
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    // Compile patterns
    let all_banned: Vec<(&str, &str, bool)> = WEB_ARTIFACTS
        .iter()
        .chain(LANG_PL.iter())
        .chain(LANG_EXCLUDE.iter())
        .chain(SPEECH_ARTIFACTS.iter())
        .copied()
        .collect();

    let (fast_groups, backref_regexes) = compile_patterns_ci(&all_banned);
    let replacements = build_replacements();

    // Load high-ppl set
    let high_ppl_set: HashSet<String> = if let Some(ref ppl_path) = cli.high_ppl {
        let content = fs::read_to_string(ppl_path)
            .with_context(|| format!("Fehler: --high-ppl Datei nicht gefunden: {}", ppl_path.display()))?;
        let lines: HashSet<String> = content
            .lines()
            .map(|l| l.trim().to_string())
            .filter(|l| !l.is_empty())
            .collect();
        println!("{} S\u{e4}tze aus {} geladen (hohe Perplexity).", lines.len(), ppl_path.display());
        lines
    } else {
        HashSet::new()
    };

    // Determine sources
    let sources: Vec<PathBuf> = if cli.files.is_empty() {
        default_sources()
    } else {
        cli.files.clone()
    };

    // Stats
    let dedup_set: Arc<DashSet<u64>> = if cli.dedup {
        Arc::new(DashSet::new())
    } else {
        Arc::new(DashSet::new())
    };
    // If not dedup, we still create the set; it's just never populated.

    let num_workers = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4);

    let mut grand_total_removed: u64 = 0;

    // For each source, process in parallel chunks
    for src in &sources {
        if !src.exists() {
            continue;
        }

        let ranges = get_byte_ranges(src, num_workers)?;
        if ranges.is_empty() {
            continue;
        }

        let src_arc = Arc::new(src.clone());

        let chunk_results: Vec<(usize, ChunkStats)> = ranges
            .par_iter()
            .enumerate()
            .map(|(idx, &(start, end))| {
                let chunk_tmp = src_arc.with_extension(format!("{}.tmp", idx));
                let stats = process_chunk(
                    &src_arc,
                    &chunk_tmp,
                    start,
                    end,
                    &fast_groups,
                    &backref_regexes,
                    &high_ppl_set,
                    &dedup_set,
                    &replacements,
                    cli.dry_run,
                )
                .unwrap_or_else(|e| {
                    eprintln!("Fehler bei {} Chunk {}: {}", src_arc.display(), idx, e);
                    ChunkStats::default()
                });
                (idx, stats)
            })
            .collect();

        // Sort chunks by index
        let mut sorted_results: Vec<(usize, ChunkStats)> = chunk_results;
        sorted_results.sort_by_key(|(idx, _)| *idx);

        // Merge temp files into final
        let final_tmp = src.with_extension("tmp");
        let mut total_stats = ChunkStats::default();

        if !cli.dry_run {
            let mut fout = BufWriter::new(File::create(&final_tmp)?);
            for (idx, _) in &sorted_results {
                let chunk_tmp = src.with_extension(format!("{}.tmp", idx));
                let content = fs::read_to_string(&chunk_tmp)
                    .unwrap_or_default();
                fout.write_all(content.as_bytes())?;
                let _ = fs::remove_file(&chunk_tmp);
            }
            fout.flush()?;
        } else {
            // Still remove temp files
            for (idx, _) in &sorted_results {
                let chunk_tmp = src.with_extension(format!("{}.tmp", idx));
                let _ = fs::remove_file(&chunk_tmp);
            }
        }

        // Accumulate stats
        for (_, s) in &sorted_results {
            total_stats.n_total += s.n_total;
            total_stats.n_kept += s.n_kept;
            total_stats.n_ppl += s.n_ppl;
            total_stats.n_dedup += s.n_dedup;
            total_stats.n_short += s.n_short;
            for (desc, count) in &s.removed_reasons {
                *total_stats.removed_reasons.entry(desc).or_insert(0) += count;
            }
            for (desc, count) in &s.replacements {
                *total_stats.replacements.entry(desc).or_insert(0) += count;
            }
        }

        let n_removed = total_stats.n_total - total_stats.n_kept;
        grand_total_removed += n_removed;
        if n_removed == 0 {
            println!("{}: keine Treffer ({} S\u{e4}tze)", src.display(), total_stats.n_total);
            if !cli.dry_run {
                let _ = fs::remove_file(&final_tmp);
            }
            continue;
        }

        println!("{}: {} S\u{e4}tze entfernt (von {})", src.display(), n_removed, total_stats.n_total);
        if total_stats.n_short > 0 {
            println!("  {:>8}\u{d7}  zu kurz (< {} Woerter nach Cleaning)", total_stats.n_short, MIN_WORDS);
        }
        if total_stats.n_ppl > 0 {
            println!("  {:>8}\u{d7}  hohe Perplexit\u{e4}t (--high-ppl)", total_stats.n_ppl);
        }
        if total_stats.n_dedup > 0 {
            println!("  {:>8}\u{d7}  Duplikat (--dedup)", total_stats.n_dedup);
        }
        for (desc, count) in &total_stats.removed_reasons {
            if *count > 0 {
                println!("  {:>8}\u{d7}  {}", count, desc);
            }
        }
        for (desc, count) in &total_stats.replacements {
            if *count > 0 {
                println!("  {:>8}\u{d7}  {}", count, desc);
            }
        }

        if !cli.dry_run {
            fs::rename(&final_tmp, src)?;
        }
    }

    println!("\nGesamt entfernt: {}", grand_total_removed);
    if cli.dry_run {
        println!("(Dry-run \u{2014} keine Datei ver\u{e4}ndert)");
    }

    Ok(())
}
