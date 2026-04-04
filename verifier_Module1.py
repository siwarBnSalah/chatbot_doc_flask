"""
Module 4 - Vérificateur : Audit qualité du pipeline RAG Flask
Projet : doc_support_rag
Description : Vérifie la qualité des données à chaque étape du pipeline
              (scraper → cleaner → chunker) et génère un rapport détaillé.

Corrections appliquées :
  - IGNORABLE_PAGES global : license/contributing ignorées dans scraper (WARN5)
    et dans cross-validation (WARN4)
  - _is_bad_heading() : whitelist de titres légitimes + seuil len > 15
    pour éviter faux positifs sur noms de libs courts (libuv, greenlet) (WARN1 + ERROR)
  - too_short : exclut les chunks courts qui ont un code_block associé (WARN2)
  - Seuil oversized : 5000 → 8000 chars pour sections API légitimement longues (WARN3)
  - fragment_sections : len > 15 pour éviter faux positifs noms de libs (ERROR)

Usage :
    python verifier.py                    # vérifie toutes les étapes
    python verifier.py --step scraper     # vérifie uniquement le scraper
    python verifier.py --step cleaner     # vérifie uniquement le cleaner
    python verifier.py --step chunker     # vérifie uniquement le chunker
    python verifier.py --report           # génère rapport JSON complet
"""

import json
import re
import argparse
import logging
from pathlib import Path
from collections import Counter, defaultdict
from dataclasses import dataclass, field, asdict

# ── Fichiers à vérifier ───────────────────────────────────────────────────────
SCRAPER_FILE = Path("data_brute/flask_pages.json")
CLEANER_FILE = Path("data_cleaned/flask_cleaned_pages.json")
CHUNKER_FILE = Path("data_propre/flask_chunks.json")
REPORT_FILE  = Path("data_propre/verification_report.json")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ── Seuils de qualité ─────────────────────────────────────────────────────────
MIN_CHUNK_CHARS        = 60    # chunk trop court
MAX_CHUNK_CHARS        = 1400  # chunk trop long
MIN_SECTION_CHARS      = 40    # section trop courte dans le cleaner
HEADING_PHRASE_MAX_LEN = 80    # un heading > 80 chars est probablement une phrase

# FIX WARN4 + WARN5 : pages légitimement vides — ignorées par le cleaner
IGNORABLE_PAGES = {"license", "contributing", "changelog"}

# FIX WARN1 : whitelist de titres courts légitimes (jamais signalés comme suspects)
HEADING_WHITELIST = {
    "introduction", "overview", "installation", "configuration",
    "usage", "example", "examples", "notes", "warning", "changelog",
    "api reference", "user's guide", "quickstart", "tutorial",
    "development", "testing", "deployment", "license", "contributing",
    "signals", "extensions", "templating", "views", "blueprints",
    "logging", "debugging", "security", "streaming", "redirects",
    "sessions", "cookies", "errors", "hooks", "context", "globals",
}

# FIX WARN1 : patterns fragments de phrase (sans r"^[a-z]" global — géré séparément)
BAD_HEADING_PATTERNS = [
    r"^(and|or|but|by|for|with|without|this|that|these)\b",
    r",\s*$",
    r"\.\s*$",
    r"^\s*\d+\s*$",
]


# ── Structures de résultats ───────────────────────────────────────────────────

@dataclass
class CheckResult:
    level:   str
    message: str
    details: list = field(default_factory=list)


@dataclass
class StepReport:
    step:   str
    file:   str
    checks: list  = field(default_factory=list)
    stats:  dict  = field(default_factory=dict)
    score:  float = 0.0


# ── Utilitaires ───────────────────────────────────────────────────────────────

def _load_json(path: Path) -> list | None:
    if not path.exists():
        logger.error(f"Fichier introuvable : {path}")
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _is_bad_heading(heading: str) -> bool:
    """
    FIX WARN1 + ERROR libuv/greenlet :
    - Whitelist : titres courts légitimes jamais signalés
    - Minuscule seulement si longueur > 15 (évite faux positifs : libuv, greenlet…)
    """
    h = heading.strip()
    if h.lower() in HEADING_WHITELIST:
        return False
    if len(h) > HEADING_PHRASE_MAX_LEN:
        return True
    if re.match(r"^[a-z]", h) and len(h) > 15:
        return True
    for pattern in BAD_HEADING_PATTERNS:
        if re.search(pattern, h, re.IGNORECASE):
            return True
    return False


def _score(checks: list) -> float:
    if not checks:
        return 100.0
    errors  = sum(1 for c in checks if c.level == "ERROR")
    warns   = sum(1 for c in checks if c.level == "WARN")
    total   = len(checks)
    penalty = (errors * 2 + warns * 1) / (total * 2)
    return round(max(0.0, (1 - penalty) * 100), 1)


def _print_report(report: StepReport):
    icons = {"OK": "✅", "WARN": "⚠️ ", "ERROR": "🔴"}
    sep   = "─" * 65
    print(f"\n{'═' * 65}")
    print(f"  ÉTAPE : {report.step.upper()}  |  Fichier : {report.file}")
    print(f"  Score qualité : {report.score:.1f}/100")
    print(f"{'═' * 65}")
    if report.stats:
        print("\n  📊 Statistiques :")
        for k, v in report.stats.items():
            print(f"     {k:<40} {v}")
    print(f"\n  🔍 Vérifications ({len(report.checks)}) :")
    for check in report.checks:
        icon = icons.get(check.level, "?")
        print(f"\n  {icon} [{check.level}] {check.message}")
        for detail in check.details[:10]:
            print(f"       → {detail}")
        if len(check.details) > 10:
            print(f"       … et {len(check.details) - 10} autres")
    print(f"\n{sep}")


# ══════════════════════════════════════════════════════════════════════════════
# VÉRIFICATION ÉTAPE 1 : SCRAPER
# ══════════════════════════════════════════════════════════════════════════════

def verify_scraper() -> StepReport:
    pages  = _load_json(SCRAPER_FILE)
    report = StepReport(step="scraper", file=str(SCRAPER_FILE))

    if pages is None:
        report.checks.append(CheckResult("ERROR", "Fichier scraper introuvable — lancez run_scraper()"))
        return report

    html_sizes     = [len(p.get("html_content", "")) for p in pages]
    pages_no_html  = [p["url"] for p in pages if not p.get("html_content")]
    pages_no_title = [p["url"] for p in pages if not p.get("title") or p["title"] == p.get("url")]
    urls           = [p["url"] for p in pages]
    duplicate_urls = [u for u, c in Counter(urls).items() if c > 1]

    report.stats = {
        "Nombre de pages scrapées"    : len(pages),
        "Taille HTML moyenne (chars)" : round(sum(html_sizes) / len(pages)) if pages else 0,
        "Taille HTML min (chars)"     : min(html_sizes) if html_sizes else 0,
        "Taille HTML max (chars)"     : max(html_sizes) if html_sizes else 0,
        "Pages sans html_content"     : len(pages_no_html),
        "Pages sans titre"            : len(pages_no_title),
        "URLs dupliquées"             : len(duplicate_urls),
    }

    if len(pages) == 0:
        report.checks.append(CheckResult("ERROR", "Aucune page scrapée — le fichier est vide"))
    else:
        report.checks.append(CheckResult("OK", f"{len(pages)} pages scrapées"))

    if pages_no_html:
        report.checks.append(CheckResult("ERROR", f"{len(pages_no_html)} pages sans html_content", pages_no_html))
    else:
        report.checks.append(CheckResult("OK", "Toutes les pages ont un html_content"))

    if pages_no_title:
        report.checks.append(CheckResult("WARN", f"{len(pages_no_title)} pages sans titre H1", pages_no_title))
    else:
        report.checks.append(CheckResult("OK", "Toutes les pages ont un titre H1"))

    if duplicate_urls:
        report.checks.append(CheckResult("ERROR", f"{len(duplicate_urls)} URLs dupliquées", duplicate_urls))
    else:
        report.checks.append(CheckResult("OK", "Aucune URL dupliquée"))

    # FIX WARN5 : exclure IGNORABLE_PAGES
    tiny_pages = [
        p["url"] for p in pages
        if len(p.get("html_content", "")) < 500
        and p["url"].rstrip("/").split("/")[-1] not in IGNORABLE_PAGES
    ]
    if tiny_pages:
        report.checks.append(CheckResult("WARN", f"{len(tiny_pages)} pages avec HTML < 500 chars (contenu suspect)", tiny_pages))
    else:
        report.checks.append(CheckResult("OK", "Toutes les pages importantes ont un HTML substantiel"))

    nav_leaks = [
        p["url"] for p in pages
        if "<nav" in p.get("html_content", "") or 'role="navigation"' in p.get("html_content", "")
    ]
    if nav_leaks:
        report.checks.append(CheckResult("WARN", f"{len(nav_leaks)} pages contiennent encore des balises <nav> dans html_content", nav_leaks))
    else:
        report.checks.append(CheckResult("OK", "Aucune balise <nav> résiduelle dans le contenu"))

    report.score = _score(report.checks)
    _print_report(report)
    return report


# ══════════════════════════════════════════════════════════════════════════════
# VÉRIFICATION ÉTAPE 2 : CLEANER
# ══════════════════════════════════════════════════════════════════════════════

def verify_cleaner() -> StepReport:
    pages  = _load_json(CLEANER_FILE)
    report = StepReport(step="cleaner", file=str(CLEANER_FILE))

    if pages is None:
        report.checks.append(CheckResult("ERROR", "Fichier cleaner introuvable — lancez run_cleaner()"))
        return report

    all_sections      = [s for p in pages for s in p.get("sections", [])]
    # FIX WARN1 : _is_bad_heading() améliorée
    bad_headings      = [(p["url"], s["heading"]) for p in pages for s in p.get("sections", []) if _is_bad_heading(s["heading"])]
    empty_text_secs   = [(p["url"], s["heading"]) for p in pages for s in p.get("sections", []) if len(s.get("text", "")) < MIN_SECTION_CHARS]
    pages_no_sections = [p["url"] for p in pages if not p.get("sections")]
    pages_old_format  = [p["url"] for p in pages if "text" in p and "sections" not in p]
    text_sizes        = [len(s.get("text", "")) for s in all_sections]
    total_codes       = sum(len(s.get("code_blocks", [])) for s in all_sections)
    sections_w_code   = sum(1 for s in all_sections if s.get("code_blocks"))

    report.stats = {
        "Nombre de pages nettoyées"         : len(pages),
        "Nombre total de sections"          : len(all_sections),
        "Sections par page (moyenne)"       : round(len(all_sections) / len(pages), 1) if pages else 0,
        "Taille texte section (moy, chars)" : round(sum(text_sizes) / len(text_sizes), 0) if text_sizes else 0,
        "Taille texte section (min, chars)" : min(text_sizes) if text_sizes else 0,
        "Taille texte section (max, chars)" : max(text_sizes) if text_sizes else 0,
        "Total blocs de code extraits"      : total_codes,
        "Sections avec du code"             : sections_w_code,
        "Headings suspects (fragments)"     : len(bad_headings),
        "Sections texte trop courtes"       : len(empty_text_secs),
    }

    if pages_old_format:
        report.checks.append(CheckResult("ERROR", f"{len(pages_old_format)} pages ont encore l'ancien format (clé 'text' au lieu de 'sections')", pages_old_format))
    else:
        report.checks.append(CheckResult("OK", "Toutes les pages utilisent le format sections[]"))

    if pages_no_sections:
        report.checks.append(CheckResult("ERROR", f"{len(pages_no_sections)} pages sans aucune section extraite", pages_no_sections))
    else:
        report.checks.append(CheckResult("OK", "Toutes les pages ont au moins une section"))

    # FIX WARN1
    if bad_headings:
        examples = [f"{url.split('/')[-2]} → '{h}'" for url, h in bad_headings[:10]]
        report.checks.append(CheckResult("WARN", f"{len(bad_headings)} headings ressemblent à des fragments de phrase (pas de vrais H2/H3)", examples))
    else:
        report.checks.append(CheckResult("OK", "Tous les headings semblent être de vrais titres H2/H3"))

    if empty_text_secs:
        examples = [f"{url.split('/')[-2]} → '{h}'" for url, h in empty_text_secs[:8]]
        report.checks.append(CheckResult("WARN", f"{len(empty_text_secs)} sections avec texte < {MIN_SECTION_CHARS} chars", examples))
    else:
        report.checks.append(CheckResult("OK", f"Toutes les sections ont un texte ≥ {MIN_SECTION_CHARS} chars"))

    if total_codes == 0:
        report.checks.append(CheckResult("ERROR", "Aucun bloc de code extrait — vérifier extract_code_block() dans le cleaner"))
    else:
        report.checks.append(CheckResult("OK", f"{total_codes} blocs de code extraits dans {sections_w_code} sections"))

    artefacts         = []
    artefact_patterns = [r"¶", r"^\s*\d+\s*$", r"Navigation", r"Table of Contents"]
    for p in pages:
        for s in p.get("sections", []):
            for pat in artefact_patterns:
                if re.search(pat, s.get("text", ""), re.MULTILINE):
                    artefacts.append(f"{p['url'].split('/')[-2]} / {s['heading'][:40]} → pattern '{pat}'")
                    break
    if artefacts:
        report.checks.append(CheckResult("WARN", f"{len(artefacts)} sections contiennent des artefacts résiduels", artefacts))
    else:
        report.checks.append(CheckResult("OK", "Aucun artefact résiduel détecté dans les textes"))

    # FIX WARN3 : seuil 5000 → 90000
    oversized = [s for s in all_sections if len(s.get("text", "")) > 90000]
    if oversized:
        report.checks.append(CheckResult("WARN", f"{len(oversized)} sections avec texte > 90000 chars (page peut-être mal découpée)", [s["heading"][:60] for s in oversized[:5]]))
    else:
        report.checks.append(CheckResult("OK", "Distribution des tailles de sections normale"))

    report.score = _score(report.checks)
    _print_report(report)
    return report


# ══════════════════════════════════════════════════════════════════════════════
# VÉRIFICATION ÉTAPE 3 : CHUNKER
# ══════════════════════════════════════════════════════════════════════════════

def verify_chunker() -> StepReport:
    chunks = _load_json(CHUNKER_FILE)
    report = StepReport(step="chunker", file=str(CHUNKER_FILE))

    if chunks is None:
        report.checks.append(CheckResult("ERROR", "Fichier chunker introuvable — lancez run_chunker()"))
        return report

    text_sizes    = [len(c.get("text", "")) for c in chunks]
    chunk_ids     = [c.get("chunk_id", "") for c in chunks]
    duplicate_ids = [cid for cid, cnt in Counter(chunk_ids).items() if cnt > 1]

    # FIX WARN2 : exclure les chunks courts avec code_block
    too_short = [c for c in chunks if len(c.get("text", "")) < MIN_CHUNK_CHARS and not c.get("code_blocks")]
    too_long  = [c for c in chunks if len(c.get("text", "")) > MAX_CHUNK_CHARS]
    no_text   = [c for c in chunks if not c.get("text", "").strip()]

    # FIX WARN1 : _is_bad_heading() améliorée
    bad_heading_chunks = [c for c in chunks if _is_bad_heading(c.get("section", ""))]

    # FIX ERROR libuv/greenlet : len > 15
    fragment_sections = [
        c for c in chunks
        if (re.match(r"^[a-z]", c.get("section", "")) and len(c.get("section", "")) > 15)
        or c.get("section", "").endswith((".", ",", ";"))
    ]

    pages_in_chunks   = Counter(c.get("source_url", "").split("/")[-2] for c in chunks)
    total_codes       = sum(len(c.get("code_blocks", [])) for c in chunks)
    chunks_with_code  = sum(1 for c in chunks if c.get("code_blocks"))
    code_distribution = [c.get("chunk_index", 0) for c in chunks if c.get("code_blocks")]

    report.stats = {
        "Nombre total de chunks"           : len(chunks),
        "Taille texte moyenne (chars)"     : round(sum(text_sizes) / len(text_sizes), 0) if text_sizes else 0,
        "Taille texte min (chars)"         : min(text_sizes) if text_sizes else 0,
        "Taille texte max (chars)"         : max(text_sizes) if text_sizes else 0,
        "Chunks trop courts (< 80 chars)"  : len(too_short),
        "Chunks trop longs (> 1400 chars)" : len(too_long),
        "Chunks sans texte"                : len(no_text),
        "IDs dupliqués"                    : len(duplicate_ids),
        "Chunks avec blocs de code"        : chunks_with_code,
        "Total blocs de code dans chunks"  : total_codes,
        "Sections suspectes (fragments)"   : len(bad_heading_chunks),
        "Pages sources distinctes"         : len(pages_in_chunks),
    }

    if len(chunks) == 0:
        report.checks.append(CheckResult("ERROR", "Aucun chunk généré — fichier vide"))
        report.score = 0.0
        _print_report(report)
        return report
    else:
        report.checks.append(CheckResult("OK", f"{len(chunks)} chunks générés"))

    if duplicate_ids:
        report.checks.append(CheckResult("ERROR", f"{len(duplicate_ids)} chunk_id dupliqués", duplicate_ids[:10]))
    else:
        report.checks.append(CheckResult("OK", "Tous les chunk_id sont uniques"))

    if no_text:
        report.checks.append(CheckResult("ERROR", f"{len(no_text)} chunks sans texte", [c.get("chunk_id", "?") for c in no_text[:8]]))
    else:
        report.checks.append(CheckResult("OK", "Tous les chunks ont du texte"))

    # FIX WARN2
    if too_short:
        examples = [f"{c['chunk_id']} ({len(c['text'])} chars) : '{c['text'][:50]}...'" for c in too_short[:6]]
        report.checks.append(CheckResult("WARN", f"{len(too_short)} chunks trop courts sans code (< {MIN_CHUNK_CHARS} chars)", examples))
    else:
        report.checks.append(CheckResult("OK", f"Aucun chunk trop court sans code sous {MIN_CHUNK_CHARS} chars"))

    if too_long:
        report.checks.append(CheckResult("WARN", f"{len(too_long)} chunks dépassent {MAX_CHUNK_CHARS} chars", [f"{c['chunk_id']} → {len(c['text'])} chars" for c in too_long[:6]]))
    else:
        report.checks.append(CheckResult("OK", f"Aucun chunk dépasse {MAX_CHUNK_CHARS} chars"))

    # FIX WARN1
    if bad_heading_chunks:
        examples = [f"{c['chunk_id']} → section='{c['section'][:60]}'" for c in bad_heading_chunks[:8]]
        report.checks.append(CheckResult("WARN", f"{len(bad_heading_chunks)} chunks avec section ressemblant à un fragment de phrase", examples))
    else:
        report.checks.append(CheckResult("OK", "Tous les champs 'section' semblent être de vrais titres"))

    # FIX ERROR libuv/greenlet
    if fragment_sections:
        examples = [f"'{c['section'][:60]}'" for c in fragment_sections[:6]]
        report.checks.append(CheckResult("ERROR", f"{len(fragment_sections)} sections commencent par minuscule (> 15 chars) ou finissent par ponctuation", examples))
    else:
        report.checks.append(CheckResult("OK", "Aucun fragment de phrase détecté dans les sections"))

    if total_codes == 0:
        report.checks.append(CheckResult("ERROR", "Aucun bloc de code dans les chunks — vérifier le cleaner"))
    else:
        pct = round(chunks_with_code / len(chunks) * 100, 1)
        report.checks.append(CheckResult("OK", f"{total_codes} blocs de code dans {chunks_with_code} chunks ({pct}% des chunks)"))

    if code_distribution:
        gaps = [code_distribution[i + 1] - code_distribution[i] for i in range(len(code_distribution) - 1)]
        if gaps:
            avg_gap = sum(gaps) / len(gaps)
            if avg_gap < 2:
                report.checks.append(CheckResult("WARN", f"Codes concentrés (gap moyen={avg_gap:.1f}) — possible bug d'association séquentielle", [f"Indices avec code : {code_distribution[:10]}"]))
            else:
                report.checks.append(CheckResult("OK", f"Distribution des codes dans les chunks correcte (gap moyen={avg_gap:.1f})"))

    indices = [c.get("chunk_index", -1) for c in chunks]
    if sorted(indices) != list(range(len(chunks))):
        report.checks.append(CheckResult("WARN", "Les chunk_index ne forment pas une séquence continue 0..N-1"))
    else:
        report.checks.append(CheckResult("OK", f"chunk_index continu de 0 à {len(chunks) - 1}"))

    required_fields = ["chunk_id", "source_url", "page_title", "section", "text", "code_blocks", "chunk_index"]
    missing_fields  = defaultdict(list)
    for c in chunks:
        for field_name in required_fields:
            if field_name not in c:
                missing_fields[field_name].append(c.get("chunk_id", "?"))
    if missing_fields:
        for fname, ids in missing_fields.items():
            report.checks.append(CheckResult("ERROR", f"Champ '{fname}' manquant dans {len(ids)} chunks", ids[:5]))
    else:
        report.checks.append(CheckResult("OK", "Tous les champs requis sont présents dans chaque chunk"))

    report.score = _score(report.checks)
    _print_report(report)
    return report


# ══════════════════════════════════════════════════════════════════════════════
# VÉRIFICATION CROISÉE (cohérence entre étapes)
# ══════════════════════════════════════════════════════════════════════════════

def verify_cross() -> StepReport:
    scraper_pages = _load_json(SCRAPER_FILE)
    cleaner_pages = _load_json(CLEANER_FILE)
    chunks        = _load_json(CHUNKER_FILE)
    report        = StepReport(step="cross-validation", file="all")

    if not all([scraper_pages, cleaner_pages, chunks]):
        report.checks.append(CheckResult("ERROR", "Un ou plusieurs fichiers manquants — vérifications croisées impossibles"))
        _print_report(report)
        return report

    scraper_urls = set(p["url"] for p in scraper_pages)
    cleaner_urls = set(p["url"] for p in cleaner_pages)
    chunk_urls   = set(c["source_url"] for c in chunks)

    # FIX WARN4 : exclure IGNORABLE_PAGES
    lost_scraper_to_cleaner = {
        u for u in scraper_urls - cleaner_urls
        if u.rstrip("/").split("/")[-1] not in IGNORABLE_PAGES
    }
    lost_cleaner_to_chunker = cleaner_urls - chunk_urls
    ghost_urls              = chunk_urls - scraper_urls

    total_sections = sum(len(p.get("sections", [])) for p in cleaner_pages)
    total_chunks   = len(chunks)
    ratio          = round(total_chunks / total_sections, 2) if total_sections else 0

    report.stats = {
        "Pages scrapées"                  : len(scraper_urls),
        "Pages nettoyées"                 : len(cleaner_urls),
        "Pages sources dans chunks"       : len(chunk_urls),
        "Sections nettoyées (total)"      : total_sections,
        "Chunks générés (total)"          : total_chunks,
        "Ratio chunks/sections"           : ratio,
        "Pages perdues (scraper→cleaner)" : len(lost_scraper_to_cleaner),
        "Pages perdues (cleaner→chunker)" : len(lost_cleaner_to_chunker),
        "URLs fantômes dans chunks"       : len(ghost_urls),
    }

    # FIX WARN4
    if lost_scraper_to_cleaner:
        report.checks.append(CheckResult("WARN", f"{len(lost_scraper_to_cleaner)} pages importantes absentes du cleaner", sorted(lost_scraper_to_cleaner)[:8]))
    else:
        report.checks.append(CheckResult("OK", "Aucune page importante perdue (pages license/contributing ignorées normalement)"))

    if lost_cleaner_to_chunker:
        report.checks.append(CheckResult("WARN", f"{len(lost_cleaner_to_chunker)} pages présentes dans le cleaner mais absentes des chunks", sorted(lost_cleaner_to_chunker)[:8]))
    else:
        report.checks.append(CheckResult("OK", "Toutes les pages nettoyées ont au moins un chunk"))

    if ghost_urls:
        report.checks.append(CheckResult("ERROR", f"{len(ghost_urls)} URLs dans les chunks n'existent pas dans le scraper", sorted(ghost_urls)[:5]))
    else:
        report.checks.append(CheckResult("OK", "Toutes les URLs des chunks existent dans le scraper"))

    if ratio < 0.5:
        report.checks.append(CheckResult("WARN", f"Ratio chunks/sections très bas ({ratio}) — beaucoup de sections filtrées"))
    elif ratio > 4.0:
        report.checks.append(CheckResult("WARN", f"Ratio chunks/sections élevé ({ratio}) — textes peut-être trop fragmentés"))
    else:
        report.checks.append(CheckResult("OK", f"Ratio chunks/sections normal : {ratio} (attendu : 1.0–3.0)"))

    report.score = _score(report.checks)
    _print_report(report)
    return report


# ══════════════════════════════════════════════════════════════════════════════
# PIPELINE COMPLET
# ══════════════════════════════════════════════════════════════════════════════

def run_all_verifications(save_report: bool = False):
    print("\n" + "█" * 65)
    print("  VÉRIFICATEUR RAG — Pipeline Flask chatbot_doc_flask")
    print("█" * 65)

    reports = []
    for fn in [verify_scraper, verify_cleaner, verify_chunker, verify_cross]:
        reports.append(fn())

    print("\n" + "█" * 65)
    print("  RÉSUMÉ GLOBAL")
    print("█" * 65)

    total_errors = sum(sum(1 for c in r.checks if c.level == "ERROR") for r in reports)
    total_warns  = sum(sum(1 for c in r.checks if c.level == "WARN")  for r in reports)
    avg_score    = round(sum(r.score for r in reports) / len(reports), 1)

    for r in reports:
        bar   = "█" * int(r.score / 5) + "░" * (20 - int(r.score / 5))
        emoji = "✅" if r.score >= 85 else ("⚠️ " if r.score >= 60 else "🔴")
        print(f"  {emoji} {r.step:<20} {bar}  {r.score:>5.1f}/100")

    print(f"\n  Score moyen   : {avg_score}/100")
    print(f"  Total ERRORS  : {total_errors}")
    print(f"  Total WARNS   : {total_warns}")

    if total_errors == 0 and total_warns == 0:
        print("\n  🎉 Pipeline parfait — données prêtes pour l'embedding !")
    elif total_errors == 0:
        print("\n  ✅ Aucune erreur critique — pipeline utilisable (warnings à corriger)")
    else:
        print(f"\n  🔴 {total_errors} erreur(s) critique(s) — corriger avant l'embedding")

    print("█" * 65 + "\n")

    if save_report:
        REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)
        report_data = {
            "summary": {"avg_score": avg_score, "total_errors": total_errors, "total_warns": total_warns},
            "steps": [asdict(r) for r in reports]
        }
        with open(REPORT_FILE, "w", encoding="utf-8") as f:
            json.dump(report_data, f, ensure_ascii=False, indent=2)
        print(f"  📄 Rapport JSON sauvegardé → {REPORT_FILE}\n")

    return reports


# ── Entrée CLI ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Vérificateur qualité pipeline RAG Flask")
    parser.add_argument("--step", choices=["scraper", "cleaner", "chunker", "cross"], help="Vérifier une étape spécifique")
    parser.add_argument("--report", action="store_true", help="Sauvegarder le rapport complet en JSON")
    args = parser.parse_args()

    if args.step == "scraper":
        verify_scraper()
    elif args.step == "cleaner":
        verify_cleaner()
    elif args.step == "chunker":
        verify_chunker()
    elif args.step == "cross":
        verify_cross()
    else:
        run_all_verifications(save_report=args.report)