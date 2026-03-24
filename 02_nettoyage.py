"""
Module 2 - Cleaner : Nettoyage du HTML Flask
Projet : doc_support_rag
Description : Nettoie les pages HTML brutes et extrait des sections structurées.

Corrections appliquées :
  - CORRECTION PRINCIPALE : extract_sections() préserve la hiérarchie H2/H3/H4
    au lieu de get_text() qui écrasait toute la structure
  - Les blocs de code sont liés à leur section HTML (pas globaux à la page)
  - Les <code> inline dans les <p> sont préservés proprement
  - Suppression des éléments de navigation/décoration avant extraction
  - Format de sortie : liste de sections {heading, text, code_blocks}
    au lieu d'un blob de texte
"""

import json
import re
import logging
from pathlib import Path
from bs4 import BeautifulSoup, Tag

INPUT_FILE  = Path("data_brute/flask_pages.json")
OUTPUT_FILE = Path("data_cleaned/flask_cleaned_pages.json")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ── Sélecteurs CSS à supprimer complètement ───────────────────────────────────
SELECTORS_TO_REMOVE = [
    "nav", "footer", "header", "aside", "script", "style", "noscript",
    ".headerlink",          # liens ¶ Sphinx
    ".sphinxsidebar",
    ".sphinxsidebarwrapper",
    ".related",
    ".navigation",
    ".breadcrumbs",
    ".clearer",
    ".footer",
    "[role='navigation']",
    "[role='banner']",
    "[role='contentinfo']",
    ".prev-next-area",
    ".prev", ".next",
    "#searchbox",
    ".search",
    ".contents.local",
    ".topic",
    ".admonition-title",    # "Note", "Warning" — labels Sphinx redondants
]

# Balises de titres reconnues comme séparateurs de section
HEADING_TAGS = {"h2", "h3", "h4"}


# ── Détection du langage ──────────────────────────────────────────────────────

def _detect_language(classes: str) -> str:
    c = classes.lower()
    if "python"     in c: return "python"
    if "bash"       in c or "shell"   in c or "console" in c: return "bash"
    if "html"       in c or "jinja"   in c: return "html/jinja"
    if "javascript" in c or "js"      in c: return "javascript"
    if "json"       in c: return "json"
    return "text"


# ── Extraction d'un bloc de code Sphinx ──────────────────────────────────────

def _extract_code_block(div: Tag) -> dict | None:
    """Extrait un dict {language, code} depuis un div.highlight Sphinx."""
    lang = _detect_language(" ".join(div.get("class", [])))
    pre  = div.find("pre")
    if not pre:
        return None
    code = pre.get_text().strip()
    # Nettoyer les numéros de ligne Sphinx (colonne gauche "1\n2\n3\n...")
    code = re.sub(r"^\d+\n", "", code, flags=re.MULTILINE)
    return {"language": lang, "code": code} if len(code) > 10 else None


# ── Nettoyage d'un paragraphe <p> ────────────────────────────────────────────

def _clean_paragraph(p_tag: Tag) -> str:
    """
    Extrait le texte d'un <p> en préservant les <code> inline lisiblement.
    Exemple : <p>Set <code>DEBUG=True</code> to enable.</p>
           → "Set `DEBUG=True` to enable."
    """
    for code in p_tag.find_all("code"):
        code.replace_with(f"`{code.get_text(strip=True)}`")
    return p_tag.get_text(" ", strip=True)


# ── Extraction structurée par sections H2/H3/H4 ──────────────────────────────

def extract_sections(soup: BeautifulSoup) -> list[dict]:
    """
    CORRECTION PRINCIPALE.
    Parcourt le DOM et regroupe le contenu par section (H2/H3/H4).
    Retourne une liste de :
        {
            "heading":     str,   # titre de section (vrai H2/H3, pas une phrase)
            "text":        str,   # texte nettoyé de la section
            "code_blocks": list   # blocs de code appartenant à cette section
        }

    Avant : get_text() → blob → chunker devait deviner les sections
    Après : les sections sont extraites directement depuis le HTML
    """
    sections          = []
    current_heading   = "Introduction"
    current_paragraphs = []
    current_codes     = []
    seen_codes        = set()   # éviter les doublons de code

    def _flush():
        """Sauvegarde la section courante si elle a du contenu."""
        text = "\n".join(current_paragraphs).strip()
        if len(text) >= 40 or current_codes:
            sections.append({
                "heading":     current_heading,
                "text":        text,
                "code_blocks": list(current_codes)
            })

    # Itérer uniquement sur les enfants directs du body principal
    # pour éviter les doublons dus à la récursion de find_all
    body = soup.find("div", class_="body") or soup.find("div", attrs={"role": "main"}) or soup

    for tag in body.descendants:
        if not isinstance(tag, Tag):
            continue

        # ── Nouveau heading → flush section précédente ────────────────────────
        if tag.name in HEADING_TAGS:
            _flush()
            current_heading    = tag.get_text(strip=True).replace("¶", "").strip()
            current_paragraphs = []
            current_codes      = []

        # ── Bloc de code Sphinx (div.highlight) ───────────────────────────────
        elif tag.name == "div" and any("highlight" in c for c in tag.get("class", [])):
            block = _extract_code_block(tag)
            if block and block["code"] not in seen_codes:
                seen_codes.add(block["code"])
                current_codes.append(block)

        # ── Paragraphe texte ──────────────────────────────────────────────────
        elif tag.name == "p":
            # Ignorer si le parent est déjà un div.highlight (éviter doublons)
            parent_classes = " ".join(tag.parent.get("class", []))
            if "highlight" in parent_classes:
                continue
            text = _clean_paragraph(tag)
            if len(text) > 20:
                current_paragraphs.append(text)

        # ── Listes (ul/ol) ────────────────────────────────────────────────────
        elif tag.name in ("ul", "ol"):
            # Ignorer les listes de navigation Sphinx
            parent = tag.find_parent(class_=re.compile(r"toctree|sidebar|nav"))
            if parent:
                continue
            items = [li.get_text(" ", strip=True) for li in tag.find_all("li", recursive=False)]
            items = [i for i in items if len(i) > 5]
            if items:
                current_paragraphs.append("\n".join(f"- {i}" for i in items))

        # ── Tableaux ──────────────────────────────────────────────────────────
        elif tag.name == "table":
            rows = []
            for tr in tag.find_all("tr"):
                cells = [td.get_text(" ", strip=True) for td in tr.find_all(["td", "th"])]
                rows.append(" | ".join(cells))
            if rows:
                current_paragraphs.append("\n".join(rows))

    # Flush dernière section
    _flush()
    return sections


# ── Nettoyage d'une page HTML ────────────────────────────────────────────────

def clean_page(html_content: str) -> list[dict]:
    """
    Nettoie une page HTML Flask.
    Retourne une liste de sections structurées.
    """
    soup = BeautifulSoup(html_content, "html.parser")

    # Supprimer navigation, décoration, liens ¶
    for selector in SELECTORS_TO_REMOVE:
        for tag in soup.select(selector):
            tag.decompose()

    return extract_sections(soup)


# ── Pipeline principal ────────────────────────────────────────────────────────

def run_cleaner():
    if not INPUT_FILE.exists():
        logger.error(f"Input file not found: {INPUT_FILE}. Run scraper first.")
        return []

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        raw_pages = json.load(f)

    logger.info(f"Cleaning {len(raw_pages)} pages...")
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    cleaned_pages = []
    skipped       = 0

    for page in raw_pages:
        sections = clean_page(page["html_content"])

        # Ignorer les pages sans contenu exploitable
        total_text = sum(len(s["text"]) for s in sections)
        if total_text < 50:
            logger.warning(f"Page ignorée (contenu vide) : {page['url']}")
            skipped += 1
            continue

        cleaned_pages.append({
            "url":              page["url"],
            "title":            page["title"],
            "sections":         sections,          # ← liste structurée (plus de blob)
            "num_sections":     len(sections),
            "num_code_blocks":  sum(len(s["code_blocks"]) for s in sections),
        })

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(cleaned_pages, f, ensure_ascii=False, indent=2)

    logger.info(
        f"\n✅ Cleaning : {len(cleaned_pages)} pages OK, {skipped} ignorées → {OUTPUT_FILE}"
    )
    return cleaned_pages


if __name__ == "__main__":
    run_cleaner()