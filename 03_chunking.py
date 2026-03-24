"""
Module 3 - Chunker : Découpage en chunks cohérents
Projet : doc_support_rag
Description : Divise les pages nettoyées en chunks prêts pour l'embedding.

Corrections appliquées :
  - CORRECTION PRINCIPALE : utilise page["sections"] (H2/H3 réels)
    au lieu de tenter de deviner les headings depuis le texte brut
  - Association code ↔ section correcte (code lié à sa section HTML)
  - split_long_text() coupe sur frontières sémantiques (paragraphe > phrase > mot)
  - Overlap repositionné sur début de dernière phrase complète
  - chunk_id stable basé sur url_slug + compteur global (toujours unique)
  - MIN_CHUNK_CHARS respecté : chunks orphelins avec code mais peu de texte conservés
"""

import json
import re
import logging
from pathlib import Path
from dataclasses import dataclass, asdict

# ── Configuration ──────────────────────────────────────────────────────────────
INPUT_FILE        = Path("data_cleaned/flask_cleaned_pages.json")
OUTPUT_FILE       = Path("data_propre/flask_chunks.json")

CHUNK_MAX_CHARS   = 1200   # taille max d'un chunk texte (≈ 250-300 tokens)
CHUNK_OVERLAP     = 150    # overlap en caractères entre chunks successifs
MIN_CHUNK_CHARS   = 80     # ignorer les chunks trop courts (sans code associé)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ── Dataclass Chunk ───────────────────────────────────────────────────────────

@dataclass
class Chunk:
    chunk_id:    str    # identifiant unique et stable
    source_url:  str    # page d'origine
    page_title:  str    # titre H1 de la page
    section:     str    # titre H2/H3 réel de la section (plus de phrases courantes)
    text:        str    # texte du chunk
    code_blocks: list   # blocs de code de cette section (liés sémantiquement)
    chunk_index: int    # index global pour debug/tri


# ── Utilitaires ───────────────────────────────────────────────────────────────

def _sanitize_id(text: str) -> str:
    """Transforme une chaîne en slug utilisable dans un ID."""
    slug = re.sub(r"[^a-zA-Z0-9_]", "_", text)
    slug = re.sub(r"_+", "_", slug).strip("_")
    return slug[:40]


# ── Découpage d'un texte long ─────────────────────────────────────────────────

def split_long_text(text: str, max_chars: int, overlap: int) -> list[str]:
    """
    CORRECTION : découpe sur frontières sémantiques avec ordre de priorité :
      1. double saut de ligne (paragraphe)
      2. saut de ligne simple
      3. fin de phrase ". "
      4. espace (dernier recours)

    L'overlap est repositionné au début de la dernière phrase complète
    (et non sur un offset caractère arbitraire).
    """
    if len(text) <= max_chars:
        return [text.strip()] if text.strip() else []

    chunks = []
    start  = 0

    while start < len(text):
        end = start + max_chars

        if end >= len(text):
            tail = text[start:].strip()
            if tail:
                chunks.append(tail)
            break

        # Chercher la meilleure coupure dans la fenêtre [start+overlap, end]
        cut = -1
        for sep in ["\n\n", "\n", ". ", " "]:
            pos = text.rfind(sep, start + overlap, end)
            if pos > start:
                cut = pos + len(sep)   # inclure le séparateur dans le chunk
                break

        if cut <= start:
            cut = end   # forcer si aucune coupure naturelle trouvée

        chunk_text = text[start:cut].strip()
        if len(chunk_text) >= MIN_CHUNK_CHARS:
            chunks.append(chunk_text)

        # CORRECTION overlap : reculer jusqu'au début de la dernière phrase
        last_sentence = text.rfind(". ", start, cut)
        if last_sentence > start:
            start = last_sentence + 2
        else:
            start = cut   # pas de phrase trouvée → pas d'overlap

    return chunks


# ── Construction des chunks d'une page ───────────────────────────────────────

def build_chunks(page: dict, global_counter: list) -> list[Chunk]:
    """
    CORRECTION PRINCIPALE : itère sur page["sections"] (extraites par le cleaner)
    au lieu de tenter de détecter les headings par heuristique regex.

    Chaque section = {heading, text, code_blocks} → déjà propre et structuré.
    Les code_blocks sont liés à leur section HTML → association correcte.
    """
    url        = page["url"]
    page_title = page["title"]
    url_slug   = _sanitize_id(url.rstrip("/").split("/")[-1] or "index")
    chunks     = []

    for section in page.get("sections", []):
        heading     = section.get("heading", "Introduction").strip()
        text        = section.get("text", "").strip()
        code_blocks = section.get("code_blocks", [])

        # Conserver les sections avec code même si texte court
        if len(text) < MIN_CHUNK_CHARS and not code_blocks:
            continue

        # Découper le texte si nécessaire
        sub_texts = split_long_text(text, CHUNK_MAX_CHARS, CHUNK_OVERLAP) if text else [""]

        for sub_i, sub_text in enumerate(sub_texts):
            # Les code_blocks sont attachés uniquement au premier sub-chunk
            # (ils appartiennent à la section, pas à un sous-morceau arbitraire)
            associated_code = code_blocks if sub_i == 0 else []

            # chunk_id unique : url_slug + compteur global
            chunk_id = f"{url_slug}__chunk_{global_counter[0]:05d}"
            global_counter[0] += 1

            chunk = Chunk(
                chunk_id    = chunk_id,
                source_url  = url,
                page_title  = page_title,
                section     = heading,          # ← vrai titre H2/H3
                text        = sub_text,
                code_blocks = associated_code,  # ← lié à la section, pas séquentiel
                chunk_index = global_counter[0] - 1
            )
            chunks.append(chunk)

    return chunks


# ── Pipeline principal ────────────────────────────────────────────────────────

def run_chunker():
    if not INPUT_FILE.exists():
        logger.error(f"Input file not found: {INPUT_FILE}. Run cleaner first.")
        return []

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        cleaned_pages = json.load(f)

    logger.info(f"Chunking {len(cleaned_pages)} pages...")
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    all_chunks     = []
    global_counter = [0]   # compteur partagé mutable entre pages

    for page in cleaned_pages:
        page_chunks = build_chunks(page, global_counter)
        all_chunks.extend(page_chunks)
        logger.info(f"  {page['title'][:55]:<55} → {len(page_chunks):>3} chunks")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump([asdict(c) for c in all_chunks], f, ensure_ascii=False, indent=2)

    logger.info(f"\n✅ Chunking terminé : {len(all_chunks)} chunks totaux → {OUTPUT_FILE}")
    return all_chunks


if __name__ == "__main__":
    run_chunker()