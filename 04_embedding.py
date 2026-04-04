"""
03_embedding.py — Vectorisation et construction de la base ChromaDB
Projet : chatbot_doc_flask
Description :
    Charge les chunks (data_propre/flask_chunks.json),
    calcule leurs vecteurs d'embedding avec sentence-transformers
    (all-MiniLM-L6-v2, 384 dimensions) et les insère dans une
    collection ChromaDB persistante (vector_db/).

    Ce script est idempotent : relancé sur une base existante,
    il vérifie la cohérence et ne réinsère que si nécessaire.
    Utilisez --reset pour reconstruire depuis zéro.

Dépendances :
    pip install sentence-transformers chromadb torch

Usage :
    python 03_embedding.py           # construction complète
    python 03_embedding.py --reset   # repart de zéro
    python 03_embedding.py --stats   # statistiques uniquement
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any

import torch
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings

# ── Configuration ──────────────────────────────────────────────────────────────
INPUT_FILE      = Path("data_propre/flask_chunks.json")
CHROMA_DIR      = Path("vector_db")
COLLECTION_NAME = "flask_documentation"

EMBEDDING_MODEL = "all-MiniLM-L6-v2"   # 384 dims, ~22 MB
BATCH_SIZE      = 32                    # batch embedding (augmenter si GPU)
INSERT_BATCH    = 100                   # batch insertion ChromaDB

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ── Device info ───────────────────────────────────────────────────────────────

def _device_info() -> str:
    if torch.cuda.is_available():
        return f"cuda ({torch.cuda.get_device_name(0)})"
    if torch.backends.mps.is_available():
        return "mps (Apple Silicon)"
    return "cpu"


# ── Texte à encoder ───────────────────────────────────────────────────────────

def _build_embedding_text(chunk: dict[str, Any]) -> str:
    """
    Concatène page_title + section + text pour enrichir le contexte sémantique
    du vecteur. Le code est exclu : il apporte du bruit sémantique.
    """
    parts = []
    page_title = chunk.get("page_title", "").strip()
    section    = chunk.get("section", "").strip()
    text       = chunk.get("text", "").strip()

    if page_title:
        parts.append(page_title)
    if section and section != page_title:
        parts.append(section)
    if text:
        parts.append(text)

    return " | ".join(parts)


# ── Embedding ─────────────────────────────────────────────────────────────────

def compute_embeddings(
    chunks: list[dict],
    model: SentenceTransformer,
) -> list[list[float]]:
    """Calcule les embeddings de tous les chunks par lots."""
    texts = [_build_embedding_text(c) for c in chunks]

    logger.info(f"Calcul des embeddings : {len(texts)} chunks (batch={BATCH_SIZE})…")
    t0 = time.perf_counter()

    vectors = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,   # normalisation L2 → similarité cosinus optimale
    )

    elapsed = time.perf_counter() - t0
    logger.info(
        f"Embeddings calculés en {elapsed:.1f}s "
        f"({elapsed / len(chunks) * 1000:.1f} ms/chunk)"
    )
    return [v.tolist() for v in vectors]


# ── ChromaDB ──────────────────────────────────────────────────────────────────

def get_collection(reset: bool = False) -> chromadb.Collection:
    """Initialise le client ChromaDB et retourne la collection."""
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(
        path=str(CHROMA_DIR),
        settings=Settings(anonymized_telemetry=False),
    )
    logger.info(f"Client ChromaDB connecté → {CHROMA_DIR}")

    if reset:
        try:
            client.delete_collection(COLLECTION_NAME)
            logger.info(f"Collection '{COLLECTION_NAME}' supprimée (--reset)")
        except Exception:
            pass

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},   # index HNSW — distance cosinus
    )
    return collection


def _build_metadata(chunk: dict) -> dict:
    """
    Construit les métadonnées ChromaDB d'un chunk.
    ChromaDB n'accepte que str, int, float, bool — les listes sont JSON-sérialisées.
    """
    code_blocks = chunk.get("code_blocks", [])
    return {
        "source_url":  chunk.get("source_url", ""),
        "page_title":  chunk.get("page_title", ""),
        "section":     chunk.get("section", ""),
        "chunk_index": int(chunk.get("chunk_index", 0)),
        "has_code":    len(code_blocks) > 0,
        "num_codes":   len(code_blocks),
        "code_blocks": json.dumps(code_blocks, ensure_ascii=False),
    }

def insert_into_chromadb(
    chunks:     list[dict],
    embeddings: list[list[float]],
    collection: chromadb.Collection,
) -> int:
    """Insère chunks + embeddings dans ChromaDB par lots."""
    total    = len(chunks)
    inserted = 0

    logger.info(f"Insertion dans ChromaDB : {total} documents (batch={INSERT_BATCH})…")
    t0 = time.perf_counter()

    for start in range(0, total, INSERT_BATCH):
        end   = min(start + INSERT_BATCH, total)
        batch = chunks[start:end]
        vecs  = embeddings[start:end]

        collection.upsert(
            ids        = [c["chunk_id"] for c in batch],
            embeddings = vecs,
            documents  = [c.get("text", "") for c in batch],
            metadatas  = [_build_metadata(c) for c in batch],
        )
        inserted += len(batch)
        logger.info(f"  {inserted}/{total} ({inserted / total * 100:.0f}%)")

    elapsed = time.perf_counter() - t0
    logger.info(f"Insertion terminée en {elapsed:.1f}s")
    return inserted


# ── Statistiques ──────────────────────────────────────────────────────────────

def print_stats(collection: chromadb.Collection) -> None:
    """
    Affiche les statistiques de la collection ChromaDB.

    FIX : sample["embeddings"] retourne un numpy array — on ne peut pas
    évaluer sa vérité avec un simple `if`. On utilise `is not None`
    et on vérifie la longueur explicitement.
    """
    count   = collection.count()
    db_size = sum(
        f.stat().st_size for f in CHROMA_DIR.rglob("*") if f.is_file()
    ) / 1_048_576

    print(f"\n{'═' * 60}")
    print(f"  📊 STATISTIQUES — Collection '{collection.name}'")
    print(f"{'═' * 60}")
    print(f"  Documents indexés     : {count}")
    print(f"  Répertoire ChromaDB   : {CHROMA_DIR}")
    print(f"  Taille sur disque     : {db_size:.1f} MB")
    print(f"  Métrique de distance  : {collection.metadata.get('hnsw:space', 'N/A')}")

    if count > 0:
        # FIX : include=["embeddings"] pour récupérer les vecteurs
        sample = collection.peek(limit=1)
        if sample["ids"]:
            meta = sample["metadatas"][0]

            # FIX : vérification explicite sans évaluer le numpy array comme booléen
            embeddings_data = sample.get("embeddings")
            if embeddings_data is not None and len(embeddings_data) > 0:
                dim = len(embeddings_data[0])
            else:
                dim = "N/A"

            print(f"\n  Exemple de document :")
            print(f"  ├─ ID         : {sample['ids'][0]}")
            print(f"  ├─ Page       : {meta.get('page_title', 'N/A')}")
            print(f"  ├─ Section    : {meta.get('section', 'N/A')[:50]}")
            print(f"  ├─ URL        : {meta.get('source_url', 'N/A')}")
            print(f"  ├─ A du code  : {'Oui' if meta.get('has_code') else 'Non'}")
            print(f"  └─ Dimension  : {dim}")

    print(f"{'═' * 60}\n")


# ── Pipeline principal ────────────────────────────────────────────────────────

def run(reset: bool = False, stats_only: bool = False) -> None:
    collection = get_collection(reset=reset)

    if stats_only:
        print_stats(collection)
        return

    # Court-circuit si déjà peuplé
    existing = collection.count()
    if existing > 0 and not reset:
        logger.info(
            f"Collection '{COLLECTION_NAME}' déjà peuplée ({existing} docs). "
            "Utilisez --reset pour reconstruire."
        )
        print_stats(collection)
        return

    # Vérification fichier source
    if not INPUT_FILE.exists():
        logger.error(
            f"Fichier introuvable : {INPUT_FILE}\n"
            "Lancez d'abord : python 02_nettoyage.py"
        )
        return

    with open(INPUT_FILE, encoding="utf-8") as f:
        chunks: list[dict] = json.load(f)
    logger.info(f"{len(chunks)} chunks chargés — device : {_device_info()}")

    # Chargement du modèle
    logger.info(f"Chargement du modèle '{EMBEDDING_MODEL}'…")
    model = SentenceTransformer(EMBEDDING_MODEL)
    dim   = model.get_sentence_embedding_dimension()
    logger.info(f"Modèle chargé — {dim} dimensions")

    # Calcul des embeddings
    embeddings = compute_embeddings(chunks, model)

    # Insertion dans ChromaDB
    inserted = insert_into_chromadb(chunks, embeddings, collection)

    # Rapport final
    print_stats(collection)
    logger.info(
        f"\n✅ Vectorisation terminée :\n"
        f"   Modèle          : {EMBEDDING_MODEL} ({dim} dims)\n"
        f"   Chunks insérés  : {inserted}\n"
        f"   Base ChromaDB   : {CHROMA_DIR}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="03_embedding.py — Vectorisation et construction ChromaDB"
    )
    parser.add_argument(
        "--reset", action="store_true",
        help="Supprime et reconstruit la collection depuis zéro"
    )
    parser.add_argument(
        "--stats", action="store_true",
        help="Affiche uniquement les statistiques de la base existante"
    )
    args = parser.parse_args()
    run(reset=args.reset, stats_only=args.stats)