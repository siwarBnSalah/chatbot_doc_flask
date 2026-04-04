"""
05_chatbot.py — Moteur RAG : Retrieval + Prompt + LLM (Mistral via Ollama)
Projet : chatbot_doc_flask
Description :
    Fournit la classe FlaskChatbot qui orchestre le pipeline RAG complet :
        1. Encode la question avec sentence-transformers
        2. Recherche les top-k chunks pertinents dans ChromaDB
        3. Construit un prompt augmenté avec les extraits de documentation
        4. Appelle Mistral via l'API Ollama locale
        5. Retourne une réponse sourcée et structurée

    Ce module est importé par app.py (interface Streamlit).
    Il peut aussi être utilisé en ligne de commande pour tester.

Prérequis :
    - Ollama installé et lancé : https://ollama.ai
    - Modèle Mistral téléchargé : ollama pull mistral
    - Base ChromaDB construite   : python 04_embedding.py

Usage :
    python 05_chatbot.py "How to configure Flask debug mode?"
    python 05_chatbot.py --interactive
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import requests
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

# ── Configuration ──────────────────────────────────────────────────────────────
CHROMA_DIR       = Path("vector_db")
COLLECTION_NAME  = "flask_documentation"
EMBEDDING_MODEL  = "all-MiniLM-L6-v2"

OLLAMA_BASE_URL  = "http://localhost:11434"
OLLAMA_MODEL     = "mistral"
OLLAMA_TIMEOUT   = 120   # secondes

TOP_K            = 4     # nombre de chunks récupérés par requête
MIN_SCORE        = 0.25  # seuil minimum de pertinence
MAX_CONTEXT_CHARS = 3500 # taille max du contexte injecté dans le prompt

# ── System prompt ──────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """Tu es un assistant technique spécialisé dans la documentation officielle Flask.

Règles strictes :
1. Réponds UNIQUEMENT en te basant sur les extraits de documentation fournis dans le contexte.
2. Si l'information n'est pas dans les extraits, réponds : "Je ne trouve pas cette information dans la documentation Flask fournie."
3. Cite toujours la section source à la fin de ta réponse (format : Source : [page] — [section]).
4. Si un extrait de code est disponible, inclus-le dans ta réponse en le formatant correctement.
5. Réponds en français si la question est en français, en anglais si la question est en anglais.
6. Ne fabrique aucune information qui ne serait pas dans les extraits fournis.
"""

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ── Structures de données ─────────────────────────────────────────────────────

@dataclass
class RetrievalResult:
    """Un chunk retourné par la recherche sémantique."""
    chunk_id:    str
    text:        str
    page_title:  str
    section:     str
    source_url:  str
    score:       float
    code_blocks: list[dict] = field(default_factory=list)
    has_code:    bool = False

    def to_context_block(self) -> str:
        """Formate le chunk en bloc de contexte pour le prompt."""
        lines = [
            f"[SOURCE] {self.page_title} — {self.section}",
            f"[URL] {self.source_url}",
            f"[CONTENU]\n{self.text}",
        ]
        for block in self.code_blocks:
            lang = block.get("language", "")
            code = block.get("code", "")
            if code:
                lines.append(f"[CODE {lang.upper()}]\n```{lang}\n{code}\n```")
        return "\n".join(lines)


@dataclass
class ChatResponse:
    """Réponse complète du chatbot RAG."""
    question:   str
    answer:     str
    sources:    list[dict]      # [{page_title, section, source_url, score}]
    chunks_used: int
    model:      str
    error:      str | None = None

    @property
    def has_error(self) -> bool:
        return self.error is not None


# ── Classe principale : FlaskChatbot ─────────────────────────────────────────

class FlaskChatbot:
    """
    Chatbot RAG sur la documentation Flask.

    Exemple d'utilisation :
        chatbot  = FlaskChatbot()
        response = chatbot.ask("How to create a Flask route?")
        print(response.answer)
        for source in response.sources:
            print(source["source_url"])
    """

    def __init__(
        self,
        chroma_dir:      str | Path = CHROMA_DIR,
        collection_name: str        = COLLECTION_NAME,
        embedding_model: str        = EMBEDDING_MODEL,
        ollama_model:    str        = OLLAMA_MODEL,
    ) -> None:
        self._chroma_dir      = Path(chroma_dir)
        self._collection_name = collection_name
        self._embedding_model = embedding_model
        self._ollama_model    = ollama_model

        self._model:      SentenceTransformer | None = None
        self._collection: chromadb.Collection | None = None

        self._initialize()

    # ── Initialisation ────────────────────────────────────────────────────────

    def _initialize(self) -> None:
        logger.info("Initialisation du chatbot RAG…")
        self._load_embedding_model()
        self._connect_chromadb()
        self._check_ollama()
        logger.info("✅ Chatbot prêt")

    def _load_embedding_model(self) -> None:
        logger.info(f"Chargement du modèle '{self._embedding_model}'…")
        self._model = SentenceTransformer(self._embedding_model)

    def _connect_chromadb(self) -> None:
        if not self._chroma_dir.exists():
            raise FileNotFoundError(
                f"Base ChromaDB introuvable : {self._chroma_dir}\n"
                "Lancez d'abord : python 03_embedding.py"
            )
        client = chromadb.PersistentClient(
            path=str(self._chroma_dir),
            settings=Settings(anonymized_telemetry=False),
        )
        try:
            self._collection = client.get_collection(self._collection_name)
            logger.info(
                f"ChromaDB connecté — {self._collection.count()} documents "
                f"dans '{self._collection_name}'"
            )
        except Exception as e:
            raise RuntimeError(
                f"Collection '{self._collection_name}' introuvable.\n"
                f"Lancez d'abord : python 03_embedding.py\nErreur : {e}"
            )

    def _check_ollama(self) -> None:
        """Vérifie qu'Ollama est lancé et que le modèle est disponible."""
        try:
            r = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
            r.raise_for_status()
            models = [m["name"] for m in r.json().get("models", [])]
            if not any(self._ollama_model in m for m in models):
                logger.warning(
                    f"Modèle '{self._ollama_model}' non trouvé dans Ollama.\n"
                    f"Lancez : ollama pull {self._ollama_model}"
                )
            else:
                logger.info(f"Ollama OK — modèle '{self._ollama_model}' disponible")
        except requests.exceptions.ConnectionError:
            logger.warning(
                "Ollama non accessible. Lancez Ollama avant d'utiliser le chatbot.\n"
                "Téléchargement : https://ollama.ai"
            )

    # ── Retrieval ─────────────────────────────────────────────────────────────

    def _encode_query(self, query: str) -> list[float]:
        """Encode la question en vecteur normalisé."""
        vector = self._model.encode(
            query,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return vector.tolist()

    def search(
        self,
        query:     str,
        top_k:     int   = TOP_K,
        min_score: float = MIN_SCORE,
    ) -> list[RetrievalResult]:
        """Recherche les chunks les plus pertinents pour une question."""
        if not query.strip():
            return []

        query_vector = self._encode_query(query)

        raw = self._collection.query(
            query_embeddings=[query_vector],
            n_results=min(top_k, self._collection.count()),
            include=["documents", "metadatas", "distances"],
        )

        results = []
        for cid, doc, meta, dist in zip(
            raw["ids"][0],
            raw["documents"][0],
            raw["metadatas"][0],
            raw["distances"][0],
        ):
            score = round(1.0 - dist / 2.0, 4)
            if score < min_score:
                continue

            code_blocks = []
            try:
                code_blocks = json.loads(meta.get("code_blocks", "[]"))
            except (json.JSONDecodeError, TypeError):
                pass

            results.append(RetrievalResult(
                chunk_id   = cid,
                text       = doc,
                page_title = meta.get("page_title", ""),
                section    = meta.get("section", ""),
                source_url = meta.get("source_url", ""),
                score      = score,
                code_blocks= code_blocks,
                has_code   = bool(meta.get("has_code", False)),
            ))

        logger.info(f"Retrieval : {len(results)} chunks pertinents trouvés (seuil={min_score})")
        return results

    # ── Construction du prompt ────────────────────────────────────────────────

    def _build_prompt(self, question: str, results: list[RetrievalResult]) -> str:
        """
        Construit le prompt augmenté :
        system_prompt + contexte (chunks) + question utilisateur.
        """
        if not results:
            context = "Aucun extrait de documentation pertinent trouvé pour cette question."
        else:
            blocks       = []
            total_chars  = 0
            for i, r in enumerate(results, 1):
                block = f"--- Extrait {i} ---\n{r.to_context_block()}"
                if total_chars + len(block) > MAX_CONTEXT_CHARS:
                    break
                blocks.append(block)
                total_chars += len(block)
            context = "\n\n".join(blocks)

        return (
            f"{SYSTEM_PROMPT}\n\n"
            f"=== DOCUMENTATION FLASK (extraits pertinents) ===\n\n"
            f"{context}\n\n"
            f"=== QUESTION ===\n{question}\n\n"
            f"=== RÉPONSE ==="
        )

    # ── Appel Ollama ──────────────────────────────────────────────────────────

    def _call_ollama(self, prompt: str) -> str:
        """Envoie le prompt à Mistral via l'API Ollama et retourne la réponse."""
        payload = {
            "model":  self._ollama_model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.1,    # faible température = réponses précises et factuelles
                "num_predict": 1024,   # longueur max de la réponse
                "top_p":       0.9,
            },
        }
        try:
            response = requests.post(
                f"{OLLAMA_BASE_URL}/api/generate",
                json=payload,
                timeout=OLLAMA_TIMEOUT,
            )
            response.raise_for_status()
            return response.json().get("response", "").strip()

        except requests.exceptions.ConnectionError:
            return (
                "⚠️ Ollama n'est pas accessible. "
                "Assurez-vous qu'Ollama est lancé (https://ollama.ai)."
            )
        except requests.exceptions.Timeout:
            return "⚠️ Le modèle met trop de temps à répondre. Réessayez."
        except Exception as e:
            logger.error(f"Erreur Ollama : {e}")
            return f"⚠️ Erreur lors de la génération : {e}"

    def _call_ollama_stream(self, prompt: str) -> Iterator[str]:
        """Version streaming : yield les tokens au fur et à mesure (pour Streamlit)."""
        payload = {
            "model":  self._ollama_model,
            "prompt": prompt,
            "stream": True,
            "options": {"temperature": 0.1, "num_predict": 1024},
        }
        try:
            with requests.post(
                f"{OLLAMA_BASE_URL}/api/generate",
                json=payload,
                stream=True,
                timeout=OLLAMA_TIMEOUT,
            ) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if line:
                        data = json.loads(line)
                        token = data.get("response", "")
                        if token:
                            yield token
                        if data.get("done"):
                            break
        except requests.exceptions.ConnectionError:
            yield "⚠️ Ollama n'est pas accessible."
        except Exception as e:
            yield f"⚠️ Erreur : {e}"

    # ── Interface principale ──────────────────────────────────────────────────

    def ask(self, question: str) -> ChatResponse:
        """
        Pipeline RAG complet : question → réponse sourcée.

        Retourne un objet ChatResponse avec la réponse, les sources
        et les métadonnées de la génération.
        """
        logger.info(f"Question : {question[:80]}…")

        # 1. Retrieval
        results = self.search(question)

        # 2. Construction du prompt
        prompt = self._build_prompt(question, results)

        # 3. Génération LLM
        answer = self._call_ollama(prompt)

        # 4. Extraction des sources pour affichage
        sources = [
            {
                "page_title": r.page_title,
                "section":    r.section,
                "source_url": r.source_url,
                "score":      r.score,
            }
            for r in results
        ]

        return ChatResponse(
            question    = question,
            answer      = answer,
            sources     = sources,
            chunks_used = len(results),
            model       = self._ollama_model,
        )

    def ask_stream(self, question: str):
        """
        Version streaming pour Streamlit.
        Yield (token, sources) — sources uniquement dans le dernier yield.
        """
        results = self.search(question)
        prompt  = self._build_prompt(question, results)
        sources = [
            {
                "page_title": r.page_title,
                "section":    r.section,
                "source_url": r.source_url,
                "score":      r.score,
            }
            for r in results
        ]

        for token in self._call_ollama_stream(prompt):
            yield token, None   # token en cours, pas encore de sources

        yield "", sources       # dernier yield : sources disponibles

    def is_relevant_query(self, question: str) -> bool:
        """Retourne True si la question est dans le périmètre Flask."""
        return len(self.search(question, top_k=1, min_score=0.35)) > 0

    @property
    def collection_size(self) -> int:
        return self._collection.count() if self._collection else 0


# ── Test en ligne de commande ─────────────────────────────────────────────────

def _interactive_mode(chatbot: FlaskChatbot) -> None:
    """Mode interactif en terminal."""
    print(f"\n{'═' * 65}")
    print("  🤖 Flask Documentation Chatbot — Mode interactif")
    print(f"  Collection : {chatbot.collection_size} documents indexés")
    print("  Tapez 'quit' pour quitter")
    print(f"{'═' * 65}\n")

    while True:
        try:
            question = input("Vous : ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nAu revoir !")
            break

        if question.lower() in ("quit", "exit", "q"):
            print("Au revoir !")
            break

        if not question:
            continue

        response = chatbot.ask(question)

        print(f"\n{'─' * 65}")
        print(f"Assistant :\n{response.answer}")

        if response.sources:
            print(f"\n📚 Sources utilisées ({response.chunks_used} extraits) :")
            for i, src in enumerate(response.sources, 1):
                print(f"  [{i}] {src['page_title']} — {src['section']} (score: {src['score']:.3f})")
                print(f"       {src['source_url']}")
        print(f"{'─' * 65}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="04_chatbot.py — Chatbot RAG Flask + Mistral")
    parser.add_argument(
        "question", nargs="?",
        help="Question à poser au chatbot"
    )
    parser.add_argument(
        "--interactive", "-i", action="store_true",
        help="Lancer le mode interactif en terminal"
    )
    args = parser.parse_args()

    chatbot = FlaskChatbot()

    if args.interactive or not args.question:
        _interactive_mode(chatbot)
    else:
        response = chatbot.ask(args.question)
        print(f"\nRéponse :\n{response.answer}")
        if response.sources:
            print(f"\nSources :")
            for src in response.sources:
                print(f"  • {src['page_title']} — {src['section']}")
                print(f"    {src['source_url']}")