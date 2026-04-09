"""
chatbot.py — Moteur RAG Agentique : utilise 05_agents.py
Projet : chatbot_doc_flask
Description :
    Version mise à jour du chatbot qui utilise la couche agentique
    complète (Module 4) au lieu du pipeline RAG simple.

    Changements vs version précédente :
        - Remplace BaseRetriever par AgenticRAGPipeline
        - Intègre les 3 agents (routing, reformulation, validation)
        - Méthodes ask() et ask_stream() enrichies avec métadonnées agentiques

Usage :
    python chatbot.py "How to create a Flask route?"
    python chatbot.py --interactive
"""

from __future__ import annotations

import argparse
import json
import logging
from typing import Iterator

import requests

from agents import AgenticRAGPipeline   # ← Module 4

# ── Configuration Ollama ───────────────────────────────────────────────────────
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL    = "mistral"
OLLAMA_TIMEOUT  = 300   # 5 minutes

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


class FlaskChatbot:
    """
    Chatbot RAG agentique sur la documentation Flask.
    Utilise AgenticRAGPipeline (Module 4) pour les 3 agents.
    """

    def __init__(self, ollama_model: str = OLLAMA_MODEL):
        self._model    = ollama_model
        self._pipeline = AgenticRAGPipeline(top_k=3, max_context_chars=2000)
        self._check_ollama()
        logger.info("✅ FlaskChatbot (version agentique) prêt")

    def _check_ollama(self):
        try:
            r = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
            r.raise_for_status()
            models = [m["name"] for m in r.json().get("models", [])]
            if any(self._model in m for m in models):
                logger.info(f"Ollama OK — modèle '{self._model}' disponible")
            else:
                logger.warning(f"Modèle '{self._model}' non trouvé → ollama pull {self._model}")
        except requests.exceptions.ConnectionError:
            logger.warning("Ollama non accessible — lancez : ollama serve")

    def _call_ollama(self, prompt: str) -> str:
        payload = {
            "model":  self._model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.1,
                "num_predict": 512,
                "num_ctx":     2048,
                "num_thread":  4,
            },
        }
        try:
            r = requests.post(
                f"{OLLAMA_BASE_URL}/api/generate",
                json=payload, timeout=OLLAMA_TIMEOUT
            )
            r.raise_for_status()
            return r.json().get("response", "").strip()
        except requests.exceptions.ConnectionError:
            return "⚠️ Ollama non accessible. Lancez : ollama serve"
        except requests.exceptions.Timeout:
            return "⚠️ Timeout — le modèle prend trop de temps. Réessayez."
        except Exception as e:
            return f"⚠️ Erreur Ollama : {e}"

    def _call_ollama_stream(self, prompt: str) -> Iterator[str]:
        payload = {
            "model":  self._model,
            "prompt": prompt,
            "stream": True,
            "options": {"temperature": 0.1, "num_predict": 512,
                        "num_ctx": 2048, "num_thread": 4},
        }
        try:
            with requests.post(
                f"{OLLAMA_BASE_URL}/api/generate",
                json=payload, stream=True, timeout=OLLAMA_TIMEOUT
            ) as r:
                r.raise_for_status()
                for line in r.iter_lines():
                    if line:
                        data  = json.loads(line)
                        token = data.get("response", "")
                        if token:
                            yield token
                        if data.get("done"):
                            break
        except requests.exceptions.ConnectionError:
            yield "⚠️ Ollama non accessible."
        except Exception as e:
            yield f"⚠️ Erreur : {e}"

    def ask(self, question: str) -> dict:
        """Pipeline RAG agentique complet → réponse sourcée."""
        # Étapes agentiques (routing + reformulation + validation)
        result = self._pipeline.run(question)

        # Si hors-scope → réponse directe sans LLM
        if not result["in_scope"]:
            return {
                "question":    question,
                "answer":      result["answer"],
                "sources":     [],
                "chunks_used": 0,
                "in_scope":    False,
                "reformulated":False,
                "agent_info":  result["agent_metadata"],
            }

        # Génération LLM avec le prompt augmenté
        answer = self._call_ollama(result["prompt"])

        return {
            "question":     question,
            "answer":       answer,
            "sources":      result["sources"],
            "chunks_used":  result["chunks_used"],
            "in_scope":     True,
            "reformulated": result["reformulated"],
            "used_question":result["used_question"],
            "validation":   result["validation_report"],
            "agent_info":   result["agent_metadata"],
        }

    def ask_stream(self, question: str):
        """Version streaming pour Streamlit."""
        result = self._pipeline.run(question)

        if not result["in_scope"]:
            yield result["answer"], result["sources"]
            return

        sources = result["sources"]
        for token in self._call_ollama_stream(result["prompt"]):
            yield token, None
        yield "", sources

    def search(self, question: str, top_k: int = 3,
               min_score: float = 0.25) -> list:
        """Recherche directe (pour la sidebar Streamlit)."""
        return self._pipeline.retriever.search(question, top_k, min_score)

    @property
    def collection_size(self) -> int:
        return self._pipeline.collection_size


# ── Mode interactif ────────────────────────────────────────────────────────────

def _interactive(chatbot: FlaskChatbot):
    print(f"\n{'═'*65}")
    print("  🤖 Flask Documentation Chatbot — Mode Agentique")
    print(f"  {chatbot.collection_size} documents indexés")
    print("  Tapez 'quit' pour quitter")
    print(f"{'═'*65}\n")

    while True:
        try:
            q = input("Vous : ").strip()
        except (KeyboardInterrupt, EOFError):
            break
        if q.lower() in ("quit", "exit", "q"):
            break
        if not q:
            continue

        result = chatbot.ask(q)

        print(f"\n{'─'*65}")
        if result["reformulated"]:
            print(f"🔄 Question reformulée : {result.get('used_question', '')}")
        print(f"Assistant :\n{result['answer']}")
        if result["sources"]:
            print(f"\n📚 Sources ({result['chunks_used']}) :")
            for s in result["sources"]:
                print(f"  [{s['score']:.3f}] {s['page_title']} — {s['section']}")
        print(f"{'─'*65}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("question", nargs="?", help="Question Flask")
    parser.add_argument("--interactive", "-i", action="store_true")
    args = parser.parse_args()

    bot = FlaskChatbot()

    if args.interactive or not args.question:
        _interactive(bot)
    else:
        r = bot.ask(args.question)
        print(f"\nRéponse :\n{r['answer']}")
        if r["sources"]:
            print("\nSources :")
            for s in r["sources"]:
                print(f"  • {s['page_title']} — {s['section']}")