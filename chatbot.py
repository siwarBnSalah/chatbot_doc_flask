"""
chatbot.py — Moteur RAG Agentique avec Groq API (GPU Cloud)
Projet : chatbot_doc_flask

Optimisations v2 :
    - Modèle Groq mis à jour : llama-3.3-70b-versatile (mixtral supprimé)
    - Suppression du double appel pipeline RAG (résultat mis en cache par session)
    - Prompt RAG restructuré avec séparation system/user (meilleure qualité)
    - Gestion d'erreurs robuste (RateLimitError, AuthenticationError, timeout)
    - Méthode ask_stream() accepte un résultat pré-calculé pour éviter le double appel

Usage :
    python chatbot.py "How to create a Flask route?"
    python chatbot.py --interactive
"""

from __future__ import annotations

import argparse
import logging
import os
import time
from typing import Iterator

from dotenv import load_dotenv
from groq import Groq, RateLimitError, AuthenticationError

from agents import AgenticRAGPipeline

# ── Chargement des variables d'environnement ──────────────────────────────────
load_dotenv()

# ── Configuration ──────────────────────────────────────────────────────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Modèles disponibles sur Groq (avril 2025) — par ordre de préférence
# llama-3.3-70b-versatile : meilleur équilibre qualité/vitesse pour RAG
# llama3-8b-8192           : fallback rapide si rate limit
GROQ_MODEL_PRIMARY  = "llama-3.3-70b-versatile"
GROQ_MODEL_FALLBACK = "llama3-8b-8192"
GROQ_MODEL          = os.getenv("GROQ_MODEL", GROQ_MODEL_PRIMARY)

# Paramètres de génération
TEMPERATURE  = 0.1    # faible = réponses précises et factuelles
MAX_TOKENS   = 600    # légèrement augmenté pour les réponses avec code
TOP_K        = 3      # chunks récupérés par question
MAX_CONTEXT  = 2500   # taille max du contexte injecté (tokens ≈ chars/4)

# ── Prompt système ────────────────────────────────────────────────────────────
# Séparé du prompt utilisateur → meilleure instruction following avec Llama 3
SYSTEM_PROMPT = """Tu es un assistant technique spécialisé dans Flask 3.0.x.

Règles absolues :
1. Réponds UNIQUEMENT en te basant sur les extraits de documentation fournis.
2. Si l'information est absente des extraits → réponds exactement : "Je ne trouve pas cette information dans la documentation Flask 3.0.x fournie."
3. Cite la source à la fin : **Source :** [Titre] — [Section] — [URL]
4. Formate les blocs de code avec des triples backticks et le langage (```python).
5. Réponds en français si la question est en français, en anglais sinon.
6. Sois concis et précis — pas de remplissage inutile.
7. Si du code a été corrigé automatiquement, mentionne-le clairement."""

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


class FlaskChatbot:
    """
    Chatbot RAG agentique avec génération LLM via Groq API.

    Workflow :
        Question
            ↓ (local, ~0.3s)
        AgenticRAGPipeline
            ├─ RoutingAgent       → hors-scope ?
            ├─ Retrieval ChromaDB → top-k chunks
            ├─ ReformulationAgent → amélioration si score faible
            └─ ValidationAgent    → vérification code Python
            ↓
        Prompt augmenté (system + user séparés)
            ↓ (~2-4s)
        Groq API LPU
            ↓
        Réponse sourcée → Streamlit
    """

    def __init__(self):
        if not GROQ_API_KEY:
            raise ValueError(
                "Clé GROQ_API_KEY manquante !\n"
                "1. Créez un compte sur https://console.groq.com\n"
                "2. Générez une clé API gratuite\n"
                "3. Ajoutez dans .env : GROQ_API_KEY=gsk_...\n"
                "4. Vérifiez avec : python test_groq.py"
            )

        self._client   = Groq(api_key=GROQ_API_KEY)
        self._model    = GROQ_MODEL
        logger.info(f"Client Groq initialisé — modèle : {self._model}")

        self._pipeline = AgenticRAGPipeline(
            top_k             = TOP_K,
            max_context_chars = MAX_CONTEXT,
        )
        logger.info("✅ FlaskChatbot (Groq GPU) prêt")

    # ── Construction du prompt user (séparé du system) ────────────────────────

    def _build_user_message(self, pipeline_result: dict) -> str:
        """
        Construit le message utilisateur à partir du résultat pipeline.
        Le system prompt est envoyé séparément → meilleure instruction following.
        """
        context = pipeline_result.get("prompt", "")

        # Extraire uniquement la partie DOCUMENTATION + QUESTION du prompt
        # (le system prompt est déjà géré via le paramètre 'system')
        if "=== DOCUMENTATION FLASK" in context:
            # Conserver uniquement la documentation et la question
            start = context.find("=== DOCUMENTATION FLASK")
            context = context[start:]

        return context

    # ── Appel LLM non-streaming ───────────────────────────────────────────────

    def _call_groq(self, pipeline_result: dict) -> str:
        """
        Envoie le prompt à Groq et retourne la réponse complète.
        Utilise la séparation system/user pour de meilleures réponses.
        """
        user_msg = self._build_user_message(pipeline_result)
        t0 = time.perf_counter()

        try:
            response = self._client.chat.completions.create(
                model    = self._model,
                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": user_msg},
                ],
                temperature = TEMPERATURE,
                max_tokens  = MAX_TOKENS,
            )
            elapsed = time.perf_counter() - t0
            answer  = response.choices[0].message.content.strip()
            logger.info(
                f"Groq OK — {elapsed:.2f}s | "
                f"{response.usage.completion_tokens} tokens générés | "
                f"modèle : {response.model}"
            )
            return answer

        except RateLimitError:
            logger.warning(f"Rate limit sur {self._model}, bascule sur {GROQ_MODEL_FALLBACK}")
            return self._call_groq_fallback(user_msg)

        except AuthenticationError:
            return "⚠️ Clé API Groq invalide. Vérifiez votre fichier .env."

        except Exception as e:
            logger.error(f"Erreur Groq : {e}")
            return f"⚠️ Erreur lors de la génération : {type(e).__name__} — {e}"

    def _call_groq_fallback(self, user_msg: str) -> str:
        """Appel sur le modèle fallback en cas de rate limit."""
        try:
            response = self._client.chat.completions.create(
                model    = GROQ_MODEL_FALLBACK,
                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": user_msg},
                ],
                temperature = TEMPERATURE,
                max_tokens  = MAX_TOKENS,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"⚠️ Erreur fallback : {e}"

    # ── Appel LLM streaming ───────────────────────────────────────────────────

    def _call_groq_stream(self, pipeline_result_or_prompt) -> Iterator[str]:
        """
        Streaming token par token pour Streamlit.

        Accepte soit :
            - un dict pipeline_result (chemin normal depuis app.py)
            - une str (prompt brut, pour compatibilité ascendante)
        """
        # Compatibilité : accepter prompt brut ou résultat pipeline
        if isinstance(pipeline_result_or_prompt, dict):
            user_msg = self._build_user_message(pipeline_result_or_prompt)
        else:
            user_msg = pipeline_result_or_prompt  # str brute

        try:
            stream = self._client.chat.completions.create(
                model    = self._model,
                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": user_msg},
                ],
                temperature = TEMPERATURE,
                max_tokens  = MAX_TOKENS,
                stream      = True,
            )
            for chunk in stream:
                token = chunk.choices[0].delta.content or ""
                if token:
                    yield token

        except RateLimitError:
            logger.warning("Rate limit — bascule fallback en streaming")
            yield from self._stream_fallback(user_msg)

        except AuthenticationError:
            yield "⚠️ Clé API Groq invalide. Vérifiez votre fichier .env."

        except Exception as e:
            logger.error(f"Erreur streaming Groq : {e}")
            yield f"⚠️ Erreur : {type(e).__name__} — {e}"

    def _stream_fallback(self, user_msg: str) -> Iterator[str]:
        """Streaming sur le modèle fallback."""
        try:
            stream = self._client.chat.completions.create(
                model    = GROQ_MODEL_FALLBACK,
                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": user_msg},
                ],
                temperature = TEMPERATURE,
                max_tokens  = MAX_TOKENS,
                stream      = True,
            )
            for chunk in stream:
                token = chunk.choices[0].delta.content or ""
                if token:
                    yield token
        except Exception as e:
            yield f"⚠️ Fallback échoué : {e}"

    # ── Interface principale ──────────────────────────────────────────────────

    def ask(self, question: str) -> dict:
        """
        Pipeline RAG complet + génération Groq → réponse sourcée.
        Pour les tests en ligne de commande.
        """
        t0     = time.perf_counter()
        result = self._pipeline.run(question)

        if not result["in_scope"]:
            return {
                "question":     question,
                "answer":       result["answer"],
                "sources":      [],
                "chunks_used":  0,
                "in_scope":     False,
                "reformulated": False,
                "elapsed_s":    round(time.perf_counter() - t0, 2),
            }

        answer = self._call_groq(result)

        return {
            "question":      question,
            "answer":        answer,
            "sources":       result["sources"],
            "chunks_used":   result["chunks_used"],
            "in_scope":      True,
            "reformulated":  result.get("reformulated", False),
            "used_question": result.get("used_question", question),
            "validation":    result.get("validation_report"),
            "elapsed_s":     round(time.perf_counter() - t0, 2),
        }

    # ── Interface streaming pour Streamlit ────────────────────────────────────

    def run_pipeline(self, question: str) -> dict:
        """
        Exécute uniquement le pipeline RAG (sans LLM).
        Appelé par app.py AVANT le streaming pour séparer les deux étapes
        et éviter le double appel.
        """
        return self._pipeline.run(question)

    def stream_from_result(self, pipeline_result: dict) -> Iterator[str]:
        """
        Génère la réponse en streaming depuis un résultat pipeline pré-calculé.
        Appelé par app.py APRÈS run_pipeline() pour éviter le double appel.
        """
        if not pipeline_result["in_scope"]:
            yield pipeline_result["answer"]
            return
        yield from self._call_groq_stream(pipeline_result)

    # ── Compatibilité ascendante (ancienne interface app.py) ──────────────────

    def ask_stream(self, question: str):
        """
        Version streaming complète (pipeline + LLM).
        Yield: (token, None) pendant la génération, ("", sources) à la fin.

        NOTE : Pour éviter le double appel depuis app.py, préférez
               run_pipeline() + stream_from_result() séparément.
        """
        result = self._pipeline.run(question)

        if not result["in_scope"]:
            yield result["answer"], result["sources"]
            return

        sources = result["sources"]
        for token in self._call_groq_stream(result):
            yield token, None
        yield "", sources

    def search(self, question: str, top_k: int = 3,
               min_score: float = 0.25) -> list:
        return self._pipeline.retriever.search(question, top_k, min_score)

    @property
    def collection_size(self) -> int:
        return self._pipeline.collection_size


# ── Mode terminal ──────────────────────────────────────────────────────────────

def _interactive(chatbot: FlaskChatbot):
    print(f"\n{'═' * 65}")
    print(f"  🐍 Flask Doc Chatbot — Groq {chatbot._model}")
    print(f"  {chatbot.collection_size} documents | Tapez 'quit' pour quitter")
    print(f"{'═' * 65}\n")

    while True:
        try:
            q = input("Vous : ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nAu revoir !")
            break
        if q.lower() in ("quit", "exit", "q"):
            break
        if not q:
            continue

        result = chatbot.ask(q)
        print(f"\n{'─' * 65}")
        if result.get("reformulated"):
            print(f"🔄 Reformulé : {result.get('used_question', '')}")
        print(f"Assistant :\n{result['answer']}")
        if result.get("sources"):
            print(f"\n📚 Sources :")
            for s in result["sources"]:
                print(f"  [{s['score']:.3f}] {s['page_title']} — {s['section']}")
        print(f"⏱ {result['elapsed_s']}s | modèle : {chatbot._model}")
        print(f"{'─' * 65}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flask RAG Chatbot — Groq GPU")
    parser.add_argument("question", nargs="?", help="Question Flask")
    parser.add_argument("--interactive", "-i", action="store_true")
    args = parser.parse_args()

    bot = FlaskChatbot()

    if args.interactive or not args.question:
        _interactive(bot)
    else:
        r = bot.ask(args.question)
        print(f"\n{'─' * 65}")
        if not r["in_scope"]:
            print(f"❌ Hors-scope : {r['answer']}")
        else:
            print(f"Réponse :\n{r['answer']}")
            if r.get("sources"):
                print("\nSources :")
                for s in r["sources"]:
                    print(f"  • {s['page_title']} — {s['section']}")
        print(f"⏱ {r['elapsed_s']}s")
        print(f"{'─' * 65}")