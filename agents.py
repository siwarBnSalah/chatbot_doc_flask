"""
agents.py — Module 4 : Couche Agentique
Projet : chatbot_doc_flask
Description :
    Implémente les trois agents qui enrichissent le pipeline RAG :
        - RoutingAgent           : détection hors-scope
        - QueryReformulationAgent: amélioration du retrieval
        - CodeValidationAgent    : validation syntaxique Python
        - AgenticRAGPipeline     : orchestrateur complet

    Ce module est la valeur ajoutée du PFE par rapport à un RAG simple.

Usage :
    # Importé par chatbot.py (remplacement direct)
    from agents import AgenticRAGPipeline
    pipeline = AgenticRAGPipeline(retriever)
    result   = pipeline.run("How to create a Flask route?")
"""

from __future__ import annotations

import ast
import json
import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

# ── Configuration ──────────────────────────────────────────────────────────────
CHROMA_DIR       = Path("vector_db")
COLLECTION_NAME  = "flask_documentation"
EMBEDDING_MODEL  = "all-MiniLM-L6-v2"
TOP_K_DEFAULT    = 4
MIN_SCORE        = 0.25
MAX_CONTEXT_CHARS= 3000

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """Tu es un assistant technique expert en Flask, spécialisé dans la documentation officielle Flask 3.0.x.

Règles strictes :
1. Réponds UNIQUEMENT en te basant sur les extraits de documentation fournis.
2. Si l'information est absente → réponds : "Je ne trouve pas cette information dans la documentation Flask 3.0.x."
3. Cite toujours la source : Source : [page] — [section] — [URL]
4. Intègre les extraits de code disponibles en les formatant correctement.
5. Réponds en français si la question est en français, en anglais sinon.
6. Ne génère aucune information absente des extraits fournis.
7. Si du code a été corrigé automatiquement, mentionne-le."""


# ── Dataclass RetrievalResult ─────────────────────────────────────────────────

@dataclass
class RetrievalResult:
    chunk_id:    str
    text:        str
    page_title:  str
    section:     str 
    source_url:  str
    score:       float
    code_blocks: list = field(default_factory=list)
    has_code:    bool = False

    def to_context_block(self) -> str:
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


# ── BaseRetriever ─────────────────────────────────────────────────────────────

class BaseRetriever:
    """Retriever sémantique de base sur ChromaDB."""

    def __init__(self):
        self._model:      SentenceTransformer | None = None
        self._collection: chromadb.Collection | None = None
        self._load()

    def _load(self):
        logger.info(f"Chargement modèle '{EMBEDDING_MODEL}'...")
        self._model = SentenceTransformer(EMBEDDING_MODEL)

        if not CHROMA_DIR.exists():
            raise FileNotFoundError(
                f"Base ChromaDB introuvable : {CHROMA_DIR}\n"
                "Lancez : python 04_embedding.py"
            )
        client = chromadb.PersistentClient(
            path=str(CHROMA_DIR),
            settings=Settings(anonymized_telemetry=False)
        )
        self._collection = client.get_collection(COLLECTION_NAME)
        logger.info(f"ChromaDB connecté — {self._collection.count()} documents")

    def search(self, query: str, top_k: int = TOP_K_DEFAULT,
               min_score: float = MIN_SCORE) -> list[RetrievalResult]:
        if not query.strip():
            return []

        vector = self._model.encode(query, normalize_embeddings=True, convert_to_numpy=True)

        raw = self._collection.query(
            query_embeddings=[vector.tolist()],
            n_results=min(top_k, self._collection.count()),
            include=["documents", "metadatas", "distances"],
        )

        results = []
        for cid, doc, meta, dist in zip(
            raw["ids"][0], raw["documents"][0],
            raw["metadatas"][0], raw["distances"][0]
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
                chunk_id    = cid,
                text        = doc,
                page_title  = meta.get("page_title", ""),
                section     = meta.get("section", ""),
                source_url  = meta.get("source_url", ""),
                score       = score,
                code_blocks = code_blocks,
                has_code    = bool(meta.get("has_code", False)),
            ))
        return sorted(results, key=lambda x: x.score, reverse=True)

    @property
    def collection_size(self) -> int:
        return self._collection.count() if self._collection else 0


# ══════════════════════════════════════════════════════════════════════════════
# AGENT 1 — Routing hors-scope
# ══════════════════════════════════════════════════════════════════════════════

class RoutingAgent:
    """
    Détecte si la question est dans le périmètre Flask.
    Version améliorée : réduction des faux positifs (hors-scope).
    """

    FLASK_POSITIVE_KEYWORDS = {
        "flask", "route", "blueprint", "jinja", "werkzeug", "wsgi",
        "app.run", "app_context", "request", "response", "redirect",
        "url_for", "render_template", "jsonify", "make_response",
        "before_request", "after_request", "endpoint", "view function",
        "context", "session", "error handler", "signal", "cli",
        "flask-login", "flask-sqlalchemy", "flask-wtf", "flask-migrate",
    }

    OUT_OF_SCOPE_KEYWORDS = {
        "django", "fastapi", "tornado", "react", "vue", "angular",
        "pytorch", "tensorflow", "keras", "pandas", "numpy",
    }

    # ✅ NOUVEAU : blacklist forte (rejet immédiat)
    HARD_BLACKLIST = {
        "pib", "capitale", "recette", "cuisine", "carbonara",
        "sport", "film", "musique",
        "chatgpt", "openai", "gpt",
        "tensorflow", "pytorch", "sklearn",
    }

    RESPONSES = {
        "other_framework": (
            "❌ Cette question concerne un autre framework ({framework}), pas Flask."
        ),
        "non_technical": (
            "❌ Cette question ne concerne pas Flask."
        ),
        "low_relevance": (
            "⚠️ Aucune information pertinente trouvée dans la documentation Flask."
        ),
    }

    def __init__(self, retriever: BaseRetriever, threshold: float = 0.45):  # ✅ seuil augmenté
        self.retriever = retriever
        self.threshold = threshold

    def _detect_framework(self, query: str) -> Optional[str]:
        for fw in ["django", "fastapi", "react", "vue", "angular"]:
            if fw in query.lower():
                return fw
        return None

    def _contains_blacklist(self, query: str) -> bool:
        q = query.lower()
        return any(word in q for word in self.HARD_BLACKLIST)

    def _kw_score(self, query: str) -> float:
        words = set(re.findall(r'\b\w+\b', query.lower()))
        pos = len(words & self.FLASK_POSITIVE_KEYWORDS)
        neg = len(words & self.OUT_OF_SCOPE_KEYWORDS)

        # ✅ Pondération améliorée
        score = 0.5 + (pos * 0.15) - (neg * 0.25)
        return max(0.0, min(1.0, score))

    def run(self, query: str) -> dict:

        # ✅ 0. Blacklist forte (ultra important)
        if self._contains_blacklist(query):
            return {
                "in_scope": False,
                "action": "reject",
                "reason": "Mot blacklist détecté",
                "response": self.RESPONSES["non_technical"],
            }

        # 1. Détection framework
        fw = self._detect_framework(query)
        if fw:
            return {
                "in_scope": False,
                "action": "reject",
                "reason": f"Autre framework : {fw}",
                "response": self.RESPONSES["other_framework"].format(framework=fw),
            }

        # 2. Score mots-clés
        kw_score = self._kw_score(query)
        if kw_score < 0.2:
            return {
                "in_scope": False,
                "action": "reject",
                "reason": "Score mots-clés faible",
                "response": self.RESPONSES["non_technical"],
            }

        # 3. Retrieval rapide
        quick = self.retriever.search(query, top_k=1, min_score=0.15)

        if not quick:
            return {
                "in_scope": False,
                "action": "reject",
                "reason": "Aucun résultat",
                "response": self.RESPONSES["low_relevance"],
            }

        score = quick[0].score

        # ✅ Seuil plus strict
        if score < self.threshold:
            return {
                "in_scope": True,
                "action": "warn",
                "reason": f"Score limite ({score:.3f})",
                "response": None,
            }

        return {
            "in_scope": True,
            "action": "proceed",
            "reason": f"OK ({score:.3f})",
            "response": None,
        }


# ══════════════════════════════════════════════════════════════════════════════
# AGENT 2 — Reformulation de requête
# ══════════════════════════════════════════════════════════════════════════════

class QueryReformulationAgent:
    """
    Améliore le retrieval en reformulant la question
    si les résultats initiaux sont de mauvaise qualité.
    """

    SYNONYMS = {
        "route":      ["endpoint", "url rule", "decorator"],
        "blueprint":  ["module flask", "composant"],
        "debug":      ["mode debug", "développement"],
        "template":   ["jinja", "jinja2", "render"],
        "config":     ["configuration", "paramètres", "settings"],
        "session":    ["cookie session", "user session"],
        "error":      ["exception", "gestionnaire erreur", "404", "500"],
        "test":       ["unittest", "pytest", "test client"],
        "request":    ["requête http", "form data"],
        "response":   ["réponse http", "jsonify"],
        "login":      ["authentification", "flask-login"],
        "deployment": ["production", "gunicorn", "wsgi"],
    }

    FRENCH_TO_ENGLISH = {
        "comment": "how to", "créer": "create", "définir": "define",
        "utiliser": "use", "activer": "enable", "configurer": "configure",
        "afficher": "display", "retourner": "return", "faire": "do",
        "ajouter": "add", "supprimer": "delete", "modifier": "modify",
    }

    def __init__(self, retriever: BaseRetriever, min_score: float = 0.35):
        self.retriever = retriever
        self.min_score = min_score

    def _reformulate(self, query: str) -> list[str]:
        variants = [query]
        q = query.lower()

        # Variante 1 : ajouter Flask
        if "flask" not in q:
            variants.append(f"Flask {query}")

        # Variante 2 : traduire en anglais
        en = query
        for fr, eng in self.FRENCH_TO_ENGLISH.items():
            en = en.replace(fr, eng)
        if en != query:
            variants.append(en)
            variants.append(f"Flask {en}")

        # Variante 3 : utiliser synonymes
        for concept, synonyms in self.SYNONYMS.items():
            if concept in q:
                for syn in synonyms[:1]:
                    variants.append(query.replace(concept, syn))

        # Variante 4 : formulation directe
        concepts = [c for c in self.SYNONYMS if c in q]
        if concepts:
            variants.append(f"Flask {' '.join(concepts)} example tutorial")

        return list(dict.fromkeys(variants))[:5]   # dédoublonner, max 5

    def run(self, query: str, initial_results: list,
            top_k: int = 4) -> dict:
        best_score = max((r.score for r in initial_results), default=0.0)

        if initial_results and best_score >= self.min_score:
            return {
                "reformulated":   False,
                "original_query": query,
                "used_query":     query,
                "results":        initial_results,
                "attempts":       1,
                "best_score":     best_score,
            }

        logger.info(f"Reformulation : score initial {best_score:.3f} < {self.min_score}")
        variants    = self._reformulate(query)
        best_results = initial_results
        best_query   = query
        attempts     = 1

        for v in variants[1:]:
            attempts += 1
            res = self.retriever.search(v, top_k=top_k, min_score=0.15)
            if res:
                current = max(r.score for r in res)
                if current > best_score:
                    best_score   = current
                    best_results = res
                    best_query   = v
                    logger.info(f"  Variante '{v}' → score {current:.3f}")
                    if current >= self.min_score:
                        break

        return {
            "reformulated":   best_query != query,
            "original_query": query,
            "used_query":     best_query,
            "results":        best_results,
            "attempts":       attempts,
            "best_score":     best_score,
        }


# ══════════════════════════════════════════════════════════════════════════════
# AGENT 3 — Validation du code Python
# ══════════════════════════════════════════════════════════════════════════════

class CodeValidationAgent:
    """
    Valide syntaxiquement les blocs de code Python extraits
    de la documentation et tente des corrections automatiques simples.
    """

    DEPRECATED = [
        (r"app\.run\(debug=True\)",
         "Préférez 'flask --debug run' (Flask 3.0.x)"),
        (r"from flask\.ext\.",
         "flask.ext est obsolète depuis Flask 1.0"),
        (r"@app\.before_first_request",
         "Supprimé dans Flask 2.3+, utilisez with app.app_context()"),
    ]

    def validate(self, code: str) -> dict:
        code_clean = code.strip()
        errors, warnings, fixed = [], [], None

        if not code_clean:
            return {"valid": True, "errors": [], "warnings": [], "fixed": None}

        # Validation syntaxique
        try:
            ast.parse(code_clean)
        except SyntaxError as e:
            errors.append(f"Ligne {e.lineno} : {e.msg}")
            fixed = self._try_fix(code_clean, e)

        # Patterns dépréciés
        for pattern, msg in self.DEPRECATED:
            if re.search(pattern, code_clean):
                warnings.append(f"Usage déprécié : {msg}")

        return {"valid": not errors, "errors": errors,
                "warnings": warnings, "fixed": fixed}

    def _try_fix(self, code: str, e: SyntaxError) -> Optional[str]:
        # Fix : parenthèses non fermées
        if "EOF" in str(e.msg):
            diff = code.count("(") - code.count(")")
            if diff > 0:
                return code + ")" * diff
        # Fix : deux-points manquants
        lines = code.split("\n")
        if e.lineno and e.lineno <= len(lines):
            line = lines[e.lineno - 1]
            if re.match(r"^(def |class |if |for |while |with )", line.strip()):
                if not line.rstrip().endswith(":"):
                    lines[e.lineno - 1] = line.rstrip() + ":"
                    return "\n".join(lines)
        return None

    def run(self, results: list[RetrievalResult]) -> dict:
        report = {
            "total_blocks":         0,
            "valid_blocks":         0,
            "blocks_with_errors":   [],
            "blocks_with_warnings": [],
            "all_valid":            True,
        }

        for r in results:
            for block in r.code_blocks:
                if block.get("language", "") not in ("python", ""):
                    continue
                v = self.validate(block.get("code", ""))
                report["total_blocks"] += 1

                if v["valid"]:
                    report["valid_blocks"] += 1
                else:
                    report["all_valid"] = False
                    report["blocks_with_errors"].append({
                        "chunk_id": r.chunk_id,
                        "errors":   v["errors"],
                        "fixed":    v["fixed"],
                    })
                    if v["fixed"]:
                        block["code"]      = v["fixed"]
                        block["corrected"] = True

                if v["warnings"]:
                    report["blocks_with_warnings"].append({
                        "chunk_id": r.chunk_id,
                        "warnings": v["warnings"],
                    })

        return report


# ══════════════════════════════════════════════════════════════════════════════
# ORCHESTRATEUR — AgenticRAGPipeline
# ══════════════════════════════════════════════════════════════════════════════

class AgenticRAGPipeline:
    """
    Orchestrateur du pipeline RAG agentique complet.

    Workflow :
      Question → [Routing] → [Retrieval] → [Reformulation] → [Validation] → Prompt
    """

    def __init__(self, top_k: int = TOP_K_DEFAULT,
                 max_context_chars: int = MAX_CONTEXT_CHARS):
        self.top_k             = top_k
        self.max_context_chars = max_context_chars

        # Initialisation des composants
        self.retriever           = BaseRetriever()
        self.routing_agent       = RoutingAgent(self.retriever)
        self.reformulation_agent = QueryReformulationAgent(self.retriever)
        self.validation_agent    = CodeValidationAgent()

        logger.info("✅ AgenticRAGPipeline initialisé")

    def _build_prompt(self, question: str, results: list,
                      metadata: dict) -> str:
        if not results:
            context = "Aucun extrait pertinent trouvé."
        else:
            blocks, chars = [], 0
            for i, r in enumerate(results, 1):
                b = f"--- Extrait {i} ---\n{r.to_context_block()}"
                if chars + len(b) > self.max_context_chars:
                    break
                blocks.append(b)
                chars += len(b)
            context = "\n\n".join(blocks)

        notes = []
        if metadata.get("reformulated"):
            notes.append(
                f"[Note : question reformulée de '{metadata['original_query']}' "
                f"vers '{metadata['used_query']}']"
            )
        if metadata.get("code_warnings"):
            notes.append("[Note : certains blocs de code ont des avertissements]")
        if metadata.get("routing_action") == "warn":
            notes.append("[Note : pertinence Flask incertaine — résultats approximatifs]")

        notes_str = ("\n\n=== NOTES ===\n" + "\n".join(notes)) if notes else ""

        return (
            f"{SYSTEM_PROMPT}\n\n"
            f"=== DOCUMENTATION FLASK 3.0.x ===\n\n"
            f"{context}{notes_str}\n\n"
            f"=== QUESTION ===\n{question}\n\n"
            f"=== RÉPONSE ==="
        )

    def run(self, question: str) -> dict:
        """
        Exécute le pipeline complet.

        Retourne :
            answer    : None (rempli par chatbot.py via Ollama)
            prompt    : prompt augmenté prêt pour le LLM
            sources   : liste des sources utilisées
            in_scope  : bool
            ...métadonnées agentiques
        """
        t0       = time.perf_counter()
        metadata = {}

        logger.info(f"Pipeline agentique : '{question[:60]}...'")

        # ── 1. Routing ────────────────────────────────────────────────────
        routing = self.routing_agent.run(question)
        metadata["routing_action"] = routing["action"]

        if not routing["in_scope"]:
            return {
                "answer":    routing["response"],
                "prompt":    None,
                "sources":   [],
                "chunks_used": 0,
                "in_scope":  False,
                "reformulated": False,
                "original_question": question,
                "used_question":     question,
                "validation_report": None,
                "elapsed_s": round(time.perf_counter() - t0, 2),
                "agent_metadata": metadata,
            }

        # ── 2. Retrieval initial ──────────────────────────────────────────
        initial = self.retriever.search(question, top_k=self.top_k, min_score=0.20)

        # ── 3. Reformulation ──────────────────────────────────────────────
        reform  = self.reformulation_agent.run(question, initial, self.top_k)
        final   = reform["results"]
        metadata.update({
            "reformulated":    reform["reformulated"],
            "original_query":  reform["original_query"],
            "used_query":      reform["used_query"],
            "best_score":      reform["best_score"],
        })

        # ── 4. Validation code ────────────────────────────────────────────
        val_report = self.validation_agent.run(final)
        metadata["code_warnings"] = len(val_report["blocks_with_warnings"]) > 0

        # ── 5. Prompt ─────────────────────────────────────────────────────
        prompt  = self._build_prompt(reform["used_query"], final, metadata)
        sources = [{
            "page_title": r.page_title,
            "section":    r.section,
            "source_url": r.source_url,
            "score":      r.score,
            "has_code":   r.has_code,
        } for r in final]

        logger.info(
            f"Pipeline terminé en {time.perf_counter()-t0:.2f}s | "
            f"chunks={len(final)} | réformulé={reform['reformulated']}"
        )

        return {
            "answer":    None,   # rempli par chatbot.py
            "prompt":    prompt,
            "sources":   sources,
            "chunks_used":       len(final),
            "in_scope":          True,
            "reformulated":      reform["reformulated"],
            "original_question": question,
            "used_question":     reform["used_query"],
            "validation_report": val_report,
            "elapsed_s":         round(time.perf_counter() - t0, 2),
            "agent_metadata":    metadata,
        }

    @property
    def collection_size(self) -> int:
        return self.retriever.collection_size


# ── Test rapide ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    pipeline = AgenticRAGPipeline()
    print(f"\n✅ Pipeline initialisé — {pipeline.collection_size} documents")

    tests = [
        "How to create a Flask route with a dynamic parameter?",
        "Comment activer le mode debug Flask ?",
        "Comment créer un modèle PyTorch ?",  # hors-scope
        "Recette de la tarte aux pommes",      # hors-scope
    ]

    for q in tests:
        r = pipeline.run(q)
        status = "✅ IN-SCOPE" if r["in_scope"] else "❌ HORS-SCOPE"
        reform = "🔄 REFORMULÉ" if r["reformulated"] else ""
        print(f"\n{status} {reform}")
        print(f"  Q : {q[:60]}")
        print(f"  Chunks : {r['chunks_used']} | Temps : {r['elapsed_s']}s")
        if not r["in_scope"]:
            print(f"  Réponse : {r['answer'][:100]}...")