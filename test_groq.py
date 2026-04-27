"""
test_groq.py — Test de connexion à l'API Groq
Projet : chatbot_doc_flask

Exécuter EN PREMIER pour vérifier que la clé API et le modèle fonctionnent.

Usage :
    python test_groq.py
"""

import os
import time
from dotenv import load_dotenv

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL   = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

print("=" * 60)
print("  TEST CONNEXION GROQ API")
print("=" * 60)

if not GROQ_API_KEY:
    print("\n❌ ERREUR : Clé GROQ_API_KEY non trouvée !")
    print("\n   Étapes :")
    print("   1. https://console.groq.com → créer un compte gratuit")
    print("   2. Menu → API Keys → Create API Key")
    print("   3. Ajouter dans .env : GROQ_API_KEY=gsk_ta_cle")
    print("   4. Relancer : python test_groq.py")
    exit(1)

print(f"\n✅ Clé API : {GROQ_API_KEY[:8]}...{GROQ_API_KEY[-4:]}")

try:
    from groq import Groq
    print("✅ Bibliothèque groq importée")
except ImportError:
    print("\n❌ groq non installé → pip install groq")
    exit(1)

client = Groq(api_key=GROQ_API_KEY)
print(f"✅ Client Groq initialisé | modèle cible : {GROQ_MODEL}")

# ── Test 1 : Appel simple ─────────────────────────────────────────────────────
print("\n" + "-" * 60)
print("  TEST 1 : Appel simple (non-streaming)")
print("-" * 60)

t0 = time.perf_counter()
try:
    response = client.chat.completions.create(
        model    = GROQ_MODEL,
        messages = [
            {"role": "system", "content": "Tu es un expert Flask. Réponds brièvement."},
            {"role": "user",   "content": "What is Flask? Answer in 2 sentences."},
        ],
        temperature = 0.1,
        max_tokens  = 100,
    )
    elapsed = time.perf_counter() - t0
    answer  = response.choices[0].message.content.strip()
    print(f"✅ {elapsed:.2f}s | {response.usage.total_tokens} tokens | modèle : {response.model}")
    print(f"   → {answer[:180]}")
except Exception as e:
    print(f"❌ ERREUR : {e}")
    print(f"\n   Si 'model not found', essayez un autre modèle dans .env :")
    print(f"   GROQ_MODEL=llama3-8b-8192")
    exit(1)

# ── Test 2 : Streaming ────────────────────────────────────────────────────────
print("\n" + "-" * 60)
print("  TEST 2 : Streaming token par token")
print("-" * 60)

t0     = time.perf_counter()
tokens = 0
print("   Réponse : ", end="", flush=True)

try:
    stream = client.chat.completions.create(
        model    = GROQ_MODEL,
        messages = [
            {"role": "system", "content": "Expert Flask. Code examples only."},
            {"role": "user",   "content": "Show a minimal Flask route example."},
        ],
        temperature = 0.1,
        max_tokens  = 120,
        stream      = True,
    )
    for chunk in stream:
        token = chunk.choices[0].delta.content or ""
        if token:
            print(token, end="", flush=True)
            tokens += 1

    elapsed = time.perf_counter() - t0
    print(f"\n\n✅ {tokens} tokens en {elapsed:.2f}s (~{int(tokens/elapsed)} tok/s)")
except Exception as e:
    print(f"\n❌ ERREUR streaming : {e}")
    exit(1)

# ── Test 3 : Prompt RAG simulé ────────────────────────────────────────────────
print("\n" + "-" * 60)
print("  TEST 3 : Simulation prompt RAG Flask")
print("-" * 60)

rag_system = """Tu es un assistant Flask 3.0.x. Réponds UNIQUEMENT depuis les extraits fournis."""

rag_user = """=== DOCUMENTATION FLASK ===
[SOURCE] Quickstart — Routing
[CONTENU] Use the route() decorator to bind a function to a URL.
@app.route('/')
def index():
    return 'Index Page'

=== QUESTION ===
Comment créer une route Flask ?

=== RÉPONSE ==="""

t0 = time.perf_counter()
try:
    response = client.chat.completions.create(
        model    = GROQ_MODEL,
        messages = [
            {"role": "system", "content": rag_system},
            {"role": "user",   "content": rag_user},
        ],
        temperature = 0.1,
        max_tokens  = 200,
    )
    elapsed = time.perf_counter() - t0
    print(f"✅ {elapsed:.2f}s")
    print(f"   → {response.choices[0].message.content.strip()[:250]}")
except Exception as e:
    print(f"❌ {e}")

# ── Résumé ────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  ✅ GROQ API OPÉRATIONNELLE")
print(f"  Modèle : {GROQ_MODEL}")
print(f"  → Lancez : streamlit run app.py")
print("=" * 60)