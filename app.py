"""
app.py — Interface utilisateur Streamlit
Projet : chatbot_doc_flask
Description :
    Interface web du chatbot RAG Flask.
    Fonctionnalités :
        - Saisie de question en langage naturel
        - Réponse en streaming (token par token)
        - Affichage des sources documentaires utilisées
        - Historique de la conversation
        - Indicateur de pertinence des résultats

Prérequis :
    - python 03_embedding.py    (base vectorielle construite)
    - ollama pull mistral       (modèle Mistral disponible)
    - ollama serve              (serveur Ollama lancé)

Usage :
    streamlit run app.py
"""

import streamlit as st
from pathlib import Path

# ── Configuration de la page ──────────────────────────────────────────────────
st.set_page_config(
    page_title="Flask Doc Chatbot",
    page_icon="🐍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS personnalisé ──────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Zone de chat principale */
    .main .block-container { padding-top: 1rem; }

    /* Bulles de message */
    .user-message {
        background-color: #1F3864;
        color: white;
        padding: 12px 16px;
        border-radius: 12px 12px 4px 12px;
        margin: 8px 0;
        max-width: 80%;
        margin-left: auto;
        font-size: 0.95rem;
    }
    .bot-message {
        background-color: #F0F4FF;
        color: #1a1a2e;
        padding: 12px 16px;
        border-radius: 12px 12px 12px 4px;
        margin: 8px 0;
        max-width: 85%;
        border-left: 4px solid #2E75B6;
        font-size: 0.95rem;
    }

    /* Carte source */
    .source-card {
        background: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 8px;
        padding: 8px 12px;
        margin: 4px 0;
        font-size: 0.82rem;
    }
    .source-score {
        color: #2E75B6;
        font-weight: bold;
    }

    /* Score badge */
    .badge-high   { color: #155724; background: #d4edda; padding: 2px 8px; border-radius: 12px; }
    .badge-medium { color: #856404; background: #fff3cd; padding: 2px 8px; border-radius: 12px; }
    .badge-low    { color: #721c24; background: #f8d7da; padding: 2px 8px; border-radius: 12px; }
</style>
""", unsafe_allow_html=True)


# ── Initialisation du chatbot (mise en cache) ─────────────────────────────────

@st.cache_resource(show_spinner="Chargement du chatbot…")
def load_chatbot():
    """
    Charge le chatbot une seule fois et le met en cache dans la session.
    @st.cache_resource évite de recharger le modèle à chaque interaction.
    """
    try:
        from chatbot import FlaskChatbot
        return FlaskChatbot(), None
    except FileNotFoundError as e:
        return None, str(e)
    except Exception as e:
        return None, str(e)


# ── Initialisation de l'état de la session ────────────────────────────────────

if "messages" not in st.session_state:
    st.session_state.messages = []   # historique [{role, content, sources}]

if "total_questions" not in st.session_state:
    st.session_state.total_questions = 0


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.image("https://flask.palletsprojects.com/en/3.0.x/_images/flask-horizontal.png",
             use_column_width=True)
    st.markdown("---")

    st.markdown("### ⚙️ Configuration")

    top_k = st.slider(
        "Nombre d'extraits (top-k)",
        min_value=1, max_value=8, value=4,
        help="Nombre de passages de documentation récupérés par question"
    )
    min_score = st.slider(
        "Seuil de pertinence",
        min_value=0.1, max_value=0.8, value=0.25, step=0.05,
        help="Score minimum de similarité pour qu'un extrait soit retenu"
    )
    show_sources = st.toggle("Afficher les sources", value=True)
    show_scores  = st.toggle("Afficher les scores", value=False)

    st.markdown("---")
    st.markdown("### 📊 Statistiques de session")

    chatbot_obj, error_msg = load_chatbot()
    if chatbot_obj:
        st.metric("Documents indexés", chatbot_obj.collection_size)
    st.metric("Questions posées", st.session_state.total_questions)

    st.markdown("---")

    if st.button("🗑️ Effacer l'historique", use_container_width=True):
        st.session_state.messages = []
        st.session_state.total_questions = 0
        st.rerun()

    st.markdown("---")
    st.markdown("""
    **📖 À propos**
    Chatbot RAG basé sur la documentation officielle
    [Flask 3.0.x](https://flask.palletsprojects.com/en/3.0.x/).

    **Stack technique :**
    - Embedding : `all-MiniLM-L6-v2`
    - Vector DB : `ChromaDB`
    - LLM : `Mistral` via Ollama
    """)


# ── Interface principale ──────────────────────────────────────────────────────

st.title("🐍 Flask Documentation Chatbot")
st.caption("Posez vos questions sur Flask — les réponses sont basées exclusivement sur la documentation officielle.")

# Vérification de l'état du chatbot
if error_msg:
    st.error(f"❌ Impossible de charger le chatbot :\n\n{error_msg}")
    st.info("💡 Assurez-vous d'avoir lancé `python 03_embedding.py` pour construire la base vectorielle.")
    st.stop()

# ── Affichage de l'historique ─────────────────────────────────────────────────

chat_container = st.container()
with chat_container:
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(
                f'<div class="user-message">👤 {msg["content"]}</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div class="bot-message">🤖 {msg["content"]}</div>',
                unsafe_allow_html=True,
            )
            # Affichage des sources
            if show_sources and msg.get("sources"):
                with st.expander(
                    f"📚 {len(msg['sources'])} source(s) utilisée(s)", expanded=False
                ):
                    for src in msg["sources"]:
                        score = src.get("score", 0)
                        badge_class = (
                            "badge-high"   if score >= 0.6 else
                            "badge-medium" if score >= 0.4 else
                            "badge-low"
                        )
                        score_html = (
                            f'<span class="{badge_class}">{score:.3f}</span>'
                            if show_scores else ""
                        )
                        st.markdown(
                            f'<div class="source-card">'
                            f'<strong>{src["page_title"]}</strong> — {src["section"]}<br>'
                            f'<a href="{src["source_url"]}" target="_blank">🔗 {src["source_url"]}</a>'
                            f' {score_html}'
                            f'</div>',
                            unsafe_allow_html=True,
                        )


# ── Questions suggérées ───────────────────────────────────────────────────────

if not st.session_state.messages:
    st.markdown("### 💡 Questions suggérées")
    suggested = [
        "How to create a basic Flask application?",
        "Comment définir des routes dans Flask ?",
        "How to use Flask blueprints?",
        "Comment gérer les erreurs HTTP dans Flask ?",
        "How to configure Flask debug mode?",
        "Comment utiliser les templates Jinja2 avec Flask ?",
    ]
    cols = st.columns(2)
    for i, q in enumerate(suggested):
        with cols[i % 2]:
            if st.button(q, key=f"suggested_{i}", use_container_width=True):
                st.session_state.pending_question = q
                st.rerun()


# ── Zone de saisie ────────────────────────────────────────────────────────────

with st.form("question_form", clear_on_submit=True):
    col1, col2 = st.columns([5, 1])
    with col1:
        user_input = st.text_input(
            "Votre question",
            placeholder="Ex : How to use Flask blueprints? / Comment définir des routes ?",
            label_visibility="collapsed",
        )
    with col2:
        submitted = st.form_submit_button("Envoyer 🚀", use_container_width=True)

# Récupérer une question suggérée si elle existe
if hasattr(st.session_state, "pending_question"):
    user_input = st.session_state.pending_question
    submitted  = True
    del st.session_state.pending_question


# ── Traitement de la question ─────────────────────────────────────────────────

if submitted and user_input.strip():
    question = user_input.strip()

    # Ajout de la question à l'historique
    st.session_state.messages.append({"role": "user", "content": question})
    st.session_state.total_questions += 1

    # Affichage immédiat de la question
    st.markdown(
        f'<div class="user-message">👤 {question}</div>',
        unsafe_allow_html=True,
    )

    # Génération de la réponse avec streaming
    with st.spinner("Recherche dans la documentation Flask…"):

        # Retrieval
        results = chatbot_obj.search(question, top_k=top_k, min_score=min_score)

        if not results:
            answer = (
                "Je ne trouve pas d'information pertinente sur ce sujet dans la "
                "documentation Flask. Cette question est peut-être hors du périmètre "
                "de la documentation Flask 3.0.x."
            )
            sources = []
        else:
            # Construction du prompt
            from chatbot import SYSTEM_PROMPT, MAX_CONTEXT_CHARS

            blocks      = []
            total_chars = 0
            for i, r in enumerate(results, 1):
                block = f"--- Extrait {i} ---\n{r.to_context_block()}"
                if total_chars + len(block) > MAX_CONTEXT_CHARS:
                    break
                blocks.append(block)
                total_chars += len(block)

            context = "\n\n".join(blocks)
            prompt  = (
                f"{SYSTEM_PROMPT}\n\n"
                f"=== DOCUMENTATION FLASK ===\n\n{context}\n\n"
                f"=== QUESTION ===\n{question}\n\n=== RÉPONSE ==="
            )

            # Streaming Mistral
            answer_placeholder = st.empty()
            full_answer        = ""

            for token, _ in chatbot_obj.ask_stream(question):
                if token:
                    full_answer += token
                    answer_placeholder.markdown(
                        f'<div class="bot-message">🤖 {full_answer}▌</div>',
                        unsafe_allow_html=True,
                    )

            answer_placeholder.markdown(
                f'<div class="bot-message">🤖 {full_answer}</div>',
                unsafe_allow_html=True,
            )
            answer  = full_answer
            sources = [
                {
                    "page_title": r.page_title,
                    "section":    r.section,
                    "source_url": r.source_url,
                    "score":      r.score,
                }
                for r in results
            ]

    # Affichage des sources
    if show_sources and sources:
        with st.expander(f"📚 {len(sources)} source(s) utilisée(s)", expanded=True):
            for src in sources:
                score = src.get("score", 0)
                badge_class = (
                    "badge-high"   if score >= 0.6 else
                    "badge-medium" if score >= 0.4 else
                    "badge-low"
                )
                score_html = (
                    f'<span class="{badge_class}">{score:.3f}</span>'
                    if show_scores else ""
                )
                st.markdown(
                    f'<div class="source-card">'
                    f'<strong>{src["page_title"]}</strong> — {src["section"]}<br>'
                    f'<a href="{src["source_url"]}" target="_blank">🔗 {src["source_url"]}</a>'
                    f' {score_html}'
                    f'</div>',
                    unsafe_allow_html=True,
                )

    # Sauvegarde dans l'historique
    st.session_state.messages.append({
        "role":    "assistant",
        "content": answer,
        "sources": sources,
    })