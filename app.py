import streamlit as st

st.set_page_config(
    page_title="Flask RAG Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personnalisé
st.markdown("""
<style>
    .source-card {
        background: #f0f7ff;
        border-left: 4px solid #2E75B6;
        padding: 8px 12px;
        margin: 4px 0;
        border-radius: 4px;
    }
    .score-badge {
        background: #2E75B6;
        color: white;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 12px;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_chatbot():
    from chatbot import FlaskChatbot
    return FlaskChatbot()

chatbot = load_chatbot()

# Session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# ── SIDEBAR ──────────────────────────────────────
with st.sidebar:
    st.image("https://flask.palletsprojects.com/en/3.0.x/_static/flask-horizontal.png", 
             width=180)
    st.markdown("---")
    
    st.header("⚙️ Paramètres")
    top_k     = st.slider("Chunks récupérés (Top-K)", 1, 8, 4)
    min_score = st.slider("Score minimal", 0.1, 0.8, 0.25, step=0.05)
    
    st.markdown("---")
    st.header("📊 Statistiques")
    col1, col2 = st.columns(2)
    col1.metric("Documents", chatbot.collection_size)
    col2.metric("Messages", len(st.session_state.messages))
    
    st.markdown("---")
    st.header("💡 Exemples de questions")
    examples = [
        "Comment créer une route Flask ?",
        "How to use blueprints?",
        "Comment activer le mode debug ?",
        "Comment gérer les erreurs 404 ?",
        "How to use Flask sessions?",
    ]
    for ex in examples:
        if st.button(ex, use_container_width=True, key=ex):
            st.session_state.pending_question = ex
    
    st.markdown("---")
    if st.button("🗑️ Effacer la conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ── TITRE ──────────────────────────────────────
st.title("🤖 Flask RAG Assistant")
st.caption("Posez vos questions sur la documentation officielle Flask 3.0.x")

# ── HISTORIQUE ──────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander(f"📚 {len(msg['sources'])} source(s) utilisée(s)"):
                for src in msg["sources"]:
                    score_pct = int(src['score'] * 100)
                    st.markdown(
                        f'<div class="source-card">'
                        f'<span class="score-badge">{score_pct}%</span> '
                        f'<b>{src["page_title"]}</b> — {src["section"]}<br>'
                        f'<small><a href="{src["source_url"]}">'
                        f'{src["source_url"]}</a></small>'
                        f'</div>',
                        unsafe_allow_html=True
                    )

# ── INPUT ──────────────────────────────────────
question = st.chat_input("Posez votre question sur Flask...")

# Gérer les exemples cliquables
if "pending_question" in st.session_state:
    question = st.session_state.pop("pending_question")

if question:
    # Message utilisateur
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    # Réponse assistant
    with st.chat_message("assistant"):
        placeholder  = st.empty()
        full_response = ""
        final_sources = []

        with st.spinner("🔍 Recherche dans la documentation Flask..."):
            try:
                for token, sources in chatbot.ask_stream(question):
                    if sources is not None:
                        final_sources = sources
                    full_response += token
                    placeholder.markdown(full_response + "▌")
            except Exception as e:
                full_response = f"⚠️ Une erreur est survenue : {e}"
        
        placeholder.markdown(full_response)

        # Afficher les sources
        if final_sources:
            with st.expander(f"📚 {len(final_sources)} source(s) utilisée(s)"):
                for src in final_sources:
                    score_pct = int(src['score'] * 100)
                    st.markdown(
                        f'<div class="source-card">'
                        f'<span class="score-badge">{score_pct}%</span> '
                        f'<b>{src["page_title"]}</b> — {src["section"]}<br>'
                        f'<small><a href="{src["source_url"]}">'
                        f'{src["source_url"]}</a></small>'
                        f'</div>',
                        unsafe_allow_html=True
                    )
        elif "❌" in full_response or "⚠️" in full_response:
            st.warning("Question hors périmètre Flask — aucune source disponible.")

    # Sauvegarder avec les sources
    st.session_state.messages.append({
        "role":    "assistant",
        "content": full_response,
        "sources": final_sources
    })