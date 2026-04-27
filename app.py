"""
app.py — Interface Streamlit professionnelle
Projet : chatbot_doc_flask — Support technique intelligent RAG + Groq GPU

Optimisations v2 :
    - Séparation run_pipeline() / stream_from_result() → zéro double appel RAG
    - Résultat pipeline mis en session_state pour réaffichage sans recalcul
    - Gestion propre des erreurs Groq (rate limit, auth)
"""

import streamlit as st
import time
from datetime import datetime

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Flask Doc Assistant",
    page_icon="🐍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Fira+Code:wght@400;500&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    #MainMenu, footer, header { visibility: hidden; }
    .stDeployButton { display: none; }

    .stApp {
        background: linear-gradient(135deg, #0f0f1a 0%, #1a1a2e 50%, #16213e 100%);
    }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d1117 0%, #161b27 100%);
        border-right: 1px solid rgba(99,102,241,0.2);
    }
    [data-testid="stSidebar"] * { color: #e2e8f0 !important; }

    .sidebar-logo {
        text-align: center; padding: 20px 10px;
        border-bottom: 1px solid rgba(99,102,241,0.3); margin-bottom: 20px;
    }
    .sidebar-logo h1 {
        font-size: 1.3rem !important; font-weight: 700 !important;
        background: linear-gradient(135deg, #818cf8, #a78bfa);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        margin: 8px 0 4px 0 !important;
    }
    .sidebar-logo p { font-size: 0.7rem !important; color: #64748b !important; margin: 0 !important; }
    .metric-card {
        background: rgba(99,102,241,0.08); border: 1px solid rgba(99,102,241,0.2);
        border-radius: 10px; padding: 10px 14px; margin: 6px 0;
        display: flex; justify-content: space-between; align-items: center;
    }
    .metric-label { font-size: 0.72rem; color: #94a3b8; }
    .metric-value { font-size: 0.95rem; font-weight: 600; color: #818cf8; }
    .sidebar-section {
        font-size: 0.62rem; font-weight: 600; letter-spacing: 0.1em;
        color: #4b5563 !important; text-transform: uppercase; padding: 10px 0 4px 0;
    }
    .main-header {
        background: linear-gradient(135deg, rgba(99,102,241,0.15), rgba(167,139,250,0.1));
        border: 1px solid rgba(99,102,241,0.3); border-radius: 16px;
        padding: 22px 28px; margin-bottom: 20px;
        display: flex; align-items: center; gap: 18px;
    }
    .main-header-icon { font-size: 2.8rem; }
    .main-header-title {
        font-size: 1.7rem !important; font-weight: 700 !important;
        background: linear-gradient(135deg, #818cf8, #a78bfa, #c084fc);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        margin: 0 !important;
    }
    .main-header-subtitle { font-size: 0.82rem; color: #64748b; margin: 3px 0 0 0; }
    .status-badge {
        background: rgba(16,185,129,0.1); border: 1px solid rgba(16,185,129,0.3);
        border-radius: 20px; padding: 5px 12px; font-size: 0.72rem;
        color: #10b981; font-weight: 500; white-space: nowrap;
    }
    .groq-badge {
        background: rgba(245,158,11,0.1); border: 1px solid rgba(245,158,11,0.3);
        border-radius: 20px; padding: 5px 12px; font-size: 0.72rem;
        color: #f59e0b; font-weight: 500; white-space: nowrap; margin-left: 8px;
    }
    .msg-user {
        display: flex; justify-content: flex-end; margin: 10px 0;
        animation: fadeInRight 0.3s ease;
    }
    .msg-user-bubble {
        background: linear-gradient(135deg, #4f46e5, #6366f1);
        color: white; padding: 11px 16px;
        border-radius: 18px 18px 4px 18px;
        max-width: 70%; font-size: 0.88rem; line-height: 1.5;
        box-shadow: 0 4px 15px rgba(99,102,241,0.3);
    }
    .msg-assistant {
        display: flex; justify-content: flex-start;
        margin: 10px 0; gap: 9px; animation: fadeInLeft 0.3s ease;
    }
    .msg-avatar {
        width: 34px; height: 34px;
        background: linear-gradient(135deg, #818cf8, #a78bfa);
        border-radius: 50%; display: flex; align-items: center;
        justify-content: center; font-size: 1rem; flex-shrink: 0; margin-top: 4px;
    }
    .msg-assistant-bubble {
        background: rgba(30,30,50,0.85); border: 1px solid rgba(99,102,241,0.2);
        color: #e2e8f0; padding: 13px 17px;
        border-radius: 4px 18px 18px 18px;
        max-width: 76%; font-size: 0.88rem; line-height: 1.6;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    .agent-badges { display: flex; gap: 5px; margin-top: 9px; flex-wrap: wrap; }
    .badge { font-size: 0.63rem; padding: 3px 9px; border-radius: 12px; font-weight: 500; }
    .badge-routing   { background:rgba(16,185,129,0.15); color:#10b981; border:1px solid rgba(16,185,129,0.3); }
    .badge-reform    { background:rgba(245,158,11,0.15);  color:#f59e0b; border:1px solid rgba(245,158,11,0.3); }
    .badge-valid     { background:rgba(99,102,241,0.15);  color:#818cf8; border:1px solid rgba(99,102,241,0.3); }
    .badge-outscope  { background:rgba(239,68,68,0.15);   color:#ef4444; border:1px solid rgba(239,68,68,0.3); }
    .badge-time      { background:rgba(100,116,139,0.15); color:#94a3b8; border:1px solid rgba(100,116,139,0.3); }
    .badge-groq      { background:rgba(245,158,11,0.15);  color:#f59e0b; border:1px solid rgba(245,158,11,0.3); }
    .sources-panel { margin-top: 10px; border-top: 1px solid rgba(99,102,241,0.15); padding-top: 10px; }
    .sources-title { font-size: 0.68rem; color: #64748b; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 7px; font-weight: 600; }
    .source-item {
        background: rgba(99,102,241,0.06); border: 1px solid rgba(99,102,241,0.12);
        border-radius: 8px; padding: 7px 11px; margin: 4px 0; font-size: 0.76rem;
    }
    .source-page { color: #a78bfa; font-weight: 600; }
    .source-sec  { color: #94a3b8; }
    .source-score{ color: #10b981; font-weight: 500; float: right; }
    .source-url  { color: #4b5563; font-size: 0.66rem; margin-top: 2px; }
    .source-url a{ color: #818cf8; text-decoration: none; }
    .thinking { display: flex; gap: 5px; padding: 11px 17px; background: rgba(30,30,50,0.85); border: 1px solid rgba(99,102,241,0.2); border-radius: 4px 18px 18px 18px; width: fit-content; }
    .thinking span { width: 7px; height: 7px; background: #818cf8; border-radius: 50%; animation: bounce 1.2s infinite; }
    .thinking span:nth-child(2){ animation-delay:.2s; }
    .thinking span:nth-child(3){ animation-delay:.4s; }
    @keyframes fadeInRight { from{opacity:0;transform:translateX(20px)} to{opacity:1;transform:translateX(0)} }
    @keyframes fadeInLeft  { from{opacity:0;transform:translateX(-20px)} to{opacity:1;transform:translateX(0)} }
    @keyframes bounce { 0%,60%,100%{transform:translateY(0)} 30%{transform:translateY(-7px)} }
    .stButton button {
        background: linear-gradient(135deg, #4f46e5, #6366f1) !important;
        border: none !important; color: white !important;
        border-radius: 10px !important; font-weight: 500 !important;
        transition: all 0.2s !important; font-size: 0.82rem !important;
    }
    .stButton button:hover { transform: translateY(-1px); box-shadow: 0 4px 15px rgba(99,102,241,0.4) !important; }
    div[data-testid="stChatInput"] {
        background: rgba(20,20,35,0.9) !important;
        border: 1px solid rgba(99,102,241,0.3) !important;
        border-radius: 14px !important;
    }
    div[data-testid="stChatInput"] textarea { color: #e2e8f0 !important; }
    pre, code {
        font-family: 'Fira Code', monospace !important;
        background: rgba(0,0,0,0.4) !important;
        border: 1px solid rgba(99,102,241,0.2) !important;
        border-radius: 8px !important; color: #a78bfa !important;
    }
    ::-webkit-scrollbar { width: 4px; }
    ::-webkit-scrollbar-thumb { background: rgba(99,102,241,0.3); border-radius: 4px; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# CHARGEMENT CHATBOT (mis en cache — chargé une seule fois)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_chatbot():
    from chatbot import FlaskChatbot
    return FlaskChatbot()

with st.spinner("⚡ Initialisation du pipeline RAG..."):
    chatbot = load_chatbot()


# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────────────────────────
for key, default in [
    ("messages",       []),
    ("total_q",        0),
    ("total_reformed", 0),
    ("total_rejected", 0),
    ("pending_q",      None),
    ("total_time",     0.0),
]:
    if key not in st.session_state:
        st.session_state[key] = default


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="sidebar-logo">
        <div style="font-size:2.2rem">🐍</div>
        <h1>Flask Doc Assistant</h1>
        <p>RAG · Groq LPU · Agentic AI · PFE 2025</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<p class="sidebar-section">📊 Session actuelle</p>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    c1.metric("Questions", st.session_state.total_q)
    c2.metric("Reformulées", st.session_state.total_reformed)

    avg_t = (st.session_state.total_time / st.session_state.total_q
             if st.session_state.total_q > 0 else 0)

    st.markdown(f"""
    <div class="metric-card">
        <span class="metric-label">📄 Documents indexés</span>
        <span class="metric-value">{chatbot.collection_size}</span>
    </div>
    <div class="metric-card">
        <span class="metric-label">⏱ Temps moyen réponse</span>
        <span class="metric-value">{avg_t:.1f}s</span>
    </div>
    <div class="metric-card">
        <span class="metric-label">❌ Questions rejetées</span>
        <span class="metric-value">{st.session_state.total_rejected}</span>
    </div>
    <div class="metric-card">
        <span class="metric-label">🧠 Embedding</span>
        <span class="metric-value">MiniLM-L6-v2</span>
    </div>
    <div class="metric-card">
        <span class="metric-label">🤖 LLM</span>
        <span class="metric-value">Llama 3.3 70B</span>
    </div>
    <div class="metric-card">
        <span class="metric-label">⚡ Accélération</span>
        <span class="metric-value">Groq LPU</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<p class="sidebar-section">⚙️ Paramètres RAG</p>', unsafe_allow_html=True)
    top_k     = st.slider("Extraits récupérés (top-k)", 1, 6, 3)
    min_score = st.slider("Seuil de pertinence", 0.10, 0.80, 0.25, step=0.05)
    show_src  = st.toggle("Afficher les sources",  value=True)
    show_agent= st.toggle("Afficher les agents",   value=True)

    st.markdown('<p class="sidebar-section">🤖 Agents actifs</p>', unsafe_allow_html=True)
    st.markdown("""
    <div style="display:flex;flex-direction:column;gap:5px">
        <div class="badge badge-routing" style="padding:6px 12px;border-radius:8px">🛡️ Routing hors-scope</div>
        <div class="badge badge-reform"  style="padding:6px 12px;border-radius:8px">🔄 Reformulation requête</div>
        <div class="badge badge-valid"   style="padding:6px 12px;border-radius:8px">✅ Validation code Python</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    if st.button("🗑️ Effacer la conversation", use_container_width=True):
        for k in ["messages", "total_q", "total_reformed", "total_rejected", "total_time"]:
            st.session_state[k] = [] if k == "messages" else 0
        st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <div class="main-header-icon">🐍</div>
    <div>
        <h1 class="main-header-title">Flask Documentation Assistant</h1>
        <p class="main-header-subtitle">RAG + Agentic AI · Documentation officielle Flask 3.0.x</p>
    </div>
    <div style="margin-left:auto;display:flex;flex-wrap:wrap;gap:6px;justify-content:flex-end">
        <div class="status-badge">⚡ Système opérationnel</div>
        <div class="groq-badge">🚀 Llama 3.3 70B via Groq</div>
    </div>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def score_bar(score: float) -> str:
    filled = int(score * 10)
    bar    = "█" * filled + "░" * (10 - filled)
    color  = "#10b981" if score >= 0.6 else "#f59e0b" if score >= 0.4 else "#ef4444"
    return f'<span style="font-family:monospace;color:{color}">{bar}</span> {score:.2f}'


def build_source_html(sources: list) -> str:
    if not sources:
        return ""
    items = "".join(f"""
    <div class="source-item">
        <span class="source-score">{score_bar(s['score'])}</span>
        <span class="source-page">{s['page_title']}</span>
        <span class="source-sec"> — {s['section']}</span>
        <div class="source-url"><a href="{s['source_url']}" target="_blank">🔗 {s['source_url']}</a></div>
    </div>""" for s in sources)
    return f"""
    <div class="sources-panel">
        <div class="sources-title">📚 {len(sources)} source(s)</div>
        {items}
    </div>"""


def render_message(msg: dict):
    if msg["role"] == "user":
        st.markdown(f"""
        <div class="msg-user">
            <div class="msg-user-bubble">{msg['content']}</div>
        </div>""", unsafe_allow_html=True)
        return

    badges_html = ""
    if show_agent and msg.get("agent_info"):
        info   = msg["agent_info"]
        badges = []
        if not info.get("in_scope"):
            badges.append('<span class="badge badge-outscope">❌ Hors-scope</span>')
        else:
            badges.append('<span class="badge badge-routing">✅ In-scope</span>')
        if info.get("reformulated"):
            badges.append('<span class="badge badge-reform">🔄 Reformulé</span>')
        if info.get("code_validated"):
            badges.append('<span class="badge badge-valid">✅ Code validé</span>')
        badges.append('<span class="badge badge-groq">⚡ Groq LPU</span>')
        if info.get("elapsed_s"):
            badges.append(f'<span class="badge badge-time">⏱ {info["elapsed_s"]}s</span>')
        badges_html = f'<div class="agent-badges">{"".join(badges)}</div>'

    src_html = build_source_html(msg.get("sources", [])) if show_src else ""

    st.markdown(f"""
    <div class="msg-assistant">
        <div class="msg-avatar">🐍</div>
        <div class="msg-assistant-bubble">
            {msg['content']}
            {badges_html}
            {src_html}
        </div>
    </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# SUGGESTIONS (conversation vide)
# ─────────────────────────────────────────────────────────────────────────────
if not st.session_state.messages:
    st.markdown("""
    <div style="text-align:center;padding:24px 20px 8px">
        <p style="color:#64748b;font-size:0.9rem">
            Posez vos questions sur <b style="color:#818cf8">Flask 3.0.x</b>
        </p>
    </div>""", unsafe_allow_html=True)

    suggestions = [
        ("🛣️", "Comment créer une route Flask avec paramètre dynamique ?"),
        ("🏗️", "How to use Flask blueprints to organize an application?"),
        ("🐛", "Comment activer le mode debug dans Flask 3.0.x ?"),
        ("⚙️", "How to configure Flask with environment variables?"),
        ("🔐", "Comment gérer les sessions utilisateur dans Flask ?"),
        ("📦", "How to deploy a Flask application with Gunicorn?"),
    ]
    cols = st.columns(2)
    for i, (icon, text) in enumerate(suggestions):
        with cols[i % 2]:
            if st.button(f"{icon}  {text}", key=f"sug_{i}", use_container_width=True):
                st.session_state.pending_q = text
                st.rerun()
    st.markdown("<br>", unsafe_allow_html=True)

# Historique
for msg in st.session_state.messages:
    render_message(msg)


# ─────────────────────────────────────────────────────────────────────────────
# INPUT
# ─────────────────────────────────────────────────────────────────────────────
question = st.chat_input("💬 Posez votre question sur Flask 3.0.x...")

if st.session_state.pending_q:
    question = st.session_state.pending_q
    st.session_state.pending_q = None


# ─────────────────────────────────────────────────────────────────────────────
# TRAITEMENT — PIPELINE SÉPARÉ DU STREAMING (pas de double appel)
# ─────────────────────────────────────────────────────────────────────────────
if question and question.strip():
    q = question.strip()

    # 1. Afficher le message utilisateur
    st.session_state.messages.append({"role": "user", "content": q})
    st.session_state.total_q += 1
    st.markdown(f"""
    <div class="msg-user">
        <div class="msg-user-bubble">{q}</div>
    </div>""", unsafe_allow_html=True)

    # 2. Animation de réflexion pendant le pipeline RAG (local, rapide)
    thinking = st.empty()
    thinking.markdown("""
    <div class="msg-assistant">
        <div class="msg-avatar">🐍</div>
        <div class="thinking"><span></span><span></span><span></span></div>
    </div>""", unsafe_allow_html=True)

    t0 = time.perf_counter()

    # ── ÉTAPE 1 : Pipeline RAG (local, ~0.3s) ────────────────────────────────
    # Un seul appel pipeline — résultat réutilisé pour le streaming
    pipeline_result = chatbot.run_pipeline(q)

    if pipeline_result.get("reformulated"):
        st.session_state.total_reformed += 1

    thinking.empty()

    # ── ÉTAPE 2A : Hors-scope → affichage direct ──────────────────────────────
    if not pipeline_result["in_scope"]:
        st.session_state.total_rejected += 1
        elapsed = round(time.perf_counter() - t0, 2)

        agent_info = {"in_scope": False, "elapsed_s": elapsed}
        badges = '<span class="badge badge-outscope">❌ Hors-scope</span>'
        badges += f'<span class="badge badge-time">⏱ {elapsed}s</span>'

        st.markdown(f"""
        <div class="msg-assistant">
            <div class="msg-avatar">🐍</div>
            <div class="msg-assistant-bubble">
                <span style="color:#ef4444">⚠️</span> {pipeline_result['answer']}
                <div class="agent-badges">{badges}</div>
            </div>
        </div>""", unsafe_allow_html=True)

        st.session_state.messages.append({
            "role":       "assistant",
            "content":    f"⚠️ {pipeline_result['answer']}",
            "sources":    [],
            "agent_info": agent_info,
        })

    # ── ÉTAPE 2B : In-scope → streaming Groq ─────────────────────────────────
    else:
        full_answer = ""
        stream_ph   = st.empty()

        # Streaming depuis le résultat pipeline pré-calculé (pas de double appel)
        for token in chatbot.stream_from_result(pipeline_result):
            full_answer += token
            stream_ph.markdown(f"""
            <div class="msg-assistant">
                <div class="msg-avatar">🐍</div>
                <div class="msg-assistant-bubble">{full_answer}▌</div>
            </div>""", unsafe_allow_html=True)

        elapsed = round(time.perf_counter() - t0, 2)
        st.session_state.total_time += elapsed

        agent_info = {
            "in_scope":       True,
            "reformulated":   pipeline_result.get("reformulated", False),
            "code_validated": pipeline_result.get("validation_report") is not None,
            "elapsed_s":      elapsed,
        }

        # Badges finaux
        badges = '<span class="badge badge-routing">✅ In-scope</span>'
        if pipeline_result.get("reformulated"):
            badges += '<span class="badge badge-reform">🔄 Reformulé</span>'
        badges += '<span class="badge badge-groq">⚡ Groq LPU</span>'
        badges += f'<span class="badge badge-time">⏱ {elapsed}s</span>'

        sources  = pipeline_result.get("sources", [])
        src_html = build_source_html(sources) if show_src else ""

        stream_ph.markdown(f"""
        <div class="msg-assistant">
            <div class="msg-avatar">🐍</div>
            <div class="msg-assistant-bubble">
                {full_answer}
                <div class="agent-badges">{badges}</div>
                {src_html}
            </div>
        </div>""", unsafe_allow_html=True)

        st.session_state.messages.append({
            "role":       "assistant",
            "content":    full_answer,
            "sources":    sources,
            "agent_info": agent_info,
        })

    st.rerun()