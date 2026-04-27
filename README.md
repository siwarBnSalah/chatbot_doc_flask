📘 Chatbot RAG Intelligent (Groq + Agents IA)

🔥 Version: version-groq
🤖 RAG + Agents + Groq LLM + Evaluation

📌 Table des matières
🧠 Aperçu
🏗️ Architecture
⚙️ Installation
▶️ Exécution
📂 Structure du projet
🤖 Fonctionnalités
🧪 Pipeline du système
🛠️ Technologies
📊 Evaluation
🚀 Améliorations futures
👨‍🎓 Auteur
🧠 Aperçu

Ce projet est un chatbot intelligent basé sur RAG (Retrieval-Augmented Generation) amélioré par une architecture agentique et un modèle LLM rapide via Groq.

👉 Objectif :

répondre aux questions à partir de documents
améliorer la qualité des réponses avec des agents IA
optimiser la recherche et la génération
🏗️ Architecture
Utilisateur
    ↓
RoutingAgent (filtrage hors-scope)
    ↓
QueryReformulationAgent (optimisation requête)
    ↓
Retrieval (ChromaDB vector store)
    ↓
Groq LLM (génération réponse)
    ↓
CodeValidationAgent (si code présent)
    ↓
Réponse finale
⚙️ Installation
git clone https://github.com/siwarBnSalah/chatbot_doc_flask.git
cd chatbot_doc_flask
git checkout version-groq
🔹 Environnement virtuel
python -m venv venv
venv\Scripts\activate   # Windows
🔹 Dépendances
pip install -r requirements.txt
▶️ Exécution
python app.py

ou :

streamlit run app.py
📂 Structure du projet
chatbot_doc_flask/
│
├── app.py                 # interface principale
├── chatbot.py            # pipeline RAG
├── agents.py             # agents IA
├── test_groq.py          # test LLM Groq
├── evaluation_reelle.py  # évaluation système
├── requirements.txt
│
├── vector_db/            # base vectorielle ChromaDB
├── evaluation_output/    # résultats evaluation
├── data_propre/          # documents source
🤖 Fonctionnalités
✔ Core System
RAG (retrieval augmented generation)
recherche sémantique (embeddings)
génération via Groq LLM
✔ Agents IA
RoutingAgent (hors-scope detection)
QueryReformulationAgent (amélioration requête)
CodeValidationAgent (validation code)
✔ Intelligence
amélioration automatique des questions
réponses contextualisées
filtrage intelligent des requêtes
🧪 Pipeline du système
Question utilisateur
        ↓
Analyse (RoutingAgent)
        ↓
Reformulation (Query Agent)
        ↓
Recherche vectorielle (ChromaDB)
        ↓
LLM Groq
        ↓
Validation
        ↓
Réponse finale
🛠️ Technologies
Python 🐍
Groq API 🤖
ChromaDB 📦
Sentence Transformers
Flask / Streamlit
Pandas / Matplotlib
📊 Evaluation

Le système inclut :

accuracy des réponses
analyse qualitative
dashboards visuels
logs d’évaluation