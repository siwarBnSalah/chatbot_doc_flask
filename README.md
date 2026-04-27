📘 Chatbot RAG Intelligent avec Groq + Agents (version-groq)
🚀 Présentation du projet

Ce projet est un chatbot intelligent basé sur RAG (Retrieval-Augmented Generation) enrichi par une architecture agentique et un modèle LLM rapide via Groq API.

Il permet d’interroger une base de documents et d’obtenir des réponses contextuelles, fiables et optimisées grâce à plusieurs agents spécialisés.

🧠 Architecture du système

Le système repose sur une architecture avancée en plusieurs couches :

1️⃣ Couche RAG (Retrieval-Augmented Generation)
Embeddings des documents
Stockage vectoriel (ChromaDB)
Recherche sémantique
Récupération des chunks pertinents
2️⃣ Couche Agentique (intelligence du système)

Le système utilise plusieurs agents :

🔄 RoutingAgent
détecte si la question est hors-scope
évite les réponses inutiles
✍️ QueryReformulationAgent
améliore la question utilisateur
optimise le retrieval
🧪 CodeValidationAgent
vérifie la validité des extraits de code
sécurise les réponses techniques
3️⃣ Couche LLM (Groq)
Modèle rapide et puissant via Groq
Génération des réponses finales
Optimisation du prompt système
4️⃣ Couche Evaluation
Evaluation réelle des réponses
Métriques de performance
Analyse des résultats
Visualisation (dashboards)
📂 Structure du projet
chatbot_doc_flask/
│
├── app.py                  # API / interface principale
├── chatbot.py             # logique RAG + pipeline
├── agents.py              # agents IA (routing, reformulation, validation)
├── test_groq.py           # tests du modèle Groq
├── evaluation_reelle.py   # évaluation du système
├── requirements.txt       # dépendances
│
├── vector_db/             # base vectorielle ChromaDB
├── evaluation_output/     # résultats d’évaluation
└── data_propre/           # documents source
⚙️ Installation
1. Cloner le projet
git clone https://github.com/siwarBnSalah/chatbot_doc_flask.git
cd chatbot_doc_flask
git checkout version-groq
2. Créer un environnement virtuel
python -m venv venv
venv\Scripts\activate   # Windows
3. Installer les dépendances
pip install -r requirements.txt
4. Configurer les variables d’environnement

Créer un fichier .env :

GROQ_API_KEY=your_api_key_here
▶️ Lancer le projet
python app.py

ou si interface Streamlit :

streamlit run app.py
🧪 Fonctionnalités principales

✔ Chatbot intelligent basé sur RAG
✔ Agents IA (routing, reformulation, validation)
✔ LLM rapide via Groq
✔ Recherche sémantique vectorielle
✔ Evaluation automatique des réponses
✔ Architecture modulaire et extensible

📊 Exemple de workflow
L’utilisateur pose une question
RoutingAgent vérifie la pertinence
QueryReformulationAgent optimise la requête
Retrieval dans ChromaDB
Groq génère la réponse finale
Evaluation optionnelle du résultat
🧠 Technologies utilisées
Python 🐍
Flask / Streamlit
LangChain (ou pipeline custom)
Groq API 🤖
ChromaDB (vector database)
Sentence Transformers
Pandas / Matplotlib (evaluation)