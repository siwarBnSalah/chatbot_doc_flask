# 🚀 Chatbot RAG Intelligent

---

## 🌟 Aperçu

Ce projet est un **chatbot intelligent basé sur RAG (Retrieval-Augmented Generation)**, amélioré par une **architecture agentique** et accéléré grâce au modèle **Groq LLM**.

Il transforme un système classique de question-réponse sur documents en un **système multi-agents intelligent capable de raisonner, filtrer et optimiser les réponses**.

---

## 🧠 Idée principale

👉 Au lieu d’un chatbot classique :
Utilisateur → LLM → Réponse ❌


Nous avons construit :


Utilisateur → Agents → Recherche (RAG) → Groq LLM → Validation → Réponse intelligente ✅


---

## 🏗️ Architecture du système


Utilisateur
↓
RoutingAgent (filtrage des questions hors-scope)
↓
QueryReformulationAgent (optimisation de la requête)
↓
Recherche vectorielle (ChromaDB)
↓
Groq LLM (génération rapide)
↓
CodeValidationAgent (validation si code)
↓
Réponse finale


---

## ⚙️ Installation

```bash id="g8w3ab"
git clone https://github.com/siwarBnSalah/chatbot_doc_flask.git
cd chatbot_doc_flask
git checkout version-groq


🐍 Environnement virtuel
python -m venv venv
venv\Scripts\activate   # Windows

📦 Installation des dépendances
pip install -r requirements.txt


▶️ Exécution du projet
python app.py

ou

streamlit run app.py


📂 Structure du projet
chatbot_doc_flask/
│
├── app.py                  # Interface principale
├── chatbot.py              # Pipeline RAG
├── agents.py               # Logique des agents IA
├── test_groq.py            # Test du modèle Groq
├── evaluation_reelle.py    # Système d’évaluation
├── requirements.txt
│
├── vector_db/              # Base vectorielle (ChromaDB)
├── evaluation_output/      # Résultats d’évaluation
├── data_propre/            # Documents source



🤖 Fonctionnalités


🔹 Système RAG
Recherche sémantique avec embeddings
Base vectorielle (ChromaDB)
Génération contextuelle des réponses


🔹 Couche Agents IA
🧭 RoutingAgent → détecte les questions hors-sujet
✍️ QueryReformulationAgent → améliore les requêtes utilisateur
🧪 CodeValidationAgent → vérifie les réponses contenant du code


🔹 Moteur LLM
⚡ API Groq (très rapide)
Génération de réponses intelligentes et contextuelles


🧪 Pipeline du système
Question utilisateur
        ↓
RoutingAgent
        ↓
Reformulation de la question
        ↓
Recherche vectorielle (ChromaDB)
        ↓
Génération avec Groq LLM
        ↓
Validation
        ↓
Réponse finale


🛠️ Technologies utilisées
Python 🐍
Groq API 🤖
ChromaDB 📦
Sentence Transformers
Flask / Streamlit
Pandas / Matplotlib