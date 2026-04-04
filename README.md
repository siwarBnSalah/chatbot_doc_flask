#  Chatbot Documentation Flask — RAG Pipeline

**Projet PFE** — Siwar BenSalah  
**Système de chatbot intelligent** basé sur la documentation officielle Flask 3.0.x,  
utilisant une architecture **RAG (Retrieval-Augmented Generation)** + **LLM local (Ollama)**.

---

## 🎯 Description

Ce projet permet aux développeurs de poser des questions en langage naturel sur Flask
et d'obtenir des réponses précises, sourcées et vérifiables — basées **exclusivement**
sur la documentation officielle Flask 3.0.x.

**Stack technique :**
- 🔍 **Embedding** : `all-MiniLM-L6-v2` (sentence-transformers, 384 dimensions)
- 🗄️ **Vector Store** : `ChromaDB` (local, persistant)
- 🤖 **LLM** : `mistral` via Ollama (local, gratuit)
- 🌐 **Interface** : Streamlit

---

## 🏗️ Architecture complète

```
Documentation Flask (officielle)
         ↓
01_scraping.py      ← Collecte 65 pages HTML
         ↓
02_nettoyage.py     ← Nettoyage HTML + extraction sections H2/H3
         ↓
03_chunking.py     ←  Découpe en chunks pour l'embedding
         ↓
04_embedding.py     ← Embeddings (384 dims) + stockage ChromaDB
         ↓
chatbot.py          ← Pipeline RAG : Retrieval + Prompt + LLM
         ↓
app.py              ← Interface Streamlit
```

---

## 📁 Structure du projet

```
chatbot_doc_flask/
│
├── 01_scraping.py          # Module 1 — Scrape 65 pages Flask
├── 02_nettoyage.py         # Module 1 — Nettoyage HTML + 388 chunks
├── 03_chunking.py         # Module 1 — + 388 chunks 
├── 03_embedding.py         # Module 2 — Embeddings + base ChromaDB
├── chatbot.py              # Module 3 — Pipeline RAG + LLM Ollama
├── app.py                  # Interface utilisateur Streamlit
├── verifier_Module1.py     # Audit qualité Module 1 (score 100/100)
│
├── data_brute/             # Pages HTML brutes (01_scraping output)
├── data_propre/            # Chunks finaux JSON  (02_nettoyage output)
├── vector_db/              # Base ChromaDB       (04_embedding output)
│
└── requirements.txt        # Dépendances Python
```

---

## 📊 Résultats du pipeline

### Module 1 — Préparation des données

| Étape | Score |
|---|---|
| Scraper | 100/100 |
| Cleaner | 100/100 |
| Chunker | 100/100 |
| Cross-validation | 100/100 |

- **65 pages** scrapées depuis flask.palletsprojects.com
- **285 sections** extraites (vrais titres H2/H3)
- **388 chunks** prêts pour l'embedding
- **341 blocs de code** associés

### Module 2 — Base vectorielle

| Métrique | Valeur |
|---|---|
| Modèle d'embedding | all-MiniLM-L6-v2 |
| Dimensions | 384 |
| Documents indexés | 388 |
| Taille ChromaDB | 4.9 MB |
| Métrique de distance | cosine |

### Module 3 — Chatbot RAG

| Composant | Technologie                             |
|-------|---------------------------------------------|
| LLM   |        mistral` via Ollama (local, gratuit) |
| Retrieval top-k | 3 chunks                          |
| Seuil de similarité | 0.25                          |
| Interface | Streamlit                               |

---

## ⚙️ Installation

### 1. Cloner le dépôt

```bash
git clone https://github.com/siwarBnSalah/chatbot_doc_flask.git
cd chatbot_doc_flask
```

### 2. Créer l'environnement virtuel

```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux / macOS
```

### 3. Installer les dépendances Python

```bash
pip install -r requirements.txt
```

### 4. Installer Ollama et télécharger le modèle LLM

```bash
# Télécharger Ollama : https://ollama.ai
ollama pull mistral
```

---

## 🚀 Utilisation

### Lancer le pipeline complet (première fois)

```bash
# Étape 1 — Scraper la documentation Flask
python 01_scraping.py

# Étape 2 — Nettoyer 
python 02_nettoyage.py

# Étape 3 —  découper en chunks
python 03_chunking.py

# Étape 4 — Générer les embeddings et construire ChromaDB
python 04_embedding.py

# Étape 5 — Lancer Ollama (dans un terminal séparé)
ollama serve

# Étape 6 — Lancer l'interface Streamlit
streamlit run app.py
```

### Tester le chatbot en ligne de commande

```bash
python chatbot.py "How to create a Flask route?"
python chatbot.py "Comment activer le mode debug Flask ?"
python chatbot.py --interactive
```

### Vérifier la qualité du pipeline

```bash
python verifier_Module1.py
python verifier_Module1.py --report    # génère verification_report.json
python 04_embedding.py --stats         # statistiques ChromaDB
```

---

## 🔧 Dépendances principales

```
requests>=2.31.0          # Scraping
beautifulsoup4>=4.12.0    # Parsing HTML
sentence-transformers>=2.7.0  # Embeddings
chromadb>=0.5.0           # Base vectorielle
streamlit>=1.35.0         # Interface web
torch>=2.1.0              # Backend PyTorch
```

Voir `requirements.txt` pour la liste complète.

---

## 💡 Exemples de questions

```
✅ "How to create a Flask application?"
✅ "Comment définir des routes dans Flask ?"
✅ "How to use Flask blueprints?"
✅ "Comment gérer les erreurs HTTP dans Flask ?"
✅ "How to configure Flask debug mode?"
✅ "Comment utiliser les templates Jinja2 ?"
```

---

## ⚠️ Notes importantes

- Le dossier `vector_db/` n'est pas inclus dans le dépôt (régénérable avec `03_embedding.py`)
- Le modèle `all-MiniLM-L6-v2` est téléchargé automatiquement au premier lancement
- Ollama doit être lancé (`ollama serve`) avant d'utiliser le chatbot
- RAM minimum recommandée : **8 GB** pour `mistral`

---

## 👩‍💻 Auteur

**Siwar BenSalah** — Projet de Fin d'Études (PFE)  
Master en Datascience