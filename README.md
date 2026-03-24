# Chatbot Documentation Flask — RAG Pipeline

**Projet PFE** — Siwar  
Système de chatbot basé sur la documentation officielle Flask (3.0.x)
utilisant une architecture RAG (Retrieval-Augmented Generation).

## Architecture du Module 1 — Préparation des données
```
Scraper → Cleaner → Chunker → Vérificateur
```

## Structure du projet
```
chatbot_doc_flask/
├── 01_scraping.py          # Scrape la doc Flask (65 pages)
├── 02_netoyage.py      # Nettoie le HTML et extrait les sections H2/H3
├── 03-chunking.py          # Découpe en chunks pour l'embedding
├── verifier_Module1.py     # Audit qualité du pipeline (score 100/100)
├── data_brute/         # Pages HTML brutes (scraper output)
├── data_cleaned/       # Pages structurées par sections (cleaner output)
└── data_propre/        # Chunks finaux prêts pour l'embedding
```

## Résultats Module 1

| Étape | Score |
|-------|-------|
| Scraper | 100/100 |
| Cleaner | 100/100 |
| Chunker | 100/100 |
| Cross-validation | 100/100 |

- **65 pages** scrapées
- **285 sections** extraites (vrais titres H2/H3)
- **388 chunks** prêts pour l'embedding
- **341 blocs de code** associés

## Installation
```bash
git clone https://github.com/siwarBnSalah/chatbot_doc_flask.git
cd chatbot_doc_flask
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

## Utilisation
```bash
# Lancer le pipeline complet
python 01_scraping.py
python 02_netoyage.py
python 03-chunking.py

# Vérifier la qualité
python verifier_Module1.py
python verifier_Module1.py --report   # génère verification_report.json
```

## Dépendances

Voir `requirements.txt`