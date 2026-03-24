"""
Module 1 - Scraper : Documentation Flask
Projet : doc_support_rag
Description : Scrape la documentation officielle Flask et sauvegarde les pages brutes en JSON.

Corrections appliquées :
  - Sauvegarde du HTML brut structuré (pas de get_text() ici)
    → le cleaner pourra extraire les vrais H2/H3
  - Sélecteurs toctree élargis pour capturer tous les liens
  - Déduplication des URLs (seen_urls)
  - Timeout et gestion d'erreurs robuste
"""

import requests
from bs4 import BeautifulSoup
import json
import time
import logging
from pathlib import Path
from urllib.parse import urljoin, urlparse

# ── Configuration ──────────────────────────────────────────────────────────────
BASE_URL               = "https://flask.palletsprojects.com/en/3.0.x/"
OUTPUT_DIR             = Path("data_brute")
OUTPUT_FILE            = OUTPUT_DIR / "flask_pages.json"
DELAY_BETWEEN_REQUESTS = 1.0   # secondes
MAX_PAGES              = 80    # toute la doc Flask ≈ 60 pages

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ── Fonctions ─────────────────────────────────────────────────────────────────

def get_all_doc_links(base_url: str) -> list[str]:
    """
    Récupère tous les liens internes depuis la page d'index Flask.
    """
    logger.info(f"Fetching index page: {base_url}")
    response = requests.get(base_url, timeout=15)
    response.raise_for_status()

    soup  = BeautifulSoup(response.text, "html.parser")
    links = set()

    # Sélecteurs Sphinx : sidebar nav + tous les niveaux de toctree
    selectors = [
        "nav a[href]",
        ".toctree-wrapper a[href]",
        ".toctree-l1 a[href]",
        ".toctree-l2 a[href]",
        ".toctree-l3 a[href]",
    ]
    for sel in selectors:
        for a_tag in soup.select(sel):
            href = a_tag.get("href", "")
            if href and not href.startswith(("http://", "https://", "#", "mailto:")):
                full_url = urljoin(base_url, href).split("#")[0]
                if urlparse(full_url).netloc == urlparse(base_url).netloc:
                    links.add(full_url)

    links.add(base_url)
    logger.info(f"Found {len(links)} unique doc pages")
    return sorted(links)


def scrape_page(url: str) -> dict | None:
    """
    Scrape une page Flask. Retourne url, title, html_content (HTML structuré).
    IMPORTANT : on sauvegarde le HTML, pas le texte — le cleaner a besoin des H2/H3.
    """
    try:
        response = requests.get(url, timeout=15)
        response.raise_for_status()
    except requests.RequestException as e:
        logger.warning(f"Failed to fetch {url}: {e}")
        return None

    soup = BeautifulSoup(response.text, "html.parser")

    # Titre H1 (sans le symbole ¶ Sphinx)
    title_tag = soup.find("h1")
    title     = title_tag.get_text(strip=True).replace("¶", "").strip() if title_tag else url

    # Zone de contenu principal — ordre de priorité pour Flask/Sphinx
    content_zone = (
        soup.find("div", class_="body")
        or soup.find("div", attrs={"role": "main"})
        or soup.find("article")
        or soup.find("main")
    )

    if not content_zone:
        logger.warning(f"No main content found for {url}")
        return None

    return {
        "url":          url,
        "title":        title,
        "html_content": str(content_zone)   # HTML brut préservé pour le cleaner
    }


# ── Pipeline ──────────────────────────────────────────────────────────────────

def run_scraper():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    links     = get_all_doc_links(BASE_URL)[:MAX_PAGES]
    seen_urls = set()
    pages     = []

    for i, url in enumerate(links, 1):
        if url in seen_urls:
            continue
        seen_urls.add(url)

        logger.info(f"[{i}/{len(links)}] Scraping: {url}")
        data = scrape_page(url)
        if data:
            pages.append(data)

        time.sleep(DELAY_BETWEEN_REQUESTS)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(pages, f, ensure_ascii=False, indent=2)

    logger.info(f"\n✅ Scraping terminé : {len(pages)} pages → {OUTPUT_FILE}")
    return pages


if __name__ == "__main__":
    run_scraper()