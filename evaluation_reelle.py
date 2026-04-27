"""
evaluation_reelle.py — Évaluation RÉELLE du pipeline RAG Flask
Projet : chatbot_doc_flask (PFE 2025) — Siwar BenSalah

CORRECTIONS APPLIQUÉES :
  ✅ Bug variable `r` : toutes variables initialisées avant le try
  ✅ MRR correct : rang réel du premier chunk pertinent dans la liste retournée
  ✅ Seaborn supprimé → matplotlib pur uniquement
  ✅ Seuils documentés avec justification empirique
  ✅ np.random.seed(42) pour reproductibilité
  ✅ Évaluation des 3 agents séparément (Routing + Retrieval + Reformulation)

DESIGN :
  ✅ Thème clair académique (blanc/gris) — adapté rapport PFE et soutenance
  ✅ Palette professionnelle : bleu #2563eb, vert #16a34a, rouge #dc2626
  ✅ Grille légère, bordures discrètes, typographie lisible
  ✅ Contraste optimal pour impression noir et blanc

Usage (depuis votre dossier projet, là où sont agents.py + vector_db/) :
    python evaluation_reelle.py
"""

from __future__ import annotations
import os, sys, json, time, warnings
from pathlib import Path
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)

warnings.filterwarnings("ignore")
np.random.seed(42)  # reproductibilité scientifique

# ── Chemins ───────────────────────────────────────────────────────────────────
PROJECT_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(PROJECT_DIR))
OUTPUT_DIR  = PROJECT_DIR / "evaluation_output"
OUTPUT_DIR.mkdir(exist_ok=True)

# ── Seuils documentés empiriquement ──────────────────────────────────────────
# Fixés après observation des scores de similarité cosinus sur le dataset Flask
# (modèle all-MiniLM-L6-v2, 384 dimensions, distance cosinus)
SEUIL_ROUTING = 0.30  # seuil routing agent  (défini dans agents.py RoutingAgent)
SEUIL_HIT_K   = 0.50  # score minimal pour qu'un chunk soit considéré "pertinent"
SEUIL_REFORM  = 0.35  # seuil déclenchant la reformulation (agents.py QueryReformulationAgent)

print(f"\n{'═'*65}")
print(f"  ÉVALUATION RÉELLE — Flask RAG Chatbot (PFE 2025)")
print(f"  Dossier projet  : {PROJECT_DIR}")
print(f"  Seuil routing   : {SEUIL_ROUTING}  (empirique, agents.py)")
print(f"  Seuil Hit@K     : {SEUIL_HIT_K}  (empirique)")
print(f"  Seuil reform.   : {SEUIL_REFORM}  (empirique, agents.py)")
print(f"{'═'*65}\n")

# ── Chargement du pipeline réel ───────────────────────────────────────────────
print("⏳ Chargement AgenticRAGPipeline (ChromaDB + embeddings)...")
try:
    from agents import AgenticRAGPipeline
    pipeline = AgenticRAGPipeline(top_k=3, max_context_chars=2000)
    print(f"✅ Pipeline chargé — {pipeline.collection_size} documents ChromaDB\n")
except FileNotFoundError as e:
    print(f"❌ vector_db/ introuvable : {e}")
    print("   → Lancez d'abord : python 04_embedding.py")
    sys.exit(1)
except Exception as e:
    print(f"❌ Erreur chargement pipeline : {e}")
    sys.exit(1)


# ─────────────────────────────────────────────────────────────────────────────
# DATASET DE TEST — 50 questions annotées manuellement
# label 1 = in-scope Flask  |  label 0 = hors-scope
# ─────────────────────────────────────────────────────────────────────────────
TEST_DATASET = [
    # ── IN-SCOPE Flask (25 questions) ─────────────────────────────────────────
    {"id":  1, "question": "How to create a Flask route?",                              "label": 1, "categorie": "routing"},
    {"id":  2, "question": "Comment créer une route Flask avec paramètre dynamique ?",  "label": 1, "categorie": "routing"},
    {"id":  3, "question": "What is a Flask blueprint?",                                "label": 1, "categorie": "blueprint"},
    {"id":  4, "question": "Comment utiliser les blueprints Flask ?",                   "label": 1, "categorie": "blueprint"},
    {"id":  5, "question": "How to enable debug mode in Flask?",                        "label": 1, "categorie": "debug"},
    {"id":  6, "question": "Comment activer le mode debug Flask ?",                     "label": 1, "categorie": "debug"},
    {"id":  7, "question": "How to use Flask sessions?",                                "label": 1, "categorie": "sessions"},
    {"id":  8, "question": "Comment gérer les sessions dans Flask ?",                   "label": 1, "categorie": "sessions"},
    {"id":  9, "question": "How to handle 404 errors in Flask?",                        "label": 1, "categorie": "errors"},
    {"id": 10, "question": "Comment gérer les erreurs HTTP dans Flask ?",               "label": 1, "categorie": "errors"},
    {"id": 11, "question": "How to use Jinja2 templates in Flask?",                     "label": 1, "categorie": "templates"},
    {"id": 12, "question": "Comment rendre un template Jinja2 avec Flask ?",            "label": 1, "categorie": "templates"},
    {"id": 13, "question": "Flask url_for function usage",                              "label": 1, "categorie": "routing"},
    {"id": 14, "question": "How to configure Flask application?",                       "label": 1, "categorie": "config"},
    {"id": 15, "question": "Flask request object how to use",                           "label": 1, "categorie": "request"},
    {"id": 16, "question": "How to return JSON response in Flask?",                     "label": 1, "categorie": "response"},
    {"id": 17, "question": "Comment déployer une application Flask ?",                  "label": 1, "categorie": "deployment"},
    {"id": 18, "question": "Flask application factory pattern",                         "label": 1, "categorie": "config"},
    {"id": 19, "question": "How to test a Flask application?",                          "label": 1, "categorie": "testing"},
    {"id": 20, "question": "Flask before_request decorator usage",                      "label": 1, "categorie": "hooks"},
    {"id": 21, "question": "How to use Flask-Login extension?",                         "label": 1, "categorie": "extensions"},
    {"id": 22, "question": "Flask werkzeug security utilities",                         "label": 1, "categorie": "security"},
    {"id": 23, "question": "How to stream responses in Flask?",                         "label": 1, "categorie": "response"},
    {"id": 24, "question": "Flask context globals g and current_app",                   "label": 1, "categorie": "context"},
    {"id": 25, "question": "Comment installer Flask avec pip ?",                        "label": 1, "categorie": "installation"},
    # ── HORS-SCOPE (15 questions) ─────────────────────────────────────────────
    {"id": 26, "question": "Comment créer un modèle de machine learning avec PyTorch ?","label": 0, "categorie": "ml_hors_scope"},
    {"id": 27, "question": "Quelle est la recette de la tarte aux pommes ?",            "label": 0, "categorie": "non_technique"},
    {"id": 28, "question": "Comment définir des routes dans Django ?",                  "label": 0, "categorie": "autre_framework"},
    {"id": 29, "question": "How to train a neural network with TensorFlow?",            "label": 0, "categorie": "ml_hors_scope"},
    {"id": 30, "question": "Quel est le PIB de la France en 2024 ?",                   "label": 0, "categorie": "non_technique"},
    {"id": 31, "question": "Comment créer un composant React ?",                        "label": 0, "categorie": "autre_framework"},
    {"id": 32, "question": "FastAPI tutorial getting started",                          "label": 0, "categorie": "autre_framework"},
    {"id": 33, "question": "Django ORM models tutorial",                                "label": 0, "categorie": "autre_framework"},
    {"id": 34, "question": "Pandas DataFrame groupby tutorial",                         "label": 0, "categorie": "ml_hors_scope"},
    {"id": 35, "question": "Comment faire du deep learning avec Keras ?",               "label": 0, "categorie": "ml_hors_scope"},
    {"id": 36, "question": "Quelle est la capitale de l'Allemagne ?",                  "label": 0, "categorie": "non_technique"},
    {"id": 37, "question": "Vue.js component lifecycle hooks",                          "label": 0, "categorie": "autre_framework"},
    {"id": 38, "question": "How to use NumPy arrays for matrix operations?",            "label": 0, "categorie": "ml_hors_scope"},
    {"id": 39, "question": "JavaScript async await promises tutorial",                  "label": 0, "categorie": "autre_framework"},
    {"id": 40, "question": "Comment cuisiner des pâtes carbonara ?",                    "label": 0, "categorie": "non_technique"},
    # ── CAS AMBIGUS (10 questions) ────────────────────────────────────────────
    {"id": 41, "question": "Python web server deployment best practices",               "label": 1, "categorie": "deployment"},
    {"id": 42, "question": "REST API with Python",                                      "label": 1, "categorie": "routing"},
    {"id": 43, "question": "comment faire une page web avec python flask",              "label": 1, "categorie": "routing"},
    {"id": 44, "question": "web framework python lightweight",                          "label": 1, "categorie": "config"},
    {"id": 45, "question": "Python HTTP request handling server side",                  "label": 1, "categorie": "request"},
    {"id": 46, "question": "JWT token authentication python web",                       "label": 1, "categorie": "security"},
    {"id": 47, "question": "gunicorn wsgi server python",                               "label": 1, "categorie": "deployment"},
    {"id": 48, "question": "SQL database ORM python",                                   "label": 0, "categorie": "ml_hors_scope"},
    {"id": 49, "question": "microservices python REST",                                 "label": 1, "categorie": "deployment"},
    {"id": 50, "question": "ChatGPT OpenAI API integration",                           "label": 0, "categorie": "ml_hors_scope"},
]


# ─────────────────────────────────────────────────────────────────────────────
# DATACLASS RÉSULTAT
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class Result:
    id:               int
    question:         str
    label_true:       int
    label_pred:       int
    in_scope:         bool
    routing_action:   str
    sources:          list
    top_score:        float
    chunks_found:     int
    reformulated:     bool
    reform_triggered: bool
    categorie:        str
    elapsed_ms:       float
    error:            str = ""


# ─────────────────────────────────────────────────────────────────────────────
# ÉVALUATION — APPEL AU VRAI PIPELINE
# ─────────────────────────────────────────────────────────────────────────────
results: list[Result] = []

print(f"  {'#':>3}  {'✓/✗':>4}  {'Décision':>9}  {'Score':>7}  {'Ref.':>5}  Question")
print(f"  {'─'*72}")

for item in TEST_DATASET:
    t0 = time.perf_counter()

    # Toutes variables initialisées AVANT le try → pas de crash si exception
    sources_list  = []
    top_score     = 0.0
    action        = "error"
    in_scope      = False
    reformed      = False
    reform_trig   = False
    label_pred    = 0
    error_msg     = ""

    try:
        r = pipeline.run(item["question"])
        elapsed = (time.perf_counter() - t0) * 1000

        in_scope     = r.get("in_scope", False)
        sources_list = r.get("sources", [])
        top_score    = max((s["score"] for s in sources_list), default=0.0)
        action       = r.get("agent_metadata", {}).get("routing_action", "?")
        reformed     = r.get("reformulated", False)
        best_score   = r.get("agent_metadata", {}).get("best_score", top_score)
        reform_trig  = (best_score < SEUIL_REFORM) and in_scope
        label_pred   = 1 if in_scope else 0

    except Exception as e:
        elapsed   = (time.perf_counter() - t0) * 1000
        error_msg = str(e)

    res = Result(
        id=item["id"], question=item["question"],
        label_true=item["label"], label_pred=label_pred,
        in_scope=in_scope, routing_action=action,
        sources=sources_list,
        top_score=top_score,
        chunks_found=len(sources_list),
        reformulated=reformed,
        reform_triggered=reform_trig,
        categorie=item["categorie"],
        elapsed_ms=round(elapsed, 1),
        error=error_msg
    )
    results.append(res)

    icon    = "✅" if label_pred == item["label"] else "❌"
    scope   = "IN-SCOPE " if in_scope else "HORS-SC. "
    ref_ico = "🔄" if reformed else "--"
    print(f"  {item['id']:>3}  {icon}  {scope:>9}  {top_score:>7.3f}  {ref_ico:>5}  "
          f"{item['question'][:36]}")

errors_count = sum(1 for r in results if r.error)
print(f"\n  ✅ {len(results)} questions évaluées | {errors_count} erreurs pipeline\n")


# ─────────────────────────────────────────────────────────────────────────────
# CALCUL DES MÉTRIQUES
# ─────────────────────────────────────────────────────────────────────────────
y_true = [r.label_true for r in results]
y_pred = [r.label_pred for r in results]

df = pd.DataFrame([
    {k: v for k, v in asdict(r).items() if k != "sources"}
    for r in results
])

# ── AGENT 1 : Routing ─────────────────────────────────────────────────────────
acc  = accuracy_score(y_true, y_pred)
prec = precision_score(y_true, y_pred, zero_division=0)
rec  = recall_score(y_true, y_pred, zero_division=0)
f1   = f1_score(y_true, y_pred, zero_division=0)
cm   = confusion_matrix(y_true, y_pred)
rpt  = classification_report(y_true, y_pred,
           target_names=["Hors-scope (0)", "In-scope (1)"],
           output_dict=True, zero_division=0)

out_df  = df[df["label_true"] == 0]
rej_acc = (out_df["label_pred"] == 0).mean() if len(out_df) > 0 else 0.0
fpr     = (out_df["label_pred"] == 1).mean() if len(out_df) > 0 else 0.0

# ── AGENT 2 : Retrieval ───────────────────────────────────────────────────────
in_df     = df[df["in_scope"] == True]
avg_score = in_df["top_score"].mean() if len(in_df) > 0 else 0.0

hit_k_list = []
for r in results:
    if r.in_scope:
        hit = any(s["score"] >= SEUIL_HIT_K for s in r.sources)
        hit_k_list.append(1 if hit else 0)
hit_k = np.mean(hit_k_list) if hit_k_list else 0.0

# MRR correct : rang réel du premier chunk pertinent
mrr_list = []
for r in results:
    if not r.in_scope:
        continue
    rr = 0.0
    for rank_idx, s in enumerate(r.sources):
        if s["score"] >= SEUIL_HIT_K:
            rr = 1.0 / (rank_idx + 1)
            break
    mrr_list.append(rr)
mrr = np.mean(mrr_list) if mrr_list else 0.0

pk_list = []
for r in results:
    if r.in_scope and r.chunks_found > 0:
        relevant = sum(1 for s in r.sources if s["score"] >= SEUIL_HIT_K)
        pk_list.append(relevant / r.chunks_found)
precision_at_k = np.mean(pk_list) if pk_list else 0.0

# ── AGENT 3 : Reformulation ────────────────────────────────────────────────────
reform_triggered  = df["reform_triggered"].sum()
reform_rate       = df["reform_triggered"].mean()
reformulated_n    = df["reformulated"].sum()
reform_success_df = df[df["reformulated"] == True]
avg_score_after_reform = reform_success_df["top_score"].mean() \
    if len(reform_success_df) > 0 else 0.0

avg_ms = df["elapsed_ms"].mean()


# ─────────────────────────────────────────────────────────────────────────────
# AFFICHAGE CONSOLE
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{'═'*65}")
print(f"  AGENT 1 — ROUTING (Classification Binaire)")
print(f"{'═'*65}")
print(f"  Accuracy              : {acc:.4f}  ({acc*100:.1f}%)")
print(f"  Precision             : {prec:.4f}")
print(f"  Recall                : {rec:.4f}")
print(f"  F1-score              : {f1:.4f}")
print(f"  Rejection Accuracy    : {rej_acc:.4f}  (hors-scope correctement rejetés)")
print(f"  False Positive Rate   : {fpr:.4f}   (hors-scope acceptés à tort)")
print(f"\n  Rapport par classe :")
print(f"  {'Classe':<22} {'Precision':>10} {'Recall':>8} {'F1':>8} {'N':>6}")
print(f"  {'─'*57}")
for cls, v in rpt.items():
    if isinstance(v, dict):
        print(f"  {cls:<22} {v['precision']:>10.3f} {v['recall']:>8.3f} "
              f"{v['f1-score']:>8.3f} {int(v['support']):>6}")
print(f"\n  Matrice de Confusion :")
print(f"                         Prédit Hors-sc.   Prédit In-sc.")
print(f"  Réel Hors-scope              TN={cm[0,0]:>4}          FP={cm[0,1]:>4}")
print(f"  Réel In-scope                FN={cm[1,0]:>4}          TP={cm[1,1]:>4}")

print(f"\n{'═'*65}")
print(f"  AGENT 2 — RETRIEVAL (ChromaDB Similarity Search)")
print(f"{'═'*65}")
print(f"  Score similarité moyen : {avg_score:.4f}   (seuil empirique={SEUIL_ROUTING})")
print(f"  Hit@K  (score≥{SEUIL_HIT_K})     : {hit_k:.4f}   ({hit_k*100:.1f}%)")
print(f"  MRR    (rang réel)     : {mrr:.4f}   (1.0=rang1, 0.5=rang2, 0.33=rang3)")
print(f"  Precision@K            : {precision_at_k:.4f}   (chunks pertinents / total retournés)")

print(f"\n{'═'*65}")
print(f"  AGENT 3 — REFORMULATION (Query Improvement)")
print(f"{'═'*65}")
print(f"  Déclenchements total   : {int(reform_triggered)}/{len(results)}")
print(f"  Taux déclenchement     : {reform_rate:.4f}  ({reform_rate*100:.1f}%)")
print(f"  Reformulations réussl. : {int(reformulated_n)}")
if len(reform_success_df) > 0:
    print(f"  Score moyen après ref. : {avg_score_after_reform:.4f}")

print(f"\n{'═'*65}")
print(f"  PIPELINE GLOBAL")
print(f"{'═'*65}")
print(f"  Temps moyen pipeline   : {avg_ms:.1f} ms")
print(f"  Documents ChromaDB     : {pipeline.collection_size}")
print(f"  Erreurs                : {errors_count}")

cats_rows = []
for cat in sorted(df["categorie"].unique()):
    sub = df[df["categorie"] == cat]
    a   = accuracy_score(sub["label_true"], sub["label_pred"])
    f   = f1_score(sub["label_true"], sub["label_pred"], zero_division=0)
    cats_rows.append({"Catégorie": cat, "N": len(sub),
                      "Accuracy": round(a, 3), "F1": round(f, 3),
                      "Score Moy.": round(sub["top_score"].mean(), 3)})
cats_df = pd.DataFrame(cats_rows).sort_values("Accuracy", ascending=False)
print(f"\n  Performance par catégorie :")
print(cats_df.to_string(index=False))


# ─────────────────────────────────────────────────────────────────────────────
# COULEURS — Thème académique clair (rapport PFE / soutenance)
# ─────────────────────────────────────────────────────────────────────────────
BG   = "#ffffff"   # fond blanc pur
CARD = "#f8fafc"   # fond carte gris très clair
PRI  = "#2563eb"   # bleu principal (titres, axes, seuils)
SEC  = "#7c3aed"   # violet doux (Agent 2 Retrieval)
GRN  = "#16a34a"   # vert (bon résultat)
RED  = "#dc2626"   # rouge (mauvais résultat / FP)
ORG  = "#ea580c"   # orange (résultat moyen)
TXT  = "#0f172a"   # texte principal noir doux
SUB  = "#64748b"   # texte secondaire gris
GRID = "#e2e8f0"   # grille légère
EDGE = "#cbd5e1"   # bordures discrètes

plt.rcParams.update({
    # Fonds
    "figure.facecolor":  BG,
    "axes.facecolor":    CARD,
    # Bordures des axes
    "axes.edgecolor":    EDGE,
    "axes.linewidth":    0.8,
    # Couleurs du texte
    "axes.labelcolor":   TXT,
    "xtick.color":       SUB,
    "ytick.color":       SUB,
    "text.color":        TXT,
    # Grille légère
    "axes.grid":         True,
    "grid.color":        GRID,
    "grid.linewidth":    0.6,
    "grid.alpha":        1.0,
    "axes.axisbelow":    True,
    # Typographie
    "font.family":       "DejaVu Sans",
    "font.size":         11,
    "axes.titlesize":    13,
    "axes.labelsize":    11,
    "xtick.labelsize":   10,
    "ytick.labelsize":   10,
    "legend.fontsize":   9,
    # Ticks discrets
    "xtick.major.size":  3,
    "ytick.major.size":  3,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    # Légende
    "legend.framealpha": 0.95,
    "legend.edgecolor":  EDGE,
    "legend.facecolor":  BG,
    # Sauvegarde
    "savefig.facecolor": BG,
    "savefig.edgecolor": "none",
    "savefig.dpi":       150,
})

saved = []

def savefig(name: str) -> Path:
    p = OUTPUT_DIR / name
    plt.savefig(p, dpi=150, bbox_inches="tight",
                facecolor=BG, edgecolor="none")
    plt.close()
    print(f"  💾 {name}")
    return p


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 1 — Dashboard synthèse
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n  Génération des graphiques (thème académique clair)...")
fig = plt.figure(figsize=(18, 10))
fig.patch.set_facecolor(BG)
gs  = fig.add_gridspec(3, 4, hspace=0.55, wspace=0.40)

# ── KPI circles ──────────────────────────────────────────────────────────────
kpis = [("Accuracy", acc, GRN), ("F1-score", f1, PRI),
        ("Hit@K",    hit_k, SEC), ("MRR",     mrr, ORG)]
for col, (name, val, color) in enumerate(kpis):
    ax = fig.add_subplot(gs[0, col])
    ax.set_facecolor(CARD)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    # Cercle fond + contour
    circle_bg = plt.Circle((0.5, 0.52), 0.40, color=color, alpha=0.10, zorder=1)
    circle_bd = plt.Circle((0.5, 0.52), 0.40,
                            fill=False, edgecolor=color, linewidth=2.0, zorder=2)
    ax.add_patch(circle_bg)
    ax.add_patch(circle_bd)

    ax.text(0.5, 0.57, f"{val:.3f}", ha="center", va="center",
            fontsize=20, fontweight="bold", color=color, zorder=3)
    ax.text(0.5, 0.22, name, ha="center", va="center",
            fontsize=10, color=SUB, zorder=3)

    for sp in ax.spines.values():
        sp.set_edgecolor(EDGE); sp.set_linewidth(1.0); sp.set_visible(True)

# ── Matrice de confusion ──────────────────────────────────────────────────────
ax_cm = fig.add_subplot(gs[1:, :2])
ax_cm.set_facecolor(CARD)
ax_cm.imshow(cm, cmap="Blues", aspect="auto", vmin=0, vmax=cm.max())
ax_cm.set_xticks([0, 1]); ax_cm.set_yticks([0, 1])
ax_cm.set_xticklabels(["Hors-scope", "In-scope"], fontsize=11, color=TXT)
ax_cm.set_yticklabels(["Hors-scope", "In-scope"], fontsize=11, color=TXT)
ax_cm.set_xlabel("Classe Prédite", fontsize=12, fontweight="bold", color=TXT)
ax_cm.set_ylabel("Classe Réelle",  fontsize=12, fontweight="bold", color=TXT)
ax_cm.set_title("Matrice de Confusion\n(Routing Agent)",
                fontsize=12, fontweight="bold", color=PRI, pad=14)
lbl_cm = [["TN", "FP"], ["FN", "TP"]]
for i in range(2):
    for j in range(2):
        c_txt = "white" if cm[i, j] > cm.max() * 0.55 else TXT
        ax_cm.text(j, i - 0.12, str(cm[i, j]), ha="center", va="center",
                   fontsize=24, fontweight="bold", color=c_txt)
        ax_cm.text(j, i + 0.28, lbl_cm[i][j], ha="center", va="center",
                   fontsize=11, color=c_txt, alpha=0.80)

# ── Bar chart Agent 1 ─────────────────────────────────────────────────────────
ax_bar = fig.add_subplot(gs[1, 2:])
ax_bar.set_facecolor(CARD)
m_names = ["Accuracy", "Precision", "Recall", "F1"]
m_vals  = [acc, prec, rec, f1]
bcols   = [GRN if v >= 0.80 else ORG if v >= 0.60 else RED for v in m_vals]
bars = ax_bar.bar(m_names, m_vals, color=bcols, width=0.50,
                  edgecolor="white", linewidth=0.8)
for b, v in zip(bars, m_vals):
    ax_bar.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.013,
                f"{v:.3f}", ha="center", fontweight="bold", fontsize=11, color=TXT)
ax_bar.axhline(0.80, color=PRI, linestyle="--", alpha=0.80,
               linewidth=1.5, label="Seuil 0.80")
ax_bar.set_ylim(0, 1.18)
ax_bar.set_title("Agent 1 — Routing Metrics",
                 fontsize=11, fontweight="bold", color=PRI)
ax_bar.legend(facecolor=BG, edgecolor=EDGE, fontsize=9)
ax_bar.tick_params(axis="x", colors=TXT)

# ── Bar chart Agent 2 ─────────────────────────────────────────────────────────
ax_rag = fig.add_subplot(gs[2, 2:])
ax_rag.set_facecolor(CARD)
r_names = ["Hit@K", "MRR", "Prec@K", "Avg\nScore"]
r_vals  = [hit_k, mrr, precision_at_k, avg_score]
rcols   = [GRN if v >= 0.75 else ORG if v >= 0.50 else RED for v in r_vals]
bars2 = ax_rag.bar(r_names, r_vals, color=rcols, width=0.45,
                   edgecolor="white", linewidth=0.8)
for b, v in zip(bars2, r_vals):
    ax_rag.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.013,
                f"{v:.3f}", ha="center", fontweight="bold", fontsize=11, color=TXT)
ax_rag.set_ylim(0, 1.18)
ax_rag.set_title("Agent 2 — Retrieval Metrics",
                 fontsize=11, fontweight="bold", color=SEC)
ax_rag.tick_params(axis="x", colors=TXT)

fig.suptitle(
    "Dashboard Évaluation RÉELLE — Flask RAG Chatbot PFE 2025\n"
    "AgenticRAGPipeline (3 agents) + ChromaDB (all-MiniLM-L6-v2, 384 dims)",
    fontsize=13, fontweight="bold", color=TXT, y=1.01)
saved.append(savefig("00_dashboard.png"))


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 2 — Matrice de confusion détaillée
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 6))
fig.patch.set_facecolor(BG)
ax.set_facecolor(CARD)
ax.imshow(cm, cmap="Blues", aspect="auto", vmin=0, vmax=cm.max())
ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
ax.set_xticklabels(["Hors-scope (0)", "In-scope (1)"], fontsize=12, color=TXT)
ax.set_yticklabels(["Hors-scope (0)", "In-scope (1)"], fontsize=12, color=TXT)
ax.set_xlabel("Classe Prédite (Routing Agent)", fontsize=13,
              fontweight="bold", color=TXT)
ax.set_ylabel("Classe Réelle (Ground Truth)",   fontsize=13,
              fontweight="bold", color=TXT)
ax.set_title(
    f"Matrice de Confusion — Routing Agent\n"
    f"Accuracy={acc:.3f}  |  F1={f1:.3f}  |  N={len(results)} questions",
    fontsize=13, fontweight="bold", color=PRI, pad=18)

desc  = [["TN\n(Rejet correct)", "FP\n(Faux positif)"],
         ["FN\n(Faux négatif)",  "TP\n(Détection correcte)"]]
dcols = [[GRN, RED], [ORG, GRN]]
for i in range(2):
    for j in range(2):
        c_txt = "white" if cm[i, j] > cm.max() * 0.55 else TXT
        ax.text(j, i - 0.12, str(cm[i, j]), ha="center", va="center",
                fontsize=28, fontweight="bold", color=c_txt)
        ax.text(j, i + 0.28, desc[i][j], ha="center", va="center",
                fontsize=8, color=dcols[i][j], alpha=0.95)

handles = [
    mpatches.Patch(color=GRN, label="Correct  (TN + TP)", alpha=0.85),
    mpatches.Patch(color=RED, label="Faux Positif (FP)",  alpha=0.85),
    mpatches.Patch(color=ORG, label="Faux Négatif (FN)",  alpha=0.85),
]
ax.legend(handles=handles, loc="upper right",
          bbox_to_anchor=(1.42, 1.02),
          facecolor=BG, edgecolor=EDGE, fontsize=9, framealpha=0.95)
plt.tight_layout()
saved.append(savefig("01_confusion_matrix.png"))


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 3 — 3 agents côte à côte
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(17, 5))
fig.patch.set_facecolor(BG)

# ── Agent 1 : Routing ─────────────────────────────────────────────────────────
ax = axes[0]; ax.set_facecolor(CARD)
n1 = ["Accuracy", "Precision", "Recall", "F1", "Rej.\nAcc."]
v1 = [acc, prec, rec, f1, rej_acc]
c1 = [GRN if v >= 0.80 else ORG if v >= 0.60 else RED for v in v1]
bars = ax.bar(n1, v1, color=c1, width=0.55, edgecolor="white", linewidth=0.8)
for b, v in zip(bars, v1):
    ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.013,
            f"{v:.3f}", ha="center", fontsize=9, fontweight="bold", color=TXT)
ax.axhline(0.80, color=PRI, linestyle="--", alpha=0.80, linewidth=1.5)
ax.set_ylim(0, 1.18)
ax.set_title("Agent 1 — ROUTING\n(Classification binaire)",
             fontsize=11, fontweight="bold", color=PRI)
ax.tick_params(axis="x", colors=TXT)

# ── Agent 2 : Retrieval ───────────────────────────────────────────────────────
ax = axes[1]; ax.set_facecolor(CARD)
n2 = ["Hit@K", "MRR\n(rang réel)", "Prec@K", "Avg\nScore"]
v2 = [hit_k, mrr, precision_at_k, avg_score]
c2 = [GRN if v >= 0.75 else ORG if v >= 0.50 else RED for v in v2]
bars = ax.bar(n2, v2, color=c2, width=0.55, edgecolor="white", linewidth=0.8)
for b, v in zip(bars, v2):
    ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.013,
            f"{v:.3f}", ha="center", fontsize=9, fontweight="bold", color=TXT)
ax.axhline(0.75, color=PRI, linestyle="--", alpha=0.80, linewidth=1.5)
ax.set_ylim(0, 1.18)
ax.set_title("Agent 2 — RETRIEVAL\n(ChromaDB similarity)",
             fontsize=11, fontweight="bold", color=SEC)
ax.tick_params(axis="x", colors=TXT)

# ── Agent 3 : Reformulation (pie) ─────────────────────────────────────────────
ax = axes[2]; ax.set_facecolor(CARD)
in_n   = int(df["in_scope"].sum())
ref_n  = int(df["reformulated"].sum())
nref_n = in_n - ref_n
out_n  = len(results) - in_n
wedges, texts, autotexts = ax.pie(
    [nref_n, ref_n, out_n],
    labels=["In-scope\nsans reform.", "In-scope\navec reform.", "Hors-scope\n(rejeté)"],
    colors=[GRN, ORG, RED],
    autopct="%1.1f%%", startangle=90, pctdistance=0.70,
    wedgeprops={"edgecolor": BG, "linewidth": 2})
for t in texts:
    t.set_color(TXT); t.set_fontsize(10)
for at in autotexts:
    at.set_color("white"); at.set_fontweight("bold"); at.set_fontsize(9)
ax.set_title(
    f"Agent 3 — REFORMULATION\n(déclenchée {int(reform_triggered)}× / {len(results)} questions)",
    fontsize=11, fontweight="bold", color=ORG)

fig.suptitle("Évaluation Séparée des 3 Agents — Flask RAG Pipeline PFE 2025",
             fontsize=13, fontweight="bold", color=TXT, y=1.03)
plt.tight_layout()
saved.append(savefig("02_three_agents.png"))


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 4 — Performance par catégorie
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.patch.set_facecolor(BG)

# ── Accuracy par catégorie ────────────────────────────────────────────────────
ax = axes[0]; ax.set_facecolor(CARD)
ca = [GRN if v >= 0.80 else ORG if v >= 0.50 else RED for v in cats_df["Accuracy"]]
bars = ax.barh(cats_df["Catégorie"], cats_df["Accuracy"],
               color=ca, edgecolor="white", linewidth=0.8)
for b, v in zip(bars, cats_df["Accuracy"]):
    ax.text(v + 0.012, b.get_y() + b.get_height() / 2,
            f"{v:.2f}", va="center", fontsize=9, fontweight="bold", color=TXT)
ax.axvline(0.80, color=PRI, linestyle="--", alpha=0.80, linewidth=1.5)
ax.set_xlim(0, 1.15)
ax.set_xlabel("Accuracy", fontsize=11, color=TXT)
ax.set_title("Accuracy par Catégorie\n(Agent 1 — Routing)",
             fontsize=11, fontweight="bold", color=PRI)
ax.tick_params(axis="y", colors=TXT)

# ── Score retrieval par catégorie ─────────────────────────────────────────────
ax2 = axes[1]; ax2.set_facecolor(CARD)
cs = [GRN if v >= 0.60 else ORG if v >= 0.35 else RED
      for v in cats_df["Score Moy."]]
bars2 = ax2.barh(cats_df["Catégorie"], cats_df["Score Moy."],
                 color=cs, edgecolor="white", linewidth=0.8)
for b, v in zip(bars2, cats_df["Score Moy."]):
    ax2.text(v + 0.006, b.get_y() + b.get_height() / 2,
             f"{v:.3f}", va="center", fontsize=9, fontweight="bold", color=TXT)
ax2.axvline(SEUIL_HIT_K, color=ORG, linestyle="--", alpha=0.80,
            linewidth=1.5, label=f"Seuil Hit@K ({SEUIL_HIT_K})")
ax2.set_xlim(0, 1.05)
ax2.set_xlabel("Score Similarité Moyen (ChromaDB)", fontsize=11, color=TXT)
ax2.set_title("Score Retrieval par Catégorie\n(Agent 2 — Retrieval)",
              fontsize=11, fontweight="bold", color=SEC)
ax2.legend(facecolor=BG, edgecolor=EDGE, fontsize=9)
ax2.tick_params(axis="y", colors=TXT)

fig.suptitle("Performance par Catégorie — Évaluation Réelle PFE 2025",
             fontsize=12, fontweight="bold", color=TXT, y=1.02)
plt.tight_layout()
saved.append(savefig("03_by_category.png"))


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 5 — Distribution scores + pie routing
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.patch.set_facecolor(BG)

# ── Histogramme scores ────────────────────────────────────────────────────────
ax = axes[0]; ax.set_facecolor(CARD)
scores_in  = df[df["in_scope"] == True]["top_score"].values
scores_out = df[(df["in_scope"] == False) & (df["top_score"] > 0)]["top_score"].values
if len(scores_in) > 0:
    ax.hist(scores_in,  bins=np.linspace(0, 1, 16), color=GRN, alpha=0.75,
            edgecolor="white", linewidth=0.6,
            label=f"In-scope (n={len(scores_in)})")
if len(scores_out) > 0:
    ax.hist(scores_out, bins=np.linspace(0, 1, 10), color=RED, alpha=0.75,
            edgecolor="white", linewidth=0.6,
            label=f"Hors-scope avec score (n={len(scores_out)})")
ax.axvline(SEUIL_ROUTING, color=ORG, linestyle="--", linewidth=2,
           label=f"Seuil routing ({SEUIL_ROUTING})")
ax.axvline(SEUIL_HIT_K,   color=PRI, linestyle="--", linewidth=2,
           label=f"Seuil Hit@K ({SEUIL_HIT_K})")
ax.set_xlabel("Score Similarité Cosinus (ChromaDB)", fontsize=11, color=TXT)
ax.set_ylabel("Nombre de Questions",                 fontsize=11, color=TXT)
ax.set_title("Distribution Réelle des\nScores de Retrieval",
             fontsize=12, fontweight="bold", color=PRI)
ax.legend(facecolor=BG, edgecolor=EDGE, fontsize=9)
ax.tick_params(colors=TXT)

# ── Pie décisions routing ─────────────────────────────────────────────────────
ax2 = axes[1]; ax2.set_facecolor(CARD)
actions  = df["routing_action"].value_counts()
a_colors = {"proceed": GRN, "warn": ORG, "reject": RED, "error": SUB, "?": SUB}
pie_c    = [a_colors.get(a, SUB) for a in actions.index]
wedges, texts, autotexts = ax2.pie(
    actions.values, labels=actions.index, colors=pie_c,
    autopct="%1.1f%%", startangle=90, pctdistance=0.70,
    wedgeprops={"edgecolor": BG, "linewidth": 2})
for t in texts:
    t.set_color(TXT); t.set_fontsize(11)
for at in autotexts:
    at.set_color("white"); at.set_fontweight("bold"); at.set_fontsize(10)
ax2.set_title("Décisions Routing Agent\n(données réelles ChromaDB)",
              fontsize=12, fontweight="bold", color=PRI)

fig.suptitle("Analyse Retrieval & Routing — Données Réelles ChromaDB PFE 2025",
             fontsize=12, fontweight="bold", color=TXT, y=1.02)
plt.tight_layout()
saved.append(savefig("04_score_distribution.png"))


# ─────────────────────────────────────────────────────────────────────────────
# EXPORT JSON
# ─────────────────────────────────────────────────────────────────────────────
report = {
    "meta": {
        "projet": "chatbot_doc_flask", "auteur": "Siwar BenSalah", "annee": "2025",
        "mode": "RÉEL — AgenticRAGPipeline + ChromaDB",
        "collection_size": pipeline.collection_size,
        "dataset_size": len(results),
        "seuils_empiriques": {
            "routing":       SEUIL_ROUTING,
            "hit_k":         SEUIL_HIT_K,
            "reformulation": SEUIL_REFORM,
            "justification": (
                "Fixés empiriquement après observation des distributions de scores "
                "de similarité cosinus sur le dataset Flask avec all-MiniLM-L6-v2 (384 dims). "
                "Seuil routing aligné sur la valeur définie dans agents.py RoutingAgent."
            )
        }
    },
    "agent1_routing": {
        "accuracy":            round(acc,     4),
        "precision":           round(prec,    4),
        "recall":              round(rec,     4),
        "f1_score":            round(f1,      4),
        "rejection_accuracy":  round(rej_acc, 4),
        "false_positive_rate": round(fpr,     4),
        "confusion_matrix": {
            "TN": int(cm[0, 0]), "FP": int(cm[0, 1]),
            "FN": int(cm[1, 0]), "TP": int(cm[1, 1]),
        },
    },
    "agent2_retrieval": {
        "avg_similarity_score":    round(avg_score,       4),
        "hit_at_k":                round(hit_k,           4),
        "mrr_rang_reel":           round(mrr,             4),
        "precision_at_k":          round(precision_at_k,  4),
        "seuil_hit_k_utilise":     SEUIL_HIT_K,
        "mrr_methode": (
            "Rang réel du premier chunk avec score >= seuil_hit_k dans la liste "
            "retournée par ChromaDB (triée par score décroissant). "
            "rr = 1/rang si trouvé, 0 sinon. MRR = moyenne des rr."
        ),
    },
    "agent3_reformulation": {
        "total_triggered":    int(reform_triggered),
        "trigger_rate":       round(reform_rate, 4),
        "total_reformulated": int(reformulated_n),
        "avg_score_after":    round(avg_score_after_reform, 4)
                              if len(reform_success_df) > 0 else None,
        "seuil_declenchement": SEUIL_REFORM,
    },
    "pipeline_global": {
        "avg_time_ms":  round(avg_ms, 2),
        "total_errors": errors_count,
    },
    "per_category": cats_df.to_dict(orient="records"),
    "predictions": [
        {k: v for k, v in asdict(r).items() if k != "sources"}
        for r in results
    ],
}
json_path = OUTPUT_DIR / "evaluation_report_reel.json"
json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2))
print(f"  💾 evaluation_report_reel.json")

# ── Résumé final ──────────────────────────────────────────────────────────────
print(f"\n{'═'*65}")
print(f"  ✅ ÉVALUATION RÉELLE TERMINÉE — thème académique clair")
print(f"  Dossier : {OUTPUT_DIR}/")
for f in sorted(OUTPUT_DIR.iterdir()):
    print(f"    {f.name:<50} {f.stat().st_size//1024:>4} KB")
print(f"{'═'*65}\n")