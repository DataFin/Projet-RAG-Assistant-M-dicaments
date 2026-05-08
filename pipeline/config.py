"""
config.py
=========
Toutes les constantes du projet RAG Médicaments.
Un seul endroit pour modifier les paramètres.
"""

# ── Modèle d'embedding ──────────────────────
MODELE_EMBEDDING = "paraphrase-multilingual-mpnet-base-v2"

# ── Chunking ────────────────────────────────
TAILLE_CHUNK  = 600   # caractères max par chunk
OVERLAP_CHUNK = 80    # chevauchement entre chunks

# ── Base vectorielle ────────────────────────
CHEMIN_DB = "data/chroma_db"   # dossier ChromaDB

# ── Recherche ───────────────────────────────
K_RESULTATS     = 4     # nombre de chunks retournés
SEUIL_CONFIANCE = 0.25  # score minimum acceptable

# ── LLM Groq ────────────────────────────────
MODELE_GROQ  = "llama-3.3-70b-versatile"
MAX_TOKENS   = 1000
TEMPERATURE  = 0.2

# ── Sections BDPM à indexer ─────────────────
SECTIONS_CIBLES = {
    "4.1": "Indications thérapeutiques",
    "4.2": "Posologie",
    "4.3": "Contre-indications",
    "4.4": "Mises en garde",
    "4.5": "Interactions médicamenteuses",
    "4.8": "Effets indésirables",
    "4.9": "Surdosage",
}