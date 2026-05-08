"""
indexation.py
=============
Orchestrateur du pipeline RAG Médicaments.
Assemble DataLoader → Chunker → VectorDB.

Usage :
    python indexation.py
"""

import os
from pipeline.config import CHEMIN_DB
from pipeline.data_loader import DataLoader
from pipeline.chunker import Chunker
from pipeline.vector_store import VectorDB


def main():
    print("=" * 60)
    print("  PIPELINE D'INDEXATION – RAG Médicaments")
    print("=" * 60)

    # ── Idempotence : base déjà créée ? ────────────────────
    print("\n[0/3] Vérification de la base existante...")
    if os.path.exists(CHEMIN_DB):
        print(f"  ✓ Base déjà présente → {CHEMIN_DB}")
        print("\n✅ Aucune réindexation nécessaire.")
        print("   Lancez directement : python rag.py\n")
        return

    # ── Étape 1 : Chargement des données ───────────────────
    print("\n[1/3] Chargement des données...")
    loader = DataLoader(
        source="tous",
        chemin_csv="data/CIS_RCP.csv",
        limite=50,
    )
    documents = loader.charger()

    # ── Étape 2 : Chunking ─────────────────────────────────
    print("\n[2/3] Chunking des documents...")
    chunker = Chunker()
    chunks = chunker.chunker_documents(documents)

    # ── Étape 3 : Base vectorielle ─────────────────────────
    print("\n[3/3] Création de la base vectorielle...")
    db = VectorDB(CHEMIN_DB, chunks=chunks)

    print("\n✅ Indexation terminée avec succès !")
    print(f"   {len(chunks)} chunks indexés.")
    print("   Lancez maintenant : python rag.py\n")


if __name__ == "__main__":
    main()