"""
vector_store.py
===============
Responsabilité : stocker et rechercher les chunks vectorisés.
Utilise ChromaDB — inspiré du code de l'enseignant.

Idempotente : charge si existe, crée sinon.
"""

import os
import chromadb
from pipeline.config import MODELE_EMBEDDING, K_RESULTATS
from pipeline.embedder import Embedder


class VectorDB:
    """
    Spécialiste de la base vectorielle — inspiré du prof.
    Idempotente : charge si existe, crée sinon.

    Usage :
        # Première fois → crée la base
        db = VectorDB("data/chroma_db", chunks=chunks)

        # Fois suivantes → charge automatiquement
        db = VectorDB("data/chroma_db")

        # Recherche
        resultats = db.rechercher("effets du doliprane")
    """

    def __init__(self, chemin_db: str = "data/chroma_db", chunks: list = None):
        """
        Constructeur — comme le prof :
        - Si la base existe → on charge
        - Si chunks fournis → on crée
        - Sinon → erreur
        """
        if os.path.exists(chemin_db):
            self._charger(chemin_db)
        elif chunks:
            self._creer(chemin_db, chunks)
        else:
            raise Exception(
                "Impossible d'initialiser VectorDB ! "
                "Donnez soit un chemin vers une base existante, "
                "soit une liste de chunks."
            )

    def _creer(self, chemin_db: str, chunks: list):
        """
        Crée la base vectorielle depuis les chunks.
        Ajoute par lots pour respecter la limite ChromaDB (max 5461).
        """
        print("  Création de la base vectorielle ChromaDB...")

        self.modele_nom = MODELE_EMBEDDING
        self.embedder   = Embedder(self.modele_nom)

        self.chroma = chromadb.PersistentClient(path=chemin_db)
        self.collection = self.chroma.get_or_create_collection(
            name="medicaments",
            metadata={"embedding_model": self.modele_nom}
        )

        textes    = [c["contenu"]  for c in chunks]
        ids       = [c["id"]       for c in chunks]
        metadatas = [c["metadata"] for c in chunks]

        embeddings = self.embedder.encoder(textes).tolist()

        # Ajouter par lots de 5000
        taille_lot = 5000
        total      = len(textes)

        for i in range(0, total, taille_lot):
            fin = min(i + taille_lot, total)
            print(f"  Ajout chunks {i} à {fin} sur {total}...")
            self.collection.add(
                ids=ids[i:fin],
                documents=textes[i:fin],
                embeddings=embeddings[i:fin],
                metadatas=metadatas[i:fin],
            )

        print(f"  ✓ {total} chunks indexés dans ChromaDB.")
        print(f"  ✓ Base sauvegardée → {chemin_db}")

    def _charger(self, chemin_db: str):
        """Charge une base vectorielle existante."""
        print("  Chargement de la base vectorielle ChromaDB...")

        self.chroma     = chromadb.PersistentClient(path=chemin_db)
        self.collection = self.chroma.get_collection("medicaments")

        self.modele_nom = self.collection.metadata.get(
            "embedding_model", MODELE_EMBEDDING
        )
        print(f"  ✓ Modèle embedding : {self.modele_nom}")

        self.embedder = Embedder(self.modele_nom)
        print(f"  ✓ Base chargée : {self.collection.count()} chunks.")

    def rechercher(self, question: str, k: int = K_RESULTATS) -> list:
        """
        Recherche les k chunks les plus pertinents pour une question.
        """
        embedding_question = self.embedder.encoder_un(question).tolist()

        resultats = self.collection.query(
            query_embeddings=[embedding_question],
            n_results=k,
        )

        chunks_trouves = []
        for i in range(len(resultats["documents"][0])):
            chunks_trouves.append({
                "contenu":  resultats["documents"][0][i],
                "metadata": resultats["metadatas"][0][i],
                "score":    1 - resultats["distances"][0][i],
            })

        return chunks_trouves
    
    def retrieve(self, question: str, n: int = K_RESULTATS):
        """
        Alias vers rechercher (API RAG standard).
        """
        chunks = self.rechercher(question, k=n)
        scores = [chunk["score"] for chunk in chunks]
        return chunks, scores