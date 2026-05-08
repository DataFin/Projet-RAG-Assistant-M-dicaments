"""
embedder.py
===========
Responsabilité : transformer les chunks en vecteurs numériques.
"""

import numpy as np
from sentence_transformers import SentenceTransformer
from pipeline.config import MODELE_EMBEDDING


class Embedder:
    """
    Spécialiste de l'embedding — transforme le texte en vecteurs.

    Usage :
        embedder = Embedder()
        vecteurs = embedder.encoder(["texte 1", "texte 2"])
    """

    def __init__(self, modele_nom: str = MODELE_EMBEDDING):
        """
        Constructeur — charge le modèle d'embedding.

        Args:
            modele_nom : nom du modèle HuggingFace
        """
        self.modele_nom = modele_nom  # MON nom de modèle
        print(f"  Chargement du modèle : {modele_nom}...")
        self.modele = SentenceTransformer(modele_nom)  # MON modèle
        print(f"  ✓ Modèle chargé.")

    def encoder(self, textes: list) -> np.ndarray:
        """
        Transforme une liste de textes en vecteurs numpy.
        Utilise le batch_size pour accélérer le traitement.

        Args:
            textes : liste de chaînes à encoder

        Returns:
            Tableau numpy shape (n_textes, 768) dtype float32
        """
        print(f"  Embedding de {len(textes)} chunks...")
        vecteurs = self.modele.encode(
            textes,
            batch_size=64,
            normalize_embeddings=True,
            show_progress_bar=True,
            convert_to_numpy=True,
        )
        print(f"  ✓ Embedding terminé — shape : {vecteurs.shape}")
        return vecteurs.astype(np.float32)

    def encoder_un(self, texte: str) -> np.ndarray:
        """
        Encode un seul texte (utilisé lors de la recherche).

        Args:
            texte : chaîne à encoder

        Returns:
            Vecteur numpy shape (768,) dtype float32
        """
        vecteur = self.modele.encode(
            texte,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return vecteur.astype(np.float32)