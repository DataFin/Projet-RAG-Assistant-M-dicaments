"""
chunker.py
==========
Responsabilité : découper les documents en chunks.
"""

from pipeline.config import TAILLE_CHUNK, OVERLAP_CHUNK


class Chunker:
    """
    Spécialiste du découpage de texte en chunks.

    Usage :
        chunker = Chunker()
        chunks  = chunker.chunker_documents(documents)
    """

    def __init__(self, taille_max: int = TAILLE_CHUNK, overlap: int = OVERLAP_CHUNK):
        self.taille_max = taille_max  # MA taille max
        self.overlap    = overlap     # MON overlap

    def chunker_documents(self, documents: list) -> list:
        """
        Découpe une liste de documents en chunks avec métadonnées.
        """
        chunks = []
        idx = 0

        for doc in documents:
            morceaux = self._chunker_texte(doc["texte"])
            for morceau in morceaux:
                chunks.append({
                    "id": f"chunk_{idx:04d}",
                    "contenu": morceau,
                    "metadata": {
                        "medicament": doc["medicament"],
                        "section":    doc["section"],
                        "source":     doc.get("source", "ANSM"),
                    },
                })
                idx += 1

        print(f"  ✓ {len(chunks)} chunks générés.")
        return chunks

    def _chunker_texte(self, texte: str) -> list:
        """
        Découpe un texte en morceaux avec overlap.
        """
        if len(texte) <= self.taille_max:
            return [texte.strip()]

        chunks = []
        debut = 0

        while debut < len(texte):
            fin = debut + self.taille_max

            if fin >= len(texte):
                chunk = texte[debut:].strip()
                if chunk:
                    chunks.append(chunk)
                break

            coupure = texte.rfind("\n\n", debut, fin)
            if coupure == -1:
                coupure = texte.rfind(".\n", debut, fin)
            if coupure == -1:
                coupure = texte.rfind(". ", debut, fin)
            if coupure == -1:
                coupure = texte.rfind(" ", debut, fin)
            if coupure == -1:
                coupure = fin

            chunk = texte[debut:coupure + 1].strip()
            if chunk:
                chunks.append(chunk)

            debut = max(debut + 1, coupure + 1 - self.overlap)

        return chunks