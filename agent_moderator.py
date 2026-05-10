"""
agent_moderator.py

Responsabilité :
- Détecter les tentatives de prompt injection ou de contournement
- Centraliser la logique de modération
"""

import re


class AgentModerator:
    """
    Modérateur simple pour détecter les prompt injections évidentes.
    """

    def __init__(self, client=None):
        # Le client LLM est optionnel (extensible plus tard)
        self.client = client

        # Règles simples et efficaces
        self.patterns = [
            r"ignore .*instructions",
            r"oublie .*contexte",
            r"oublie .*prompt",
            r"system prompt",
            r"tu n'es plus",
            r"réponds toujours",
            r"fais semblant",
            r"prompt injection",
        ]

    def moderate(self, texte: str) -> dict:
        """
        Analyse une requête utilisateur.

        Returns:
            dict : {
                "is_prompt_injection": bool,
                "reason": str
            }
        """
        texte_lower = texte.lower()

        for pattern in self.patterns:
            if re.search(pattern, texte_lower):
                return {
                    "is_prompt_injection": True,
                    "reason": f"Pattern détecté : {pattern}"
                }

        return {
            "is_prompt_injection": False,
            "reason": "RAS"
        }