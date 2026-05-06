"""
=====================================================================
CHATBOT D'AIDE À LA DÉCISION CLINIQUE - INTERFACE UTILISATEUR
=====================================================================
- Pose des questions structurées à un professionnel de santé
- Génère un contexte patient structuré
- Envoie ce contexte au RAG existant via answer_question()
=====================================================================
"""

from typing import Dict
from rag import answer_contexte_clinique   # ← le RAG EXISTANT

import json

def extraire_recommandations(reponse_llm: str) -> list:
    """
    Extrait la liste des recommandations depuis la réponse JSON du LLM.
    """
    try:
        data = json.loads(reponse_llm)
        return data.get("recommandations", [])
    except json.JSONDecodeError:
        print("❌ Erreur : réponse LLM non parsable en JSON")
        return []
import csv
import os
from datetime import datetime


def exporter_recommandations_csv(patient_id: str, recommandations: list):
    """
    Exporte les recommandations médicamenteuses dans un fichier CSV.
    """
    if not recommandations:
        print("⚠️ Aucune recommandation à exporter.")
        return

    os.makedirs("exports", exist_ok=True)

    nom_fichier = (
        f"exports/recommandations_{patient_id}_"
        f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    )

    champs = [
        "medicament",
        "indication",
        "contre_indications",
        "interactions",
        "points_vigilance",
        "sources"
    ]

    with open(nom_fichier, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=champs, delimiter=";")
        writer.writeheader()
        for rec in recommandations:
            writer.writerow(rec)

    print(f"\n✅ Recommandations exportées dans : {nom_fichier}")

class ChatbotClinique:
    """
    Interface conversationnelle pour recueillir les informations patient
    et interroger le système RAG existant.
    """

    def afficher_banniere_accueil(self):
        print("\n" + "=" * 75)
        print("🏥 CHATBOT D'AIDE À LA DÉCISION CLINIQUE")
        print("   Destiné aux professionnels de santé uniquement")
        print("=" * 75)
        print("\n⚠️ AVERTISSEMENT :")
        print("Cet outil s'appuie sur les notices officielles BDPM (ANSM).")
        print("Il ne remplace PAS un avis médical.")
        print("=" * 75)

    def poser_question(self, question: str) -> str:
        while True:
            reponse = input(f"\n❓ {question}\n> ").strip()
            if reponse:
                return reponse
            print("⚠️ Réponse obligatoire.")

    def recueillir_donnees_patient(self) -> Dict[str, str]:
        donnees = {}

        print("\n📋 INFORMATIONS PATIENT")

        donnees["identifiant"] = self.poser_question(
            "Identifiant patient (initiales / pseudo) :"
        )

        donnees["age"] = self.poser_question(
            "Âge du patient (en années) :"
        )

        donnees["sexe"] = self.poser_question(
            "Sexe du patient (M/F/Autre) :"
        )

        print("\n🩺 SYMPTÔMES")
        donnees["symptomes"] = self.poser_question(
            "Décrivez les symptômes principaux :"
        )

        print("\n📚 ANTÉCÉDENTS")
        donnees["antecedents"] = self.poser_question(
            "Antécédents médicaux (ou 'Aucun') :"
        )

        print("\n💊 TRAITEMENTS EN COURS")
        donnees["traitements"] = self.poser_question(
            "Traitements actuels (ou 'Aucun') :"
        )

        print("\n⚠️ ALLERGIES")
        donnees["allergies"] = self.poser_question(
            "Allergies ou intolérances connues (ou 'Aucune') :"
        )

        print("\n📌 AUTRES INFORMATIONS")
        donnees["autres"] = self.poser_question(
            "Autres informations pertinentes (poids, grossesse, etc.) :"
        )

        return donnees

    def construire_contexte_patient(self, donnees: Dict[str, str]) -> str:
        """
        ⚠️ CE TEXTE EST ENVOYÉ DIRECTEMENT AU RAG
        """

        return f"""
CONTEXTE CLINIQUE – PROFESSIONNEL DE SANTÉ
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Patient : {donnees['identifiant']}
Âge : {donnees['age']} ans
Sexe : {donnees['sexe']}

SYMPTÔMES :
{donnees['symptomes']}

ANTÉCÉDENTS :
{donnees['antecedents']}

TRAITEMENTS ACTUELS :
{donnees['traitements']}

ALLERGIES / INTOLÉRANCES :
{donnees['allergies']}

AUTRES INFORMATIONS CLINIQUES :
{donnees['autres']}

Analyse ce contexte et propose des médicaments potentiellement pertinents
UNIQUEMENT à partir des notices officielles BDPM fournies dans le contexte.
"""

    def lancer(self):
        self.afficher_banniere_accueil()

        while True:
            donnees = self.recueillir_donnees_patient()
            contexte_patient = self.construire_contexte_patient(donnees)

            print("\n🔍 Analyse via la base BDPM...\n")
            # 🔥 APPEL DIRECT À TON RAG EXISTANT
            reponse = answer_contexte_clinique(contexte_patient)

            # ✅ Extraction des recommandations depuis le JSON du LLM
            recommandations = extraire_recommandations(reponse)

            # ✅ Export CSV
            exporter_recommandations_csv(
            donnees["identifiant"],
            recommandations
)           
            continuer = input("\nAnalyser un autre patient ? (oui/non) : ").lower()
            if continuer not in ["oui", "o"]:
                print("\n👋 Fin de session.")
                break



if __name__ == "__main__":
    print("✅ DANS __main__")
    chatbot = ChatbotClinique()
    chatbot.lancer()



