from rag import RAG
from pdf_export import generer_pdf_rapport
import re

def supprimer_tous_les_json(texte: str) -> str:
    if not texte:
        return ""

    texte = re.sub(r"\{[\s\S]*?\}", "", texte)
    texte = re.sub(r"```[a-zA-Z]*", "", texte)
    texte = texte.replace("```", "")

    return texte.strip()


def collecter_formulaire_patient() -> dict:
    print("\n📝 FORMULAIRE CLINIQUE")
    return {
        "identifiant": input("Identifiant patient : ").strip(),
        "age": input("Âge : ").strip(),
        "sexe": input("Sexe (M/F/Autre) : ").strip(),
        "symptomes": input("Symptômes : ").strip(),
        "antecedents": input("Antécédents médicaux : ").strip(),
        "traitements": input("Traitements en cours : ").strip(),
        "allergies": input("Allergies / intolérances : ").strip(),
        "autres": input("Autres informations cliniques : ").strip(),
    }


def construire_contexte_patient(data: dict) -> str:
    return f"""
CONTEXTE CLINIQUE – PROFESSIONNEL DE SANTÉ
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Patient : {data['identifiant']}
Âge : {data['age']} ans
Sexe : {data['sexe']}

SYMPTÔMES :
{data['symptomes']}

ANTÉCÉDENTS :
{data['antecedents']}

TRAITEMENTS ACTUELS :
{data['traitements']}

ALLERGIES / INTOLÉRANCES :
{data['allergies']}

AUTRES INFORMATIONS :
{data['autres']}
"""


def main():
    data_patient = collecter_formulaire_patient()
    contexte = construire_contexte_patient(data_patient)

    rag = RAG(vector_db_path="data/chroma_db")
    print("\n🔍 Analyse via la base BDPM...")

    reponse_rag = rag.answer(contexte)

    texte_pour_pdf = supprimer_tous_les_json(reponse_rag)

    generer_pdf_rapport(
        patient_id=data_patient["identifiant"],
        contexte_patient=contexte,
        reponse_rag=texte_pour_pdf
    )

    print("\n✅ Rapport PDF généré avec succès.")


if __name__ == "__main__":
    main()