"""
FORMULAIRE CLINIQUE - ENVOI AU RAG + EXPORT PDF
"""

from rag import answer_contexte_clinique
from pdf_export import generer_pdf_rapport
import json
import re

def extraire_recommandations(json_part: str) -> list:
    """
    Extraction robuste des recommandations depuis la sortie JSON du LLM.
    Ne fait jamais planter le pipeline.
    """

    if not json_part:
        return []

    # 🔹 Nettoyage : on garde uniquement le bloc { ... }
    match = re.search(r"\{[\s\S]*\}", json_part)
    if not match:
        print("⚠️ Aucun bloc JSON détecté")
        return []

    json_str = match.group(0)

    # 🔹 Normalisation des guillemets éventuels
    json_str = json_str.replace("“", '"').replace("”", '"')

    try:
        data = json.loads(json_str)
        recommandations = data.get("recommandations", [])

        if not isinstance(recommandations, list):
            print("⚠️ Format inattendu pour 'recommandations'")
            return []

        return recommandations

    except json.JSONDecodeError as e:
        print(f"⚠️ JSON invalide malgré nettoyage : {e}")
        return []
    
def separer_texte_et_json(reponse_llm: str) -> tuple[str, str]:
    """
    Sépare la réponse texte du LLM et le bloc JSON final.
    Retourne (texte_humain, json_str)
    """

    debut_json = reponse_llm.find("{")
    if debut_json == -1:
        # Pas de JSON trouvé
        return reponse_llm.strip(), ""

    texte = reponse_llm[:debut_json].strip()
    json_part = reponse_llm[debut_json:].strip()

    return texte, json_part


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

    print("\n🔍 Analyse via la base BDPM...")
    reponse_rag = answer_contexte_clinique(contexte)
    texte_humain, json_part = separer_texte_et_json(reponse_rag)

    # ✅ Génération du PDF (même si aucune reco trouvée)
    generer_pdf_rapport(
        patient_id=data_patient["identifiant"],
        contexte_patient=contexte,
        reponse_rag=texte_humain
    )
    recommandations = extraire_recommandations(json_part)
    print("\n✅ Rapport PDF généré avec succès.")


if __name__ == "__main__":
    main()