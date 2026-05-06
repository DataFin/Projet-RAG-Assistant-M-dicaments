"""
indexation.py
=============
Phase 1 du pipeline RAG – Médicaments
Récupère les données, découpe en chunks, génère les embeddings
et persiste la base vectorielle FAISS sur disque.

Usage :
    python indexation.py
"""

import os
import json
import time
import requests
import numpy as np
import faiss
import pandas as pd
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from bs4 import BeautifulSoup

load_dotenv()

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
MODELE_EMBEDDING = "paraphrase-multilingual-mpnet-base-v2"
TAILLE_CHUNK = 600
OVERLAP_CHUNK = 80
CHEMIN_INDEX = "data/faiss_index.bin"
CHEMIN_META  = "data/chunks_meta.json"

MEDICAMENTS_CORPUS = [
    "doliprane", "dafalgan", "efferalgan",
    "ibuprofène", "advil", "nurofen",
    "aspirin", "aspégic",
    "amoxicilline", "augmentin",
    "smecta", "imodium",
    "ventoline",
    "oméprazole", "inexium",
    "metformine", "glucophage",
]

# ─────────────────────────────────────────────
# ÉTAPE 0 – TÉLÉCHARGEMENT / EXTRACTION ZIP
# ─────────────────────────────────────────────

def telecharger_bdpm(dossier: str = "data") -> str:
    """
    Extrait le fichier ZIP BDPM et retourne le chemin du fichier CSV.
    """
    import zipfile

    chemin_zip = os.path.join(dossier, "cis-rcp.zip")
    chemin_csv = os.path.join(dossier, "CIS_RCP.csv")

    if os.path.exists(chemin_csv):
        print(f"  ✓ Fichier déjà extrait → {chemin_csv}")
        return chemin_csv

    if os.path.exists(chemin_zip):
        print(f"  Extraction de {chemin_zip}...")
        with zipfile.ZipFile(chemin_zip, 'r') as z:
            print(f"  Fichiers dans le ZIP : {z.namelist()}")
            z.extractall(dossier)
        for nom in os.listdir(dossier):
            if nom.endswith('.csv') and 'RCP' in nom.upper():
                chemin_csv = os.path.join(dossier, nom)
                print(f"  ✓ Fichier extrait → {chemin_csv}")
                return chemin_csv

    raise FileNotFoundError(
        "Placez le fichier cis-rcp.zip dans le dossier data/ et relancez le script."
    )


# ─────────────────────────────────────────────
# ÉTAPE 1 – NETTOYAGE ET CHARGEMENT CSV
# ─────────────────────────────────────────────

def nettoyer_html(html: str) -> str:
    """Supprime les balises HTML avec BeautifulSoup."""
    import re
    soup = BeautifulSoup(html, 'html.parser')
    texte = soup.get_text(separator=' ')
    texte = re.sub(r'\s+', ' ', texte)
    return texte.strip()


def nettoyer_nom(nom: str) -> str:
    """Nettoie le nom du médicament."""
    import re
    nom = nom.strip()
    nom = re.sub(r'\s+', ' ', nom)
    return nom


def charger_csv(chemin: str, limite: int = 50) -> list[dict]:
    """
    Charge les notices médicales depuis le CSV BDPM.
    Extrait chaque section clinique séparément pour un chunking sémantique.
    """
    import re

    SECTIONS_CIBLES = {
        "4.1": "Indications thérapeutiques",
        "4.2": "Posologie",
        "4.3": "Contre-indications",
        "4.4": "Mises en garde",
        "4.5": "Interactions médicamenteuses",
        "4.8": "Effets indésirables",
        "4.9": "Surdosage",
    }

    df = pd.read_csv(chemin, encoding='latin-1', sep='\t')
    documents = []

    for _, row in df.head(limite).iterrows():
        html = str(row['RCP_html'])
        html = html.encode('latin-1').decode('utf-8', errors='replace')
        soup = BeautifulSoup(html, 'html.parser')

        nom_tag = (soup.find(class_='AmmDenomination') or
                   soup.find(class_='AmmCorpsTexteGras'))
        nom = nettoyer_nom(nom_tag.get_text()) if nom_tag else f"Médicament {row['Code_CIS']}"

        titres = soup.find_all(class_='AmmAnnexeTitre2')
        for titre in titres:
            titre_texte = titre.get_text().strip()

            section_nom = None
            for code, nom_section in SECTIONS_CIBLES.items():
                if titre_texte.startswith(code):
                    section_nom = nom_section
                    break

            if not section_nom:
                continue

            contenu = []
            for element in titre.find_next_siblings():
                if 'AmmAnnexeTitre2' in element.get('class', []):
                    break
                texte = element.get_text(separator=' ').strip()
                if texte:
                    contenu.append(texte)

            texte_section = ' '.join(contenu)
            texte_section = re.sub(r'\s+', ' ', texte_section).strip()

            if len(texte_section) > 50:
                documents.append({
                    "medicament": nom,
                    "section": section_nom,
                    "texte": f"{section_nom} du médicament {nom} : {texte_section}",
                })

    return documents


# ─────────────────────────────────────────────
# DONNÉES DE SECOURS (fallback local)
# ─────────────────────────────────────────────

def corpus_fallback() -> list[dict]:
    """
    Corpus local de médicaments courants avec sections détaillées.
    Le nom du médicament et la section sont inclus dans le texte
    pour améliorer la pertinence de la recherche vectorielle.
    """
    return [
        # ── DOLIPRANE / PARACÉTAMOL ──────────────────────────────────────
        {
            "medicament": "Doliprane (paracétamol)",
            "section": "Indications",
            "texte": (
                "Indications du Doliprane (paracétamol) : "
                "Le Doliprane est indiqué pour le traitement symptomatique des douleurs "
                "d'intensité légère à modérée et/ou des états fébriles. Il est utilisé contre les maux de tête, "
                "les douleurs dentaires, les douleurs musculaires, les états grippaux et la fièvre."
            ),
        },
        {
            "medicament": "Doliprane (paracétamol)",
            "section": "Posologie",
            "texte": (
                "Posologie du Doliprane (paracétamol) : "
                "Adulte : 500 mg à 1 g par prise, 3 à 4 fois par jour si nécessaire. "
                "Ne pas dépasser 4 g par jour (soit 8 comprimés de 500 mg ou 4 comprimés de 1 g). "
                "Respecter un intervalle d'au moins 4 heures entre chaque prise. "
                "En cas d'insuffisance rénale ou hépatique, espacer les prises à minimum 8 heures."
            ),
        },
        {
            "medicament": "Doliprane (paracétamol)",
            "section": "Contre-indications",
            "texte": (
                "Contre-indications du Doliprane (paracétamol) : "
                "Contre-indiqué en cas d'allergie au paracétamol ou à l'un des excipients. "
                "Contre-indiqué en cas d'insuffisance hépatique sévère. "
                "Déconseillé en cas de consommation régulière et importante d'alcool. "
                "Attention lors d'association avec d'autres médicaments contenant du paracétamol "
                "(risque de surdosage)."
            ),
        },
        {
            "medicament": "Doliprane (paracétamol)",
            "section": "Effets indésirables",
            "texte": (
                "Effets indésirables du Doliprane (paracétamol) : "
                "Rares à très rares : réactions allergiques (éruption cutanée, urticaire, choc anaphylactique). "
                "Exceptionnels : atteintes hépatiques graves en cas de surdosage. "
                "Très rare : thrombopénie, leucopénie. "
                "Le paracétamol est bien toléré aux doses thérapeutiques recommandées."
            ),
        },
        {
            "medicament": "Doliprane (paracétamol)",
            "section": "Interactions médicamenteuses",
            "texte": (
                "Interactions médicamenteuses du Doliprane (paracétamol) : "
                "Association déconseillée avec : les anticoagulants oraux (warfarine) en cas de prise régulière "
                "et prolongée, l'alcool (risque d'atteinte hépatique). "
                "Précautions avec : les inducteurs enzymatiques (rifampicine, carbamazépine) qui réduisent "
                "l'efficacité du paracétamol et augmentent le risque de toxicité hépatique."
            ),
        },
        # ── IBUPROFÈNE / ADVIL / NUROFEN ────────────────────────────────
        {
            "medicament": "Ibuprofène (Advil, Nurofen)",
            "section": "Indications",
            "texte": (
                "Indications de l'ibuprofène (Advil, Nurofen) : "
                "L'ibuprofène est un anti-inflammatoire non stéroïdien (AINS) indiqué dans : "
                "les douleurs légères à modérées (maux de tête, douleurs dentaires, douleurs menstruelles, "
                "douleurs musculaires et articulaires), les états fébriles, et les douleurs rhumatismales."
            ),
        },
        {
            "medicament": "Ibuprofène (Advil, Nurofen)",
            "section": "Posologie",
            "texte": (
                "Posologie de l'ibuprofène (Advil, Nurofen) : "
                "Adulte et enfant > 40 kg : 200 à 400 mg par prise, 3 fois par jour. "
                "Dose maximale : 1 200 mg par jour en automédication. "
                "Prendre pendant les repas pour limiter les troubles digestifs. "
                "Utiliser la dose minimale efficace pendant la durée la plus courte possible."
            ),
        },
        {
            "medicament": "Ibuprofène (Advil, Nurofen)",
            "section": "Contre-indications",
            "texte": (
                "Contre-indications de l'ibuprofène (Advil, Nurofen) : "
                "Contre-indiqué en cas d'allergie aux AINS ou à l'aspirine (risque de réaction croisée). "
                "Contre-indiqué à partir du 6e mois de grossesse. "
                "Contre-indiqué en cas d'ulcère gastroduodénal actif, d'insuffisance rénale, hépatique "
                "ou cardiaque sévère. Déconseillé en cas d'antécédents d'ulcère ou de saignements digestifs."
            ),
        },
        {
            "medicament": "Ibuprofène (Advil, Nurofen)",
            "section": "Effets indésirables",
            "texte": (
                "Effets indésirables de l'ibuprofène (Advil, Nurofen) : "
                "Fréquents : troubles digestifs (nausées, vomissements, douleurs abdominales, diarrhées). "
                "Peu fréquents : maux de tête, vertiges, réactions allergiques cutanées. "
                "Rares : ulcère gastrique, hémorragie digestive (surtout en cas d'utilisation prolongée). "
                "Très rares : atteintes rénales, réactions cardiovasculaires (infarctus, AVC à fortes doses prolongées)."
            ),
        },
        {
            "medicament": "Ibuprofène (Advil, Nurofen)",
            "section": "Interactions médicamenteuses",
            "texte": (
                "Interactions médicamenteuses de l'ibuprofène (Advil, Nurofen) : "
                "Ne pas associer à d'autres AINS ni à l'aspirine (risque de toxicité additive). "
                "Prudence avec les anticoagulants (risque hémorragique), les diurétiques et les IEC "
                "(risque d'insuffisance rénale). "
                "Peut réduire l'effet des antihypertenseurs. Augmente la toxicité du méthotrexate."
            ),
        },
        # ── ASPIRINE / ASPÉGIC ──────────────────────────────────────────
        {
            "medicament": "Aspirine (Aspégic)",
            "section": "Indications",
            "texte": (
                "Indications de l'aspirine (Aspégic) : "
                "L'aspirine (acide acétylsalicylique) est utilisée comme antalgique, antipyrétique et "
                "anti-inflammatoire à doses élevées (500 mg - 1 g). À faibles doses (75-325 mg), elle est "
                "indiquée en prévention des accidents cardiovasculaires (antiagrégant plaquettaire)."
            ),
        },
        {
            "medicament": "Aspirine (Aspégic)",
            "section": "Posologie",
            "texte": (
                "Posologie de l'aspirine (Aspégic) : "
                "Antalgique/antipyrétique adulte : 500 mg à 1 g par prise, toutes les 4 à 6 heures. "
                "Maximum 3 g par jour. "
                "Antiagrégant plaquettaire : 75 à 160 mg par jour (sur prescription médicale). "
                "À prendre pendant les repas avec un grand verre d'eau."
            ),
        },
        {
            "medicament": "Aspirine (Aspégic)",
            "section": "Contre-indications",
            "texte": (
                "Contre-indications de l'aspirine (Aspégic) : "
                "Contre-indiqué chez les enfants et adolescents de moins de 16 ans atteints d'un syndrome "
                "viral (risque de syndrome de Reye). "
                "Contre-indiqué en cas d'allergie aux salicylés ou AINS. "
                "Contre-indiqué en cas d'ulcère gastroduodénal, de troubles de la coagulation, "
                "d'insuffisance rénale sévère ou hépatique sévère."
            ),
        },
        {
            "medicament": "Aspirine (Aspégic)",
            "section": "Effets indésirables",
            "texte": (
                "Effets indésirables de l'aspirine (Aspégic) : "
                "Fréquents : troubles digestifs (nausées, douleurs épigastriques, risque d'ulcère). "
                "Risque hémorragique (saignements digestifs, cutanés). "
                "Rares : réactions allergiques (urticaire, bronchospasme chez les asthmatiques sensibles). "
                "Acouphènes et vertiges en cas de surdosage (intoxication salicylée)."
            ),
        },
        {
            "medicament": "Aspirine (Aspégic)",
            "section": "Interactions médicamenteuses",
            "texte": (
                "Interactions médicamenteuses de l'aspirine (Aspégic) : "
                "Contre-indiqué avec le méthotrexate à fortes doses. "
                "Déconseillé avec les anticoagulants oraux (risque hémorragique majeur). "
                "Déconseillé avec les autres AINS, l'ibuprofène notamment. "
                "Aspirine et ibuprofène pris simultanément peuvent annuler l'effet antiagrégant de l'aspirine."
            ),
        },
        # ── AMOXICILLINE / AUGMENTIN ─────────────────────────────────────
        {
            "medicament": "Amoxicilline (Augmentin)",
            "section": "Indications",
            "texte": (
                "Indications de l'amoxicilline (Augmentin) : "
                "L'amoxicilline est un antibiotique de la famille des pénicillines, indiqué dans le traitement "
                "des infections bactériennes : angines streptococciques, sinusites, otites moyennes aiguës, "
                "bronchites, pneumonies, infections urinaires, infections cutanées. "
                "L'Augmentin (amoxicilline + acide clavulanique) couvre un spectre plus large."
            ),
        },
        {
            "medicament": "Amoxicilline (Augmentin)",
            "section": "Posologie",
            "texte": (
                "Posologie de l'amoxicilline (Augmentin) : "
                "Adulte : 1 g deux à trois fois par jour selon la gravité de l'infection. "
                "Durée de traitement : généralement 5 à 10 jours. Ne jamais interrompre le traitement "
                "avant la fin (risque de résistance). "
                "Augmentin 1 g/125 mg : 1 comprimé 2 à 3 fois par jour. "
                "Toujours prendre sur prescription médicale."
            ),
        },
        {
            "medicament": "Amoxicilline (Augmentin)",
            "section": "Contre-indications",
            "texte": (
                "Contre-indications de l'amoxicilline (Augmentin) : "
                "Contre-indiqué en cas d'allergie aux pénicillines ou aux céphalosporines (allergie croisée). "
                "Contre-indiqué en cas de mononucléose infectieuse (risque de rash cutané). "
                "Prudence en cas d'insuffisance rénale (adaptation posologique nécessaire)."
            ),
        },
        {
            "medicament": "Amoxicilline (Augmentin)",
            "section": "Effets indésirables",
            "texte": (
                "Effets indésirables de l'amoxicilline (Augmentin) : "
                "Fréquents : troubles digestifs (diarrhées, nausées, vomissements), surtout avec l'Augmentin. "
                "Peu fréquents : réactions allergiques cutanées (rash, urticaire). "
                "Rares mais graves : choc anaphylactique (allergie sévère), colite pseudomembraneuse. "
                "Possible déséquilibre de la flore intestinale (candidose)."
            ),
        },
        # ── SMECTA / IMODIUM ────────────────────────────────────────────
        {
            "medicament": "Smecta (diosmectite)",
            "section": "Indications et posologie",
            "texte": (
                "Indications et posologie du Smecta (diosmectite) : "
                "Le Smecta est un antidiarrhéique d'action locale indiqué dans le traitement "
                "symptomatique des diarrhées aiguës et chroniques chez l'adulte et l'enfant. "
                "Adulte : 3 sachets par jour, dilués dans de l'eau. "
                "En cas de diarrhée aiguë, la dose peut être doublée en début de traitement."
            ),
        },
        {
            "medicament": "Imodium (lopéramide)",
            "section": "Indications et posologie",
            "texte": (
                "Indications et posologie de l'Imodium (lopéramide) : "
                "L'Imodium est un antidiarrhéique d'action centrale indiqué dans les diarrhées "
                "aiguës et chroniques. Il ralentit le transit intestinal. "
                "Adulte : 2 mg (1 gélule) après chaque selle liquide, sans dépasser 8 mg par jour. "
                "Contre-indiqué dans les diarrhées sanglantes et les infections intestinales bactériennes. "
                "Ne pas utiliser plus de 48 heures sans avis médical."
            ),
        },
        # ── VENTOLINE ────────────────────────────────────────────────────
        {
            "medicament": "Ventoline (salbutamol)",
            "section": "Indications",
            "texte": (
                "Indications de la Ventoline (salbutamol) : "
                "La Ventoline est un bronchodilatateur de courte durée d'action (bêta-2 agoniste) "
                "indiqué dans le traitement et la prévention des crises d'asthme et des bronchospasmes "
                "associés à la broncho-pneumopathie chronique obstructive (BPCO)."
            ),
        },
        {
            "medicament": "Ventoline (salbutamol)",
            "section": "Posologie et effets indésirables",
            "texte": (
                "Posologie et effets indésirables de la Ventoline (salbutamol) : "
                "En cas de crise : 1 à 2 bouffées (100 à 200 microgrammes). "
                "Peut être répété après 5 minutes si nécessaire. "
                "Ne pas dépasser 8 bouffées par jour sans avis médical. "
                "Effets indésirables : tremblements des mains (fréquents), tachycardie, palpitations, "
                "céphalées. Rare : hypokaliémie en cas de doses élevées."
            ),
        },
        # ── OMÉPRAZOLE / INEXIUM ─────────────────────────────────────────
        {
            "medicament": "Oméprazole (Inexium / ésoméprazole)",
            "section": "Indications",
            "texte": (
                "Indications de l'oméprazole (Inexium / ésoméprazole) : "
                "L'oméprazole et l'ésoméprazole (Inexium) sont des inhibiteurs de la pompe à protons (IPP). "
                "Indiqués dans : le reflux gastro-oesophagien (RGO), l'ulcère gastroduodénal, "
                "l'éradication d'Helicobacter pylori (en association), la gastroprotection lors de prise d'AINS."
            ),
        },
        {
            "medicament": "Oméprazole (Inexium / ésoméprazole)",
            "section": "Posologie et effets indésirables",
            "texte": (
                "Posologie et effets indésirables de l'oméprazole (Inexium / ésoméprazole) : "
                "Posologie habituelle : 20 à 40 mg par jour, le matin à jeun. "
                "Effets indésirables fréquents : maux de tête, diarrhées, nausées, constipation, flatulences. "
                "Prise prolongée : risque de carence en magnésium, en vitamine B12. "
                "Interactions : réduit l'absorption de certains médicaments (clopidogrel, antifongiques)."
            ),
        },
        # ── METFORMINE / GLUCOPHAGE ──────────────────────────────────────
        {
            "medicament": "Metformine (Glucophage)",
            "section": "Indications",
            "texte": (
                "Indications de la metformine (Glucophage) : "
                "La metformine est un antidiabétique oral de la classe des biguanides, "
                "indiqué dans le traitement du diabète de type 2, en première intention, "
                "notamment chez les patients en surpoids. "
                "Elle agit en diminuant la production hépatique de glucose et en améliorant "
                "la sensibilité à l'insuline."
            ),
        },
        {
            "medicament": "Metformine (Glucophage)",
            "section": "Posologie et effets indésirables",
            "texte": (
                "Posologie et effets indésirables de la metformine (Glucophage) : "
                "Débuter à 500 mg 1 à 2 fois par jour pendant les repas, augmenter progressivement. "
                "Dose maximale : 3 g par jour. "
                "Effets indésirables : troubles digestifs très fréquents en début de traitement "
                "(nausées, diarrhées, douleurs abdominales). "
                "Risque rare mais grave : acidose lactique (surtout en cas d'insuffisance rénale). "
                "Contre-indiqué en cas d'insuffisance rénale sévère (DFG < 30 mL/min)."
            ),
        },
    ]


# ─────────────────────────────────────────────
# ÉTAPE 2 – CHUNKING
# ─────────────────────────────────────────────

def chunker(texte: str, taille_max: int = TAILLE_CHUNK, overlap: int = OVERLAP_CHUNK) -> list[str]:
    """
    Découpe un texte en chunks avec chevauchement.
    Priorité : paragraphes > phrases > espaces.
    """
    if len(texte) <= taille_max:
        return [texte.strip()]

    chunks = []
    debut = 0

    while debut < len(texte):
        fin = debut + taille_max

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

        debut = max(debut + 1, coupure + 1 - overlap)

    return chunks


# ─────────────────────────────────────────────
# ÉTAPE 3 – EMBEDDING
# ─────────────────────────────────────────────

def embedder_chunks(chunks: list[str], modele: SentenceTransformer) -> np.ndarray:
    """
    Transforme une liste de chunks en vecteurs via sentence-transformers.
    """
    vecteurs = []
    total = len(chunks)
    for i, chunk in enumerate(chunks):
        if i % 10 == 0:
            print(f"  Embedding : {i}/{total} chunks traités...")
        vecteur = modele.encode(chunk, convert_to_numpy=True)
        vecteurs.append(vecteur)
    print(f"  Embedding : {total}/{total} chunks traités. ✓")
    return np.array(vecteurs, dtype=np.float32)


# ─────────────────────────────────────────────
# ÉTAPE 4 – FAISS : CRÉATION ET PERSISTANCE
# ─────────────────────────────────────────────

def creer_index_faiss(vecteurs: np.ndarray) -> faiss.Index:
    """
    Crée un index FAISS avec similarité cosinus (IndexFlatIP + normalisation L2).
    """
    dimension = vecteurs.shape[1]
    faiss.normalize_L2(vecteurs)
    index = faiss.IndexFlatIP(dimension)
    index.add(vecteurs)
    print(f"  Index FAISS créé : {index.ntotal} vecteurs, dimension {dimension}.")
    return index


def sauvegarder_index(index: faiss.Index, chunks_avec_meta: list[dict], chemin_index: str, chemin_meta: str):
    """
    Persiste l'index FAISS et les métadonnées sur disque.
    """
    os.makedirs(os.path.dirname(chemin_index), exist_ok=True)
    faiss.write_index(index, chemin_index)
    with open(chemin_meta, "w", encoding="utf-8") as f:
        json.dump(chunks_avec_meta, f, ensure_ascii=False, indent=2)
    print(f"  Index sauvegardé → {chemin_index}")
    print(f"  Métadonnées sauvegardées → {chemin_meta}")


# ─────────────────────────────────────────────
# PIPELINE PRINCIPAL
# ─────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  PIPELINE D'INDEXATION – RAG Médicaments")
    print("=" * 60)

    # ── 1. Construction du corpus ───────────────────────────────
    print("\n[1/4] Construction du corpus...")
    chemin_csv = telecharger_bdpm()
    documents_bruts = charger_csv(chemin_csv, limite=50)
    documents_bruts += corpus_fallback()
    print(f"  {len(documents_bruts)} sections chargées au total (CSV + corpus manuel).")

    # ── 2. Chunking ────────────────────────────────────────────
    print("\n[2/4] Chunking des documents...")
    chunks_avec_meta = []
    idx = 0
    for doc in documents_bruts:
        morceaux = chunker(doc["texte"])
        for morceau in morceaux:
            chunks_avec_meta.append({
                "id": f"chunk_{idx:04d}",
                "contenu": morceau,
                "metadata": {
                    "medicament": doc["medicament"],
                    "section": doc["section"],
                    "source": "Notice ANSM / corpus pédagogique",
                },
            })
            idx += 1
    print(f"  {len(chunks_avec_meta)} chunks générés.")

    # ── 3. Embedding ───────────────────────────────────────────
    print(f"\n[3/4] Génération des embeddings avec '{MODELE_EMBEDDING}'...")
    t0 = time.time()
    modele = SentenceTransformer(MODELE_EMBEDDING)
    textes = [c["contenu"] for c in chunks_avec_meta]
    vecteurs = embedder_chunks(textes, modele)
    print(f"  Temps d'embedding : {time.time() - t0:.1f} s")

    # ── 4. Index FAISS + persistance ──────────────────────────
    print("\n[4/4] Création et sauvegarde de la base FAISS...")
    index = creer_index_faiss(vecteurs)
    sauvegarder_index(index, chunks_avec_meta, CHEMIN_INDEX, CHEMIN_META)

    print("\n✅  Indexation terminée avec succès !")
    print(f"   Base prête : {len(chunks_avec_meta)} chunks indexés.")
    print("   Lancez maintenant : python rag.py\n")


if __name__ == "__main__":
    main()
