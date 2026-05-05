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
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

load_dotenv()

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
MODELE_EMBEDDING = "paraphrase-multilingual-mpnet-base-v2"  # multilingue, idéal pour le français
TAILLE_CHUNK = 600       # caractères max par chunk
OVERLAP_CHUNK = 80       # chevauchement entre chunks consécutifs
CHEMIN_INDEX = "data/faiss_index.bin"
CHEMIN_META  = "data/chunks_meta.json"

# Corpus : liste de médicaments courants à indexer
MEDICAMENTS_CORPUS = [
    "doliprane", "dafalgan", "efferalgan",     # paracétamol
    "ibuprofène", "advil", "nurofen",           # ibuprofène
    "aspirin", "aspégic",                       # aspirine
    "amoxicilline", "augmentin",                # antibiotiques
    "smecta", "imodium",                        # digestif
    "ventoline",                                # asthme
    "oméprazole", "inexium",                    # estomac
    "metformine", "glucophage",                 # diabète
]

# ─────────────────────────────────────────────
# ÉTAPE 1 – RÉCUPÉRATION DES DONNÉES VIA API
# ─────────────────────────────────────────────

def recuperer_medicament_api(nom: str) -> dict | None:
    """
    Interroge l'API publique de la BDPM pour un médicament donné.
    Retourne un dictionnaire avec les informations clés, ou None si introuvable.
    """
    url = "https://api.medicaments.gouv.fr/1.0/medicaments"
    try:
        response = requests.get(url, params={"denomination": nom}, timeout=10)
        response.raise_for_status()
        data = response.json()
        if data and isinstance(data, list) and len(data) > 0:
            return data[0]  # on prend le premier résultat
    except Exception as e:
        print(f"  [WARN] API indisponible pour '{nom}' : {e}")
    return None


def construire_texte_medicament(nom: str, data: dict) -> str:
    """
    Convertit les données JSON d'un médicament en texte libre structuré
    pour maximiser la pertinence lors de l'embedding.
    """
    lignes = [f"MÉDICAMENT : {nom.upper()}"]

    denomination = data.get("denomination", "")
    if denomination:
        lignes.append(f"Dénomination officielle : {denomination}")

    forme = data.get("formePharmaceutique", "")
    if forme:
        lignes.append(f"Forme pharmaceutique : {forme}")

    voie = data.get("voiesAdministration", "")
    if voie:
        lignes.append(f"Voie d'administration : {voie}")

    statut = data.get("statutAdministratifAmm", "")
    if statut:
        lignes.append(f"Statut AMM : {statut}")

    return "\n".join(lignes)


# ─────────────────────────────────────────────
# DONNÉES DE SECOURS (fallback local)
# ─────────────────────────────────────────────

def corpus_fallback() -> list[dict]:
    """
    Corpus local de secours utilisé si l'API est indisponible.
    Chaque entrée représente une section de notice médicale.
    Les sections reflètent la structure réelle d'une notice ANSM.
    """
    return [
        # ── DOLIPRANE / PARACÉTAMOL ──────────────────────────────────────
        {
            "medicament": "Doliprane (paracétamol)",
            "section": "Indications",
            "texte": (
                "Le Doliprane (paracétamol) est indiqué pour le traitement symptomatique des douleurs "
                "d'intensité légère à modérée et/ou des états fébriles. Il est utilisé contre les maux de tête, "
                "les douleurs dentaires, les douleurs musculaires, les états grippaux et la fièvre."
            ),
        },
        {
            "medicament": "Doliprane (paracétamol)",
            "section": "Posologie",
            "texte": (
                "Posologie adulte : 500 mg à 1 g par prise, 3 à 4 fois par jour si nécessaire. "
                "Ne pas dépasser 4 g par jour (soit 8 comprimés de 500 mg ou 4 comprimés de 1 g). "
                "Respecter un intervalle d'au moins 4 heures entre chaque prise. "
                "En cas d'insuffisance rénale ou hépatique, espacer les prises à minimum 8 heures."
            ),
        },
        {
            "medicament": "Doliprane (paracétamol)",
            "section": "Contre-indications",
            "texte": (
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
                "L'ibuprofène est un anti-inflammatoire non stéroïdien (AINS) indiqué dans : "
                "les douleurs légères à modérées (maux de tête, douleurs dentaires, douleurs menstruelles, "
                "douleurs musculaires et articulaires), les états fébriles, et les douleurs rhumatismales."
            ),
        },
        {
            "medicament": "Ibuprofène (Advil, Nurofen)",
            "section": "Posologie",
            "texte": (
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
                "L'aspirine (acide acétylsalicylique) est utilisée comme antalgique, antipyrétique et "
                "anti-inflammatoire à doses élevées (500 mg – 1 g). À faibles doses (75–325 mg), elle est "
                "indiquée en prévention des accidents cardiovasculaires (antiagrégant plaquettaire)."
            ),
        },
        {
            "medicament": "Aspirine (Aspégic)",
            "section": "Posologie",
            "texte": (
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
                "L'amoxicilline est un antibiotique de la famille des pénicillines, indiqué dans le traitement "
                "des infections bactériennes : angines streptococciques, sinusites, otites moyennes aiguës, "
                "bronchites, pneumonies, infections urinaires, infections cutanées. "
                "L'Augmentin (amoxicilline + acide clavulanique) couvre un spectre plus large incluant les "
                "bactéries productrices de bêtalactamases."
            ),
        },
        {
            "medicament": "Amoxicilline (Augmentin)",
            "section": "Posologie",
            "texte": (
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
                "Contre-indiqué en cas d'allergie aux pénicillines ou aux céphalosporines (allergie croisée). "
                "Contre-indiqué en cas de mononucléose infectieuse (risque de rash cutané). "
                "Prudence en cas d'insuffisance rénale (adaptation posologique nécessaire)."
            ),
        },
        {
            "medicament": "Amoxicilline (Augmentin)",
            "section": "Effets indésirables",
            "texte": (
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
                "Le Smecta (diosmectite) est un antidiarrhéique d'action locale indiqué dans le traitement "
                "symptomatique des diarrhées aiguës et chroniques chez l'adulte et l'enfant. "
                "Adulte : 3 sachets par jour, dilués dans de l'eau. "
                "En cas de diarrhée aiguë, la dose peut être doublée en début de traitement. "
                "Il agit en tapissant la muqueuse intestinale et en absorbant les toxines et agents infectieux."
            ),
        },
        {
            "medicament": "Imodium (lopéramide)",
            "section": "Indications, posologie et contre-indications",
            "texte": (
                "L'Imodium (lopéramide) est un antidiarrhéique d'action centrale indiqué dans les diarrhées "
                "aiguës et chroniques. Il ralentit le transit intestinal. "
                "Adulte : 2 mg (1 gélule) après chaque selle liquide, sans dépasser 8 mg par jour. "
                "Contre-indiqué dans les diarrhées sanglantes, les infections intestinales bactériennes "
                "prouvées, et chez l'enfant de moins de 2 ans. "
                "Ne pas utiliser plus de 48 heures sans avis médical."
            ),
        },
        # ── VENTOLINE ────────────────────────────────────────────────────
        {
            "medicament": "Ventoline (salbutamol)",
            "section": "Indications",
            "texte": (
                "La Ventoline (salbutamol) est un bronchodilatateur de courte durée d'action (bêta-2 agoniste) "
                "indiqué dans le traitement et la prévention des crises d'asthme et des bronchospasmes "
                "associés à la broncho-pneumopathie chronique obstructive (BPCO)."
            ),
        },
        {
            "medicament": "Ventoline (salbutamol)",
            "section": "Posologie et effets indésirables",
            "texte": (
                "En cas de crise : 1 à 2 bouffées (100 à 200 microgrammes). "
                "Peut être répété après 5 minutes si nécessaire. "
                "Ne pas dépasser 8 bouffées par jour sans avis médical. "
                "Effets indésirables : tremblements des mains (fréquents), tachycardie, palpitations, "
                "céphalées. Rare : hypokaliémie en cas de doses élevées. "
                "Si les crises surviennent plus de 3 fois par semaine, consulter un médecin."
            ),
        },
        # ── OMÉPRAZOLE / INEXIUM ─────────────────────────────────────────
        {
            "medicament": "Oméprazole (Inexium / ésoméprazole)",
            "section": "Indications",
            "texte": (
                "L'oméprazole et l'ésoméprazole (Inexium) sont des inhibiteurs de la pompe à protons (IPP). "
                "Indiqués dans : le reflux gastro-œsophagien (RGO), l'ulcère gastroduodénal, "
                "l'éradication d'Helicobacter pylori (en association), la gastroprotection lors de prise d'AINS."
            ),
        },
        {
            "medicament": "Oméprazole (Inexium / ésoméprazole)",
            "section": "Posologie, effets indésirables et interactions",
            "texte": (
                "Posologie habituelle : 20 à 40 mg par jour, le matin à jeun. "
                "Effets indésirables fréquents : maux de tête, diarrhées, nausées, constipation, flatulences. "
                "Prise prolongée : risque de carence en magnésium, en vitamine B12, "
                "et augmentation du risque de fractures osseuses. "
                "Interactions : réduit l'absorption de certains médicaments (clopidogrel, antifongiques). "
                "Ne pas arrêter brutalement un traitement prolongé."
            ),
        },
        # ── METFORMINE / GLUCOPHAGE ──────────────────────────────────────
        {
            "medicament": "Metformine (Glucophage)",
            "section": "Indications",
            "texte": (
                "La metformine (Glucophage) est un antidiabétique oral de la classe des biguanides, "
                "indiqué dans le traitement du diabète de type 2, en première intention, "
                "notamment chez les patients en surpoids. "
                "Elle agit en diminuant la production hépatique de glucose et en améliorant "
                "la sensibilité à l'insuline."
            ),
        },
        {
            "medicament": "Metformine (Glucophage)",
            "section": "Posologie, effets indésirables et contre-indications",
            "texte": (
                "Posologie : débuter à 500 mg 1 à 2 fois par jour pendant les repas, "
                "augmenter progressivement. Dose maximale : 3 g par jour. "
                "Effets indésirables : troubles digestifs très fréquents en début de traitement "
                "(nausées, diarrhées, douleurs abdominales) — atténués si prise pendant les repas. "
                "Risque rare mais grave : acidose lactique (surtout en cas d'insuffisance rénale). "
                "Contre-indiqué en cas d'insuffisance rénale sévère (DFG < 30 mL/min), "
                "d'insuffisance hépatique, d'alcoolisme. "
                "Arrêter 48 h avant injection de produit de contraste iodé."
            ),
        },
    ]


# ─────────────────────────────────────────────
# ÉTAPE 2 – CHUNKING
# ─────────────────────────────────────────────

def chunker(texte: str, taille_max: int = TAILLE_CHUNK, overlap: int = OVERLAP_CHUNK) -> list[str]:
    """
    Découpe un texte en chunks avec chevauchement.
    Priorité : découpage sur les sauts de ligne doubles (paragraphes),
    puis sur les phrases, puis par tranche de caractères.
    """
    # Si le texte est déjà court, pas besoin de découper
    if len(texte) <= taille_max:
        return [texte.strip()]

    chunks = []
    debut = 0

    while debut < len(texte):
        fin = debut + taille_max

        if fin >= len(texte):
            # Dernier morceau
            chunk = texte[debut:].strip()
            if chunk:
                chunks.append(chunk)
            break

        # Chercher un séparateur naturel (paragraphe > phrase > espace)
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

        # Avancer avec overlap
        debut = max(debut + 1, coupure + 1 - overlap)

    return chunks


# ─────────────────────────────────────────────
# ÉTAPE 3 – EMBEDDING
# ─────────────────────────────────────────────

def embedder_chunks(chunks: list[str], modele: SentenceTransformer) -> np.ndarray:
    """
    Transforme une liste de chunks en vecteurs via sentence-transformers.
    Affiche une progression tous les 10 chunks.
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
    Crée un index FAISS à partir des vecteurs normalisés.
    On utilise IndexFlatIP (produit scalaire) avec vecteurs L2-normalisés
    pour obtenir une similarité cosinus, plus appropriée pour du texte.
    """
    dimension = vecteurs.shape[1]

    # Normalisation L2 pour simuler la similarité cosinus
    faiss.normalize_L2(vecteurs)

    index = faiss.IndexFlatIP(dimension)  # IP = Inner Product (produit scalaire)
    index.add(vecteurs)
    print(f"  Index FAISS créé : {index.ntotal} vecteurs, dimension {dimension}.")
    return index


def sauvegarder_index(index: faiss.Index, chunks_avec_meta: list[dict], chemin_index: str, chemin_meta: str):
    """
    Persiste l'index FAISS et les métadonnées associées sur disque.
    Les deux fichiers doivent rester en phase : même ordre, même nombre d'entrées.
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
    documents_bruts = corpus_fallback()

    # Tentative d'enrichissement via l'API (optionnel)
    # Décommenter pour essayer l'API en ligne :
    # for nom in MEDICAMENTS_CORPUS:
    #     data = recuperer_medicament_api(nom)
    #     if data:
    #         texte = construire_texte_medicament(nom, data)
    #         documents_bruts.append({"medicament": nom, "section": "API BDPM", "texte": texte})

    print(f"  {len(documents_bruts)} sections de notices chargées.")

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
