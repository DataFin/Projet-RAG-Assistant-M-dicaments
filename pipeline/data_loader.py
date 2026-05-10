"""
data_loader.py
==============
Responsabilité : charger et nettoyer les données médicales.

3 sources disponibles :
- CSV BDPM      : fichier local CIS_RCP.csv
- API BDPM      : appel HTTP en temps réel (optionnel)
- Fallback      : corpus manuel de secours

Si l'API est indisponible → on continue sans elle automatiquement.
"""

import os
import re
import math
import requests
import pandas as pd
from bs4 import BeautifulSoup
from pipeline.config import SECTIONS_CIBLES


class DataLoader:
    """
    Spécialiste du chargement des données médicales.

    Usage :
        loader = DataLoader(source="tous", chemin_csv="data/CIS_RCP.csv")
        documents = loader.charger()
    """

    def __init__(
        self,
        source: str = "tous",
        chemin_csv: str = "data/CIS_RCP.csv",
        limite: int = 50,
        medicaments: list = None,
    ):
        self.source      = source
        self.chemin_csv  = chemin_csv
        self.limite      = limite
        self.medicaments = medicaments or [
            "doliprane", "ibuprofène", "aspirine",
            "amoxicilline", "ventoline", "oméprazole",
            "metformine", "smecta", "imodium",
        ]
        self.documents = []

    def charger(self) -> list:
        """Charge les données selon la source choisie."""
        if self.source == "csv":
            self.documents = self._charger_csv()

        elif self.source == "api":
            self.documents = self._charger_api()

        elif self.source == "fallback":
            self.documents = self._corpus_fallback()

        elif self.source == "tous":
            print("  [1/3] Chargement depuis le CSV...")
            docs_csv = self._charger_csv()

            print("  [2/3] Tentative de l'API BDPM...")
            try:
                docs_api = self._charger_api()
            except Exception as e:
                print(f"  ⚠️  API indisponible : {e}")
                print("  ℹ️  Continuation sans l'API.")
                docs_api = []

            print("  [3/3] Ajout du corpus de secours...")
            docs_fallback = self._corpus_fallback()

            self.documents = docs_csv + docs_api + docs_fallback

        else:
            raise ValueError(f"Source inconnue : {self.source}.")

        print(f"  ✓ {len(self.documents)} sections chargées au total.")
        return self.documents

    def _charger_csv(self) -> list:
        """Charge les notices depuis le CSV BDPM avec extraction des tableaux."""
        if not os.path.exists(self.chemin_csv):
            print(f"  ⚠️  CSV introuvable : {self.chemin_csv}")
            return []

        df = pd.read_csv(self.chemin_csv, encoding='latin-1', sep='\t')
        documents = []

        for _, row in df.head(self.limite).iterrows():
            if not self._html_est_valide(row['RCP_html']):
                continue

            html = str(row['RCP_html'])
            html = html.encode('latin-1').decode('utf-8', errors='replace')
            soup = BeautifulSoup(html, 'html.parser')

            nom_tag = (soup.find(class_='AmmDenomination') or
                       soup.find(class_='AmmCorpsTexteGras'))
            nom = (
                self._nettoyer_nom(nom_tag.get_text())
                if nom_tag
                else f"Médicament {row['Code_CIS']}"
            )

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
                    if element.find('table'):
                        texte_table = self._extraire_tableaux(element)
                        if texte_table:
                            contenu.append(texte_table)
                    else:
                        texte = element.get_text(separator=' ').strip()
                        texte = re.sub(r'\s+', ' ', texte)
                        if texte:
                            contenu.append(texte)

                texte_section = ' '.join(contenu)
                texte_section = re.sub(r'\s+', ' ', texte_section).strip()

                if len(texte_section) > 50:
                    documents.append({
                        "medicament": nom,
                        "section": section_nom,
                        "texte": f"{section_nom} du médicament {nom} : {texte_section}",
                        "source": "CSV BDPM",
                    })

        print(f"    → {len(documents)} sections extraites du CSV.")
        return documents

    def _charger_api(self) -> list:
        """
        Interroge l'API BDPM.
        Arrête immédiatement si l'API est inaccessible.
        """
        documents = []
        url_base  = "https://api.medicaments.gouv.fr/1.0/medicaments"

        for nom in self.medicaments:
            try:
                print(f"    API → {nom}...")
                response = requests.get(
                    url_base,
                    params={"denomination": nom},
                    timeout=5,
                )
                response.raise_for_status()
                data = response.json()

                if not data or not isinstance(data, list):
                    continue

                medicament   = data[0]
                nom_officiel = medicament.get("denomination", nom)
                sections     = self._json_vers_sections(nom_officiel, medicament)
                documents.extend(sections)

            except requests.Timeout:
                print("    ⚠️  Timeout — API trop lente, on arrête.")
                break

            except requests.RequestException:
                print("    ⚠️  API inaccessible — continuation sans API.")
                break

        print(f"    → {len(documents)} sections récupérées via API.")
        return documents

    def _json_vers_sections(self, nom: str, data: dict) -> list:
        sections = []
        mapping  = {
            "indicationsTherapeutiques":   "Indications thérapeutiques",
            "posologie":                   "Posologie",
            "contreIndications":           "Contre-indications",
            "effetsIndesirables":          "Effets indésirables",
            "interactionsMedicamenteuses": "Interactions médicamenteuses",
        }
        for champ, nom_section in mapping.items():
            valeur = data.get(champ, "")
            if valeur and len(str(valeur)) > 30:
                sections.append({
                    "medicament": nom,
                    "section":    nom_section,
                    "texte":      f"{nom_section} du médicament {nom} : {valeur}",
                    "source":     "API BDPM",
                })
        return sections

    def _corpus_fallback(self) -> list:
        """Corpus local de médicaments courants."""
        return [
            {
                "medicament": "Doliprane (paracétamol)",
                "section":    "Indications",
                "texte": (
                    "Indications du Doliprane (paracétamol) : "
                    "Traitement des douleurs légères à modérées et états fébriles. "
                    "Maux de tête, douleurs dentaires, musculaires, fièvre."
                ),
                "source": "Fallback",
            },
            {
                "medicament": "Doliprane (paracétamol)",
                "section":    "Posologie",
                "texte": (
                    "Posologie du Doliprane (paracétamol) : "
                    "Adulte : 500 mg à 1 g par prise, max 4 g/jour. "
                    "Enfant : 15 mg/kg par prise toutes les 6 heures. "
                    "Enfant 25 kg : 375 mg par prise. "
                    "Intervalle minimum 4 heures entre prises."
                ),
                "source": "Fallback",
            },
            {
                "medicament": "Doliprane (paracétamol)",
                "section":    "Contre-indications",
                "texte": (
                    "Contre-indications du Doliprane (paracétamol) : "
                    "Allergie au paracétamol. Insuffisance hépatique sévère. "
                    "Alcool. Association avec autres médicaments à base de paracétamol."
                ),
                "source": "Fallback",
            },
            {
                "medicament": "Doliprane (paracétamol)",
                "section":    "Effets indésirables",
                "texte": (
                    "Effets indésirables du Doliprane (paracétamol) : "
                    "Rares : réactions allergiques. "
                    "Exceptionnels : atteintes hépatiques en cas de surdosage. "
                    "Bien toléré aux doses recommandées."
                ),
                "source": "Fallback",
            },
            {
                "medicament": "Doliprane (paracétamol)",
                "section":    "Interactions médicamenteuses",
                "texte": (
                    "Interactions du Doliprane (paracétamol) : "
                    "Déconseillé avec anticoagulants oraux et alcool. "
                    "Danger : association avec Dafalgan ou Efferalgan = surdosage !"
                ),
                "source": "Fallback",
            },
            {
                "medicament": "Ibuprofène (Advil, Nurofen)",
                "section":    "Indications",
                "texte": (
                    "Indications de l'ibuprofène (Advil, Nurofen) : "
                    "AINS. Douleurs légères à modérées, fièvre, "
                    "douleurs menstruelles, rhumatismales."
                ),
                "source": "Fallback",
            },
            {
                "medicament": "Ibuprofène (Advil, Nurofen)",
                "section":    "Posologie",
                "texte": (
                    "Posologie de l'ibuprofène : "
                    "Adulte et enfant > 40 kg : 200-400 mg, 3 fois/jour. "
                    "Max 1200 mg/jour. Prendre pendant les repas."
                ),
                "source": "Fallback",
            },
            {
                "medicament": "Ibuprofène (Advil, Nurofen)",
                "section":    "Contre-indications",
                "texte": (
                    "Contre-indications de l'ibuprofène : "
                    "Allergie aux AINS. Grossesse 6e mois. "
                    "Ulcère gastroduodénal. Insuffisance rénale, hépatique, cardiaque."
                ),
                "source": "Fallback",
            },
            {
                "medicament": "Ibuprofène (Advil, Nurofen)",
                "section":    "Effets indésirables",
                "texte": (
                    "Effets indésirables de l'ibuprofène : "
                    "Fréquents : troubles digestifs. "
                    "Rares : ulcère gastrique, hémorragie digestive."
                ),
                "source": "Fallback",
            },
            {
                "medicament": "Aspirine (Aspégic)",
                "section":    "Contre-indications",
                "texte": (
                    "Contre-indications de l'aspirine : "
                    "Enfants < 16 ans avec syndrome viral (risque Reye). "
                    "Allergie aux salicylés. Ulcère. Insuffisance rénale/hépatique."
                ),
                "source": "Fallback",
            },
            {
                "medicament": "Amoxicilline (Augmentin)",
                "section":    "Indications",
                "texte": (
                    "Indications de l'amoxicilline : "
                    "Antibiotique pour infections bactériennes : "
                    "angines, sinusites, otites, pneumonies, infections urinaires."
                ),
                "source": "Fallback",
            },
            {
                "medicament": "Ventoline (salbutamol)",
                "section":    "Indications",
                "texte": (
                    "Indications de la Ventoline : "
                    "Bronchodilatateur pour crises d'asthme et bronchospasmes."
                ),
                "source": "Fallback",
            },
            {
                "medicament": "Metformine (Glucophage)",
                "section":    "Indications",
                "texte": (
                    "Indications de la metformine : "
                    "Antidiabétique oral pour diabète type 2."
                ),
                "source": "Fallback",
            },
        ]

    def _html_est_valide(self, valeur) -> bool:
        if valeur is None:
            return False
        if isinstance(valeur, float) and math.isnan(valeur):
            return False
        return True

    def _extraire_tableaux(self, soup) -> str:
        texte_tableaux = []
        for table in soup.find_all('table'):
            lignes = table.find_all('tr')
            if not lignes:
                continue
            entetes = []
            for cellule in lignes[0].find_all(['th', 'td']):
                texte = re.sub(r'\s+', ' ', cellule.get_text(separator=' ').strip())
                if texte:
                    entetes.append(texte)
            for ligne in lignes[1:]:
                valeurs = []
                for cellule in ligne.find_all(['th', 'td']):
                    texte = re.sub(r'\s+', ' ', cellule.get_text(separator=' ').strip())
                    if texte:
                        valeurs.append(texte)
                if not valeurs:
                    continue
                if entetes and len(entetes) == len(valeurs):
                    phrase = " | ".join(f"{e} : {v}" for e, v in zip(entetes, valeurs))
                else:
                    phrase = " | ".join(valeurs)
                if len(phrase) > 20:
                    texte_tableaux.append(phrase)
        return "\n".join(texte_tableaux)

    def _nettoyer_nom(self, nom: str) -> str:
        return re.sub(r'\s+', ' ', nom.strip())
