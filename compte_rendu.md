# Compte-rendu — TP RAG Médicaments

---

## Architecture finale du projet

```
Projet-RAG-Assistant-Médicaments/
│
├── pipeline/
│   ├── config.py          → Constantes centralisées
│   ├── data_loader.py     → Chargement et nettoyage des données
│   ├── chunker.py         → Découpage des documents en chunks
│   ├── embedder.py        → Transformation texte → vecteurs
│   ├── vector_store.py    → Base vectorielle ChromaDB
│   └── cache.py           → Cache des réponses LLM
│
├── indexation.py          → Orchestrateur idempotent
├── rag.py                 → Classe RAG (retrieval + sécurité + LLM)
├── agent_moderator.py     → Détection prompt injections
├── context.txt            → Prompt système externalisé
├── formulaire_clinique.py → Interface CLI
├── formulaire_streamlit.py → Interface Web Streamlit
└── pdf_export.py          → Export PDF
```

---

## Flux du pipeline

```
PHASE 1 — INDEXATION (une seule fois)

CSV BDPM / API BDPM / Corpus fallback
            ↓
       DataLoader
  Chargement + nettoyage HTML
  Extraction des sections et tableaux
            ↓
         Chunker
  Découpage sémantique par section
  puis par taille (600 car., overlap 80)
            ↓
         Embedder
  paraphrase-multilingual-mpnet-base-v2
  texte → vecteurs 768 dimensions
            ↓
         VectorDB
  ChromaDB — stockage persistant
  Idempotence native
            ↓
     data/chroma_db/


PHASE 2 — UTILISATION (à chaque question)

Médecin remplit le formulaire patient
            ↓
    formulaire_clinique.py (CLI)
    formulaire_streamlit.py (Web)
            ↓
  AgentModerator.moderate(requete)
  Détection prompt injections
            ↓
     RAG.answer(requete)
  VectorDB.retrieve() → chunks pertinents
  context.txt → prompt système
  Groq LLM → réponse structurée
            ↓
  CacheLLM → sauvegarde sur disque
            ↓
  pdf_export.py → rapport PDF
```

---

## Difficultés rencontrées

### 1. Accès aux données (BDPM)

L'API publique `api.medicaments.gouv.fr` est inaccessible depuis le réseau de l'école (erreur DNS). Les RCP complets sont dans les fichiers `CIS_RCP.csv` de data.gouv.fr, mais leur format HTML nécessite un nettoyage important (balises, encodage Latin-1, tableaux complexes).

**Décision prise** : trois sources complémentaires avec fallback automatique :
- CSV BDPM comme source principale
- API BDPM conservée dans le code avec `break` au premier échec réseau
- Corpus manuel de 13 médicaments courants (Doliprane, Ibuprofène, Amoxicilline...) en dernier recours

### 2. Encodage et nettoyage des données

Les fichiers CSV sont encodés en Latin-1. Certaines lignes ont des valeurs `RCP_html` manquantes (`float NaN`). Les tableaux HTML des notices produisaient du texte illisible avec `get_text()`.

**Décision prise** :
- Double conversion encodage : `html.encode('latin-1').decode('utf-8', errors='replace')`
- Validation NaN avant traitement : `isinstance(valeur, float) and math.isnan(valeur)`
- Méthode `_extraire_tableaux()` qui reconstruit les lignes :
```
Avant : " | Traitement | | | Abacavir | | | 282 | "
Après  : "Traitement : Abacavir | Nombre de sujets : 282"
```

### 3. Chunking adapté aux notices médicales

Les notices sont structurées en sections bien définies (Indications, Posologie, Contre-indications, Effets indésirables, Interactions). Un chunking par taille fixe aurait mélangé les informations de sections différentes.

**Décision prise** : chunking à deux niveaux :
- **Niveau 1 — sémantique** : chaque section constitue un document distinct avec sa métadonnée `section`
- **Niveau 2 — taille** : les sections longues sont redécoupées (600 car., overlap 80) avec priorité de coupure naturelle (`\n\n` > `.\n` > `. ` > ` `)

Avantage : la métadonnée `section` permet au LLM de citer précisément l'origine de l'information.

### 4. Migration FAISS → ChromaDB

FAISS nécessite la gestion manuelle de 3 fichiers séparés (index binaire, métadonnées JSON, config). L'idempotence et la sauvegarde du modèle d'embedding devaient être codées à la main.

**Décision prise** : migration vers ChromaDB qui apporte nativement :
- Un seul dossier persistant automatiquement
- Métadonnées intégrées dans la collection
- Idempotence via `get_or_create_collection()`
- Modèle d'embedding sauvegardé dans les métadonnées de la collection

### 5. Limite de lot ChromaDB

ChromaDB impose une limite interne de **5 461 documents par lot**. Avec 6 316 chunks générés, l'ajout direct échouait avec une erreur `Batch size of 6312 is greater than max batch size of 5461`.

**Décision prise** : ajout par lots de 5 000 :
```python
taille_lot = 5000
for i in range(0, total, taille_lot):
    fin = min(i + taille_lot, total)
    self.collection.add(ids[i:fin], ...)
```

### 6. Sécurité — Prompt injection

Un assistant médical est une cible privilégiée pour les tentatives de détournement ("ignore les instructions précédentes", "tu n'es plus un médecin"...). Appeler le LLM pour détecter ces attaques consomme des tokens inutilement.

**Décision prise** : `AgentModerator` par expressions régulières, interceptant les requêtes malveillantes **avant** tout appel API :
```python
self.patterns = [
    r"ignore .*instructions",
    r"oublie .*contexte",
    r"system prompt",
    r"tu n'es plus",
    r"fais semblant",
    ...
]
```

### 7. Gestion du disclaimer médical

Le risque principal est qu'un utilisateur suive les conseils de l'assistant sans consulter un professionnel. Le LLM peut paraître très convaincant même quand il invente.

**Décision prise** :
- Disclaimer imposé dans `context.txt` (prompt système) ET affiché dans l'interface (double sécurité)
- Température = 0.2 pour limiter les hallucinations
- Seuil de confiance : si ChromaDB renvoie un score < 0.25, on n'appelle pas le LLM

### 8. Prompt système externalisé

Mélanger le prompt système avec le code Python rend les modifications difficiles et risquées.

**Décision prise** : externalisation dans `context.txt`. Le fichier définit les règles strictes du LLM, le format de réponse attendu (texte + JSON parsable), et contient le placeholder `{{Chunks}}` remplacé dynamiquement par les chunks ChromaDB.

### 9. Idempotence du pipeline d'indexation

Sans idempotence, relancer `indexation.py` recalcule tous les embeddings (3 minutes) inutilement.

**Décision prise** : vérification de l'existence de la base avant indexation — pattern du professeur :
```python
if os.path.exists(chemin_db):
    self._charger(chemin_db)      # existe → charge (2 secondes)
elif chunks:
    self._creer(chemin_db, chunks) # n'existe pas → crée (3 minutes)
```

---

## Décisions de conception

| Décision | Justification |
|----------|---------------|
| Chunking sémantique par section | Structure naturelle des notices ANSM, citation précise des sources |
| `paraphrase-multilingual-mpnet-base-v2` | Corpus en français, reconnaissance des paraphrases médicales |
| ChromaDB plutôt que FAISS | Persistance automatique, métadonnées intégrées, idempotence native |
| Normalisation L2 des vecteurs | Similarité cosinus — standard en NLP pour le texte sémantique |
| `llama-3.3-70b-versatile` | Meilleure précision médicale, plus fiable pour ce cas d'usage |
| Temperature 0.2 | Réponses factuelles et reproductibles, limitation des hallucinations |
| k = 4 chunks | Équilibre entre contexte riche et risque de dépassement du contexte LLM |
| Seuil confiance 0.25 | Refus de répondre si aucun chunk suffisamment pertinent |
| Cache LLM sur disque | Zéro appel API redondant pour la même question |
| AgentModerator par regex | Blocage des injections avant appel LLM, sans consommer de tokens |
| Prompt système dans context.txt | Séparation code / configuration, modifications sans toucher au Python |
| Format JSON dans la réponse LLM | Données parsables par l'interface pour un affichage structuré |
| Lots de 5 000 pour ChromaDB | Respect de la limite interne ChromaDB (max 5 461 par lot) |
| Export PDF ReportLab | Rapport de consultation archivable et professionnel |
