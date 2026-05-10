# MediRAG — Assistant d'aide à la décision médicale

Système RAG (Retrieval-Augmented Generation) construit avec Python, ChromaDB et Groq.
Ce projet répond à des questions sur les médicaments à partir des notices officielles ANSM (BDPM).

---

## Lancer le projet

```bash
# 1. Créer l'environnement virtuel
python -m venv venv
source venv/bin/activate       # Linux / Mac
venv\Scripts\activate          # Windows

# 2. Installer les dépendances
pip install -r requirements.txt

# 3. Configurer la clé API Groq
echo "GROQ_API_KEY=votre_clé_ici" > .env

# 4. Télécharger la base de médicaments BDPM
Téléchargez le fichier `cis-rcp.zip` depuis data.gouv.fr :
https://www.data.gouv.fr/fr/datasets/base-de-donnees-publique-des-medicaments-medicaments-fr/

Placez le fichier dans le dossier `data/` :

# 5. Indexer la base (une seule fois — idempotent)
python indexation.py

# 6a. Lancer l'interface web Streamlit
streamlit run formulaire_streamlit.py

# 6b. Ou lancer l'interface CLI
python formulaire_clinique.py
```

---

## Architecture


```
PHASE 1 — INDEXATION (indexation.py)

  CSV BDPM / API BDPM / Corpus fallback
              ↓
         DataLoader
    Chargement + nettoyage HTML
    Extraction sections et tableaux
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


PHASE 2 — UTILISATION (formulaire_streamlit.py / formulaire_clinique.py)

  Médecin remplit le formulaire patient
              ↓
    AgentModerator — détection injections
              ↓
         RAG.answer(requete)
    VectorDB.retrieve() → chunks pertinents
    context.txt → prompt système
    Groq LLaMA 3.3 → réponse structurée
              ↓
         CacheLLM — sauvegarde disque
              ↓
       pdf_export.py → rapport PDF
```

---

## Structure du projet

```
Projet-RAG-Assistant-Médicaments/
│
├── pipeline/
│   ├── config.py              → Constantes centralisées
│   ├── data_loader.py         → class DataLoader (CSV + API + Fallback)
│   ├── chunker.py             → class Chunker
│   ├── embedder.py            → class Embedder
│   ├── vector_store.py        → class VectorDB (ChromaDB)
│   └── cache.py               → class CacheLLM
│
├── indexation.py              → Orchestrateur idempotent
├── rag.py                     → class RAG (retrieval + sécurité + LLM)
├── agent_moderator.py         → Détection prompt injections
├── context.txt                → Prompt système externalisé
├── formulaire_clinique.py     → Interface CLI
├── formulaire_streamlit.py    → Interface Web Streamlit
├── pdf_export.py              → Export PDF (ReportLab)
│
├── data/                      → Généré par indexation.py
│   ├── chroma_db/             → Base vectorielle ChromaDB
│   └── cache_llm.json         → Cache des réponses LLM
│
├── exports/                   → Rapports PDF générés
├── requirements.txt
├── .env                       → Clé API (non commité)
├── .gitignore
└── README.md
```

---

## Choix techniques

### Modèle d'embedding
`paraphrase-multilingual-mpnet-base-v2` (768 dimensions)

Choisi car le corpus est en français. Ce modèle est entraîné sur 50+ langues
et reconnaît les paraphrases médicales ("mal de tête" = "céphalées").

### Stratégie de chunking
Découpage à **deux niveaux** :
1. **Sémantique** : chaque section de notice constitue un chunk distinct (Indications, Posologie, Contre-indications, Effets indésirables, Interactions, Mises en garde, Surdosage)
2. **Par taille** : les sections longues sont redécoupées (600 car., overlap 80)

La métadonnée `section` permet au LLM de citer précisément l'origine de chaque information.

### Base vectorielle
**ChromaDB** (remplace FAISS) — persistance automatique, métadonnées intégrées,
idempotence native via `get_or_create_collection()`.
Le modèle d'embedding est sauvegardé dans les métadonnées de la collection,
garantissant la cohérence entre indexation et recherche.

### Modèle LLM
`llama-3.3-70b-versatile` (Groq) — meilleure précision médicale.
Temperature `0.2` pour des réponses factuelles et reproductibles.

### Sécurité
`AgentModerator` détecte les tentatives de prompt injection par expressions régulières
**avant** tout appel LLM — sans consommer de tokens.

### Cache LLM
Chaque réponse est mise en cache sur disque via hash MD5.
L'API Groq n'est appelée qu'une seule fois par question unique.

---

## Fonctionnalités

| Fonctionnalité | Description |
|----------------|-------------|
| Idempotence | `indexation.py` ne réindexe pas si la base existe déjà |
| Seuil de confiance | Si score ChromaDB < 0.25 → pas d'appel LLM |
| Cache LLM | Réponse instantanée si la question a déjà été posée |
| AgentModerator | Blocage des prompt injections avant appel API |
| Formulaire patient | Collecte profil clinique complet (âge, allergies, traitements...) |
| Export PDF | Rapport de consultation archivable (ReportLab) |
| Interface Streamlit | Design SaaS médical avec caducée SVG |

---

## Réponses aux questions de réflexion

**Q1. Stratégie de chunking pour des notices longues ?**
Chunking sémantique par section (Indications, Posologie...) puis par taille (600 car.)
uniquement si la section est trop longue. L'overlap de 80 caractères préserve le contexte aux coupures.

**Q2. Exploiter la structure des notices ?**
Oui : chaque section est indexée comme un chunk distinct avec sa métadonnée `section`.
ChromaDB permet de filtrer nativement par section lors de la recherche.

**Q3. Distinguer effets secondaires vs posologie ?**
Via la métadonnée `section` stockée avec chaque chunk dans ChromaDB.
Le LLM est guidé par `context.txt` à toujours citer la section source.

**Q4. Question portant sur deux médicaments ?**
La recherche retourne k=4 chunks. Si la question mentionne deux médicaments,
ChromaDB retourne des chunks des deux. Le prompt demande au LLM de les traiter séparément
et de signaler les interactions éventuelles.

**Q5. Prompt système sécurisé ?**
Le prompt `context.txt` interdit explicitement d'inventer, impose la citation des sources,
et requiert le disclaimer médical. L'`AgentModerator` bloque en amont les tentatives
de contournement avant tout appel LLM.

---

## Exemple de session

```
🏥 MediRAG — Clinical Decision Support

Identifiant patient : PAT_001
Âge : 8 ans
Poids : 25 kg
Allergies : aucune
Symptômes : fièvre 38.5°C, douleurs musculaires
Question : Quelle dose de Doliprane donner à cet enfant ?

🔍 Recherche dans les notices ANSM...
⏳ Génération de la recommandation médicale...

D'après la notice officielle du Doliprane (paracétamol), section "Posologie" [Source 1] :

Pour un enfant de 25 kg, la dose recommandée est de 15 mg/kg par prise,
soit 375 mg par prise, toutes les 4 à 6 heures.
Ne pas dépasser 4 prises par jour.

Aucune contre-indication détectée pour ce profil patient.

⚕️ Ces informations ne remplacent pas l'avis d'un professionnel de santé.

📚 Sources consultées :
  [1] Doliprane (paracétamol) — Posologie (pertinence : 91%)
  [2] Doliprane (paracétamol) — Contre-indications (pertinence : 74%)
```

---

*Document pédagogique — Cours RAG & LLM*
