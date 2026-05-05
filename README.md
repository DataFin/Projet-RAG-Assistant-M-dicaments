# RAG Médicaments — Assistant d'information pharmaceutique

Système RAG (Retrieval-Augmented Generation) construit avec Python, FAISS et Groq.
Ce projet répond à des questions sur les médicaments courants à partir de notices officielles ANSM.

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

# 4. Indexer la base (à faire une seule fois)
python indexation.py

# 5. Lancer l'assistant
python rag.py
```

---

## Architecture

```
PHASE 1 – INDEXATION (indexation.py)
  Corpus notices → Chunking par section → Embedding → Index FAISS
                                                          ↓
PHASE 2 – INTERROGATION (rag.py)
  Question → Reformulation → Embedding → Recherche FAISS → Groq LLaMA 3 → Réponse + sources
```

---

## Choix techniques

### Modèle d'embedding
`paraphrase-multilingual-mpnet-base-v2` (768 dimensions)

Choisi car le corpus est en français. Ce modèle est entraîné sur 50+ langues
et produit des embeddings de haute qualité pour des phrases et paragraphes courts.

### Stratégie de chunking
Découpage **par section de notice** (Indications / Posologie / Contre-indications /
Effets indésirables / Interactions médicamenteuses).

Justification : une notice médicale est structurée sémantiquement — chaque section
répond à un type de question différent. Conserver cette structure évite de mélanger
la posologie avec les effets indésirables dans un même chunk.

Taille max : **600 caractères** avec **80 caractères d'overlap**.
Ce format est adapté aux sections de notices (ni trop court, ni trop long pour le contexte Groq).

### Index FAISS
`IndexFlatIP` (produit scalaire) sur des vecteurs **normalisés L2**.
Cela revient à calculer la similarité cosinus, plus pertinente que la distance euclidienne
pour du texte sémantique. Un score proche de 1.0 = haute pertinence.

### Modèle LLM
`llama3-70b-8192` (Groq) — meilleure qualité de raisonnement médical.
Fallback : `llama3-8b-8192` si quota dépassé.

### Température
`temperature=0.2` pour des réponses factuelless stables, évitant les hallucinations.

---

## Bonus implémentés

| Bonus | Description |
|-------|-------------|
| **A — Historique** | Les 3 derniers échanges sont injectés dans chaque appel Groq |
| **B — Score de confiance** | Si le score FAISS < 0.25, avertissement affiché sans appel LLM |
| **C — Reformulation** | Groq reformule la question en mots-clés médicaux avant la recherche |

---

## Réponses aux questions de réflexion

**Q1. Stratégie de chunking pour des notices longues ?**
Chunking par section sémantique (Indications, Posologie…). Taille de 600 caractères
adaptée au rapport signal/bruit des notices médicales.

**Q2. Exploiter la structure des notices ?**
Oui : chaque section est indexée comme un chunk distinct avec sa métadonnée `section`.
Cela permet au LLM de citer la section exacte dans sa réponse.

**Q3. Distinguer effets secondaires vs posologie ?**
Via la métadonnée `section` stockée avec chaque chunk. Le LLM est guidé par le prompt
à toujours citer la section source.

**Q4. Question portant sur deux médicaments ?**
La recherche retourne k=4 chunks. Si la question mentionne deux médicaments,
la reformulation (Bonus C) inclut les deux noms, et FAISS retourne des chunks des deux.
Le prompt demande au LLM de les traiter séparément.

**Q5. Prompt système sécurisé ?**
Le prompt interdit explicitement d'inventer, impose la citation des sources,
et impose le disclaimer médical en fin de chaque réponse.

---

## Structure du projet

```
rag_medicaments/
├── indexation.py       # Pipeline d'indexation (Phase 1)
├── rag.py              # Assistant Q&A (Phase 2)
├── requirements.txt    # Dépendances Python
├── README.md           # Ce fichier
├── .gitignore          # .env et data/ exclus du git
├── .env                # Clé API (non commité)
└── data/               # Généré par indexation.py
    ├── faiss_index.bin
    └── chunks_meta.json
```

---

## Exemple de session

```
💊 Votre question : Quels sont les effets secondaires de l'ibuprofène ?

  🔍 Recherche sur : « ibuprofène effets indésirables effets secondaires »

  ⏳ Génération de la réponse...

D'après la notice officielle de l'ibuprofène (Advil, Nurofen), section "Effets indésirables" [Source 1] :

Les effets indésirables les plus fréquents sont :
- **Troubles digestifs** : nausées, vomissements, douleurs abdominales, diarrhées (surtout si pris à jeun)
- **Maux de tête et vertiges** (peu fréquents)
- **Réactions allergiques cutanées** (rares)
- En cas d'utilisation prolongée : risque d'ulcère gastrique et d'hémorragie digestive

⚕️ Ces informations ne remplacent pas l'avis d'un professionnel de santé.

📚 Sources consultées :
  [1] Ibuprofène (Advil, Nurofen) — Effets indésirables (pertinence : 91%)
  [2] Ibuprofène (Advil, Nurofen) — Contre-indications (pertinence : 74%)
```

---

*Document pédagogique – Cours RAG & LLM*
