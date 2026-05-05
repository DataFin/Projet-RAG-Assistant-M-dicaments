# Compte-rendu — TP RAG Médicaments

## Difficultés rencontrées

### 1. Accès aux données (BDPM)
L'API publique `api.medicaments.gouv.fr` retourne des données structurées JSON mais peu détaillées
sur la posologie et les effets indésirables. Les RCP (Résumés des Caractéristiques du Produit)
complets sont dans les fichiers `CIS_RCP.zip` de data.gouv.fr, mais leur format texte brut
nécessite un nettoyage important (balises, encodage Latin-1).

**Décision prise** : constitution d'un corpus manuel de qualité pour 10 médicaments,
organisé par sections de notice, plutôt qu'un corpus automatique mais bruité.

### 2. Chunking adapté aux notices médicales
Les notices sont structurées en sections bien définies (Indications, Posologie, Contre-indications,
Effets indésirables, Interactions). Un chunking par taille fixe aurait mélangé les informations.

**Décision prise** : chunking **sémantique par section** — chaque section constitue un chunk distinct,
avec découpage par taille (600 caractères, overlap 80) uniquement si la section est très longue.
Avantage : la métadonnée `section` permet au LLM de citer précisément l'origine de l'information.

### 3. Choix de la mesure de similarité FAISS
`IndexFlatL2` mesure la distance euclidienne — moins adaptée au texte sémantique.
`IndexFlatIP` avec vecteurs normalisés L2 simule la similarité cosinus, standard en NLP.

**Décision prise** : `IndexFlatIP` + normalisation L2 systématique des vecteurs
(à l'indexation ET à la recherche).

### 4. Gestion du disclaimer médical
Le risque principal d'un assistant médicaments est qu'un utilisateur suive ses conseils
sans consulter un professionnel. Le LLM peut paraître très convaincant même quand il invente.

**Décision prise** :
- Disclaimer imposé dans le prompt système ET affiché séparément dans le code (double sécurité)
- Temperature = 0.2 pour limiter les hallucinations
- Seuil de confiance : si FAISS renvoie un score < 0.25, on n'appelle pas le LLM du tout

### 5. Questions sur plusieurs médicaments simultanément
Ex : "Puis-je prendre Doliprane et ibuprofène en même temps ?"
Un seul appel de recherche peut ne pas ramener les chunks des deux médicaments.

**Décision prise** : la reformulation (Bonus C) inclut les deux noms, ce qui améliore le rappel.
Le prompt demande au LLM de traiter chaque médicament séparément et de mettre en évidence
les interactions éventuelles.

## Décisions de conception

| Décision | Justification |
|----------|---------------|
| Chunking par section de notice | Structure sémantique naturelle des notices ANSM |
| `paraphrase-multilingual-mpnet-base-v2` | Corpus en français, modèle haute qualité multilingue |
| `IndexFlatIP` + normalisation L2 | Similarité cosinus standard pour du texte |
| `llama3-70b-8192` | Meilleure précision médicale, plus sûr pour ce cas d'usage |
| Temperature 0.2 | Réponses factuelles et reproductibles |
| k=4 chunks | Équilibre entre contexte riche et risque de dépassement du contexte LLM |
| Bonus C (reformulation) | Les questions colloquiales ("j'ai mal au ventre") ne matchent pas bien les termes médicaux |
