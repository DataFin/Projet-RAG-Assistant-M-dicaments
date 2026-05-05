def retrieve(question, sentence_transformer, collection=None, n=3):
    """
    Retourne les n chunks les plus pertinents pour une question donnée.

    Pour l’instant : MOCK
    Plus tard : recherche vectorielle réelle (FAISS / Chroma)
    """

    # MOCK de chunks (simulation de la base vectorielle)
    chunks = [
        {
            "medicament": "Doliprane",
            "section": "Effets indésirables",
            "contenu": (
                "Le paracétamol peut provoquer des réactions cutanées, "
                "des troubles hépatiques en cas de surdosage, "
                "et plus rarement des troubles sanguins."
            ),
        },
        {
            "medicament": "Ibuprofène",
            "section": "Contre-indications",
            "contenu": (
                "L’ibuprofène est contre-indiqué en cas d’ulcère gastro-duodénal, "
                "d’insuffisance rénale sévère ou de dernier trimestre de grossesse."
            ),
        },
        {
            "medicament": "Amoxicilline",
            "section": "Posologie",
            "contenu": (
                "La posologie usuelle chez l’adulte est de 1 g trois fois par jour, "
                "à adapter selon la gravité de l’infection."
            ),
        },
    ]

    # Plus tard :
    # - calcul embedding de la question
    # - similarité cosinus
    # - tri par score

    return chunks[:n], None