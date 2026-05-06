import json
# Permet de récupérer les données indéxés
def load_chunks(path="data/chunks_meta.json"):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
    

def retrieve(contexte_patient, model, collection=None, n=3):
    chunks = load_chunks()

    # Pour l'instant : sélection naïve
    # (suffisant pour valider tout le RAG)
    return chunks[:n], None
