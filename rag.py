from groq import Groq
from dotenv import load_dotenv
import json
import os 

import chromadb
from vector_db import retrieve

#Permet de transformer les phrases en vecteurs numériques
from sentence_transformers import SentenceTransformer

# Permet de charger ma clé API depuis l'envirronnement virtuel où il se trouve
load_dotenv()

#Permet de charger mon context de manière dynamique
def read_file(file_path):
    with open(file_path, "r") as file:
        return file.read()


# Permet de faire le formatage de mes chunks 
def format_chunks(chunks):
    formatted_chunks = []

    for chunk in chunks:
        formatted_chunks.append(
            f"Médicament : {chunk['medicament']}\n"
            f"Section : {chunk['section']}\n"
            f"Contenu : {chunk['contenu']}"
        )

    return "\n\n---\n\n".join(formatted_chunks)

def build_context(question):
    context = read_file(file_path="context.txt")
    # Chargement d'un modèle pré-entrainé de sentence-transformers
    sentence_transformer_object = SentenceTransformer(
        "distiluse-base-multilingual-cased-v2"
    )

    chroma = chromadb.PersistentClient(path="./my_first_vector_db")
    collection = chroma.get_or_create_collection("random_knowledge")

    chunks, _ = retrieve(question, sentence_transformer_object, collection, n=3)
    context_filled = context.replace("{{Chunks}}", format_chunks(chunks))

    return context_filled




def answer_question(question):

    client = Groq(api_key=os.environ["GROQ_API_KEY"])

    chat_completion = client.chat.completions.create(
        messages=[

            {
                "role": "system",
                "content": build_context(question),
            },

            {
                "role": "user",
                "content": question,
            }
        ],

        model="llama-3.3-70b-versatile"
    )

    return chat_completion.choices[0].message.content


if __name__ == "__main__":

    response = answer_question(question="Comment s'appelle le champignon de mon gnome ? il est de couleur bleu?")
    print(response)