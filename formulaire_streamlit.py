import streamlit as st
from rag import RAG
from pdf_export import generer_pdf_rapport
from pipeline.vector_store import VectorDB
from pipeline.config import CHEMIN_DB
import os
from datetime import datetime
import re

# ======================================================
# UTILITAIRES LLM
# ======================================================

def separer_texte_et_json(reponse_llm: str) -> tuple[str, str]:
    debut_json = reponse_llm.find("{")
    if debut_json == -1:
        return reponse_llm.strip(), ""
    return reponse_llm[:debut_json].strip(), reponse_llm[debut_json:].strip()


def nettoyer_texte_llm(texte: str) -> str:
    texte = re.sub(r"```[a-zA-Z]*", "", texte)
    texte = texte.replace("```", "")
    return texte.strip()

# ======================================================
# CONFIGURATION PAGE
# ======================================================

st.set_page_config(
    page_title="MediRAG — Assistant Médical",
    page_icon="🏥",
    layout="wide"
)

# ======================================================
# STYLE CSS (inchangé)
# ======================================================

st.markdown("""
<style>
.stApp {
    background-color: #F5F7FB;
    color: #0F172A;
}

h1, h2, h3 {
    color: #0A0E3A;
    font-weight: 700;
}

p, label, span {
    color: #0F172A;
}

textarea, input {
    border-radius: 10px !important;
    border: 1px solid #CBD5E1 !important;
    background-color: #FFFFFF !important;
}

button[kind="primary"] {
    background: linear-gradient(90deg, #2563EB, #38BDF8);
    border: none;
    border-radius: 12px;
    color: white;
    font-weight: 700;
    padding: 12px 20px;
}

button[kind="primary"]:hover {
    background: linear-gradient(90deg, #1D4ED8, #0EA5E9);
}

.medirag-result {
    background:#FFFFFF;
    padding:20px;
    border-left:6px solid #3B6EEB;
    border-radius:14px;
    box-shadow: 0 14px 36px rgba(10,14,58,0.22);
}

hr {
    border: none;
    height: 1px;
    background: linear-gradient(
        90deg,
        rgba(0,0,0,0) 0%,
        #CBD5E1 50%,
        rgba(0,0,0,0) 100%
    );
    margin: 32px 0;
}

footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ======================================================
# HEADER — TITRE & SOUS-TITRE CENTRÉS
# ======================================================

st.markdown("""
<h1 style="text-align:center; font-size:48px; color:#065F46;">
🏥 MediRAG
</h1>

<p style="text-align:center; font-size:20px; font-weight:700; color:#047857;">
Assistant d’aide à la décision médicale — Notices officielles ANSM (BDPM)
</p>
""", unsafe_allow_html=True)

st.warning("⚠️ Ces informations ne remplacent pas l’avis d’un professionnel de santé.")

# ======================================================
# INITIALISATION SYSTEME
# ======================================================

@st.cache_resource
def init_systeme():
    rag = RAG(vector_db_path=CHEMIN_DB)
    vectordb = VectorDB(CHEMIN_DB)
    return rag, vectordb

rag, vectordb = init_systeme()

# ======================================================
# BASE DE CONNAISSANCES & GUIDE — ALIGNEMENT PARFAIT
# ======================================================

col_stats, col_guide = st.columns(2, gap="large")

with col_stats:
    st.markdown("### Base de connaissances MediRAG")
    st.markdown(f"- **Chunks indexés** : {vectordb.collection.count():,}".replace(",", " "))
    st.markdown("- **Source** : ANSM / BDPM")
    st.markdown("- **Type de documents** : Notices officielles")

with col_guide:
    st.markdown("### Guide d’utilisation")
    st.markdown("- **Saisir les informations cliniques du patient**")
    st.markdown("- **Décrire précisément les symptômes et le contexte médical**")
    st.markdown("- **Analyser les recommandations issues des notices officielles ANSM**")
    st.markdown("- **Télécharger et archiver le rapport PDF généré**")

st.divider()

# ======================================================
# FORMULAIRE CLINIQUE
# ======================================================

st.markdown("## Dossier patient")

with st.form("formulaire_patient"):
    col_g, col_d = st.columns(2)

    with col_g:
        identifiant = st.text_input("Identifiant patient", value="PAT_TEST")
        age = st.number_input("Âge", 0, 130, 40)
        sexe = st.selectbox("Sexe", ["F", "M", "Autre"])

    with col_d:
        allergies = st.text_area("Allergies / intolérances")
        traitements = st.text_area("Traitements en cours")

    symptomes = st.text_area("Symptômes", placeholder="Fièvre, douleurs musculaires…")
    antecedents = st.text_area("Antécédents médicaux")
    autres = st.text_area("Autres informations cliniques")

    submitted = st.form_submit_button(
        "🔍 Analyser le dossier patient",
        use_container_width=True
    )

# ======================================================
# TRAITEMENT
# ======================================================

if submitted:

    if not symptomes.strip():
        st.error("❌ Les symptômes sont obligatoires.")
        st.stop()

    contexte_patient = f"""
Patient : {identifiant}
Âge : {age} ans
Sexe : {sexe}

SYMPTÔMES :
{symptomes}

ANTÉCÉDENTS :
{antecedents}

TRAITEMENTS :
{traitements}

ALLERGIES :
{allergies}

AUTRES :
{autres}
"""

    with st.spinner("🔍 Analyse MediRAG en cours..."):
        reponse_brute = rag.answer(contexte_patient)

    texte_brut, _ = separer_texte_et_json(reponse_brute)
    texte_propre = nettoyer_texte_llm(texte_brut)

    st.markdown("## Recommandation médicale")
    st.markdown(
        f"<div class='medirag-result'>{texte_propre}</div>",
        unsafe_allow_html=True
    )

    os.makedirs("exports", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    pdf_path = f"exports/rapport_{identifiant}_{timestamp}.pdf"

    generer_pdf_rapport(
        patient_id=identifiant,
        contexte_patient=contexte_patient,
        reponse_rag=texte_propre
    )

    with open(pdf_path, "rb") as f:
        st.download_button(
            "📥 Télécharger le rapport MediRAG (PDF)",
            data=f,
            file_name=os.path.basename(pdf_path),
            mime="application/pdf",
            use_container_width=True
        )

    st.success("✅ Analyse MediRAG terminée")

# ======================================================
# FOOTER
# ======================================================

st.markdown("""
<hr>
<p style="text-align:center; font-size:12px; color:#475569;">
MediRAG — Assistant médical basé sur les notices officielles ANSM (BDPM).<br>
Outil d’aide à la décision clinique, non substitut à un avis médical.
</p>
""", unsafe_allow_html=True)