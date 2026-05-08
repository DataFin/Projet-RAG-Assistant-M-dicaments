import os
from datetime import datetime
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.enums import TA_LEFT, TA_CENTER


def generer_pdf_rapport(
    patient_id: str,
    contexte_patient: str,
    reponse_rag: str
):
    """
    Génère un rapport PDF d’aide à la décision clinique,
    même si aucune recommandation n’est trouvée.
    """

    os.makedirs("exports", exist_ok=True)

    filename = (
        f"exports/rapport_{patient_id}_"
        f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    )

    doc = SimpleDocTemplate(
        filename,
        pagesize=A4,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40
    )

    styles = getSampleStyleSheet()

    # Styles personnalisés
    styles.add(ParagraphStyle(
        name="TitreCentre",
        parent=styles["Title"],
        alignment=TA_CENTER
    ))

    styles.add(ParagraphStyle(
        name="Section",
        parent=styles["Heading2"],
        spaceBefore=12,
        spaceAfter=6
    ))

    styles.add(ParagraphStyle(
        name="NormalJustifie",
        parent=styles["Normal"],
        alignment=TA_LEFT,
        spaceAfter=6
    ))

    styles.add(ParagraphStyle(
        name="Disclaimer",
        parent=styles["Italic"],
        textColor="grey",
        spaceBefore=12
    ))

    elements = []

    # ─────────────────────────────
    # Titre
    # ─────────────────────────────
    elements.append(
        Paragraph(
            "Rapport d’aide à la décision clinique",
            styles["TitreCentre"]
        )
    )
    elements.append(
        Paragraph(
            f"Généré le {datetime.now().strftime('%d/%m/%Y %H:%M')}",
            styles["Italic"]
        )
    )
    elements.append(Spacer(1, 20))

    # ─────────────────────────────
    # Contexte patient
    # ─────────────────────────────
    elements.append(Paragraph("1. Contexte patient", styles["Section"]))
    elements.append(
        Paragraph(
            contexte_patient.replace("\n", "<br/>"),
            styles["NormalJustifie"]
        )
    )
    elements.append(Spacer(1, 12))

    # ─────────────────────────────
    # Analyse RAG
    # ─────────────────────────────
    elements.append(
        Paragraph(
            "2. Analyse issue de la base BDPM",
            styles["Section"]
        )
    )

    elements.append(
        Paragraph(
            reponse_rag.replace("\n", "<br/>"),
            styles["NormalJustifie"]
        )
    )

    # ─────────────────────────────
    # Conclusion standardisée
    # ─────────────────────────────
    elements.append(Spacer(1, 12))
    elements.append(
        Paragraph(
            "3. Conclusion clinique",
            styles["Section"]
        )
    )

    if "aucun médicament" in reponse_rag.lower():
        conclusion = (
            "Aucun médicament pertinent n’a pu être identifié dans le corpus "
            "BDPM fourni pour ce contexte clinique. Une évaluation médicale "
            "complète est nécessaire."
        )
    else:
        conclusion = (
            "Des options médicamenteuses ont été identifiées à partir des "
            "notices officielles BDPM. Leur prescription éventuelle relève "
            "de la responsabilité du professionnel de santé."
        )

    elements.append(Paragraph(conclusion, styles["NormalJustifie"]))

    # ─────────────────────────────
    # Disclaimer
    # ─────────────────────────────
    elements.append(Spacer(1, 20))
    elements.append(
        Paragraph(
            "⚠️ Ces informations ne remplacent pas l’avis d’un professionnel de santé.",
            styles["Disclaimer"]
        )
    )

    doc.build(elements)

    print(f"📄 Rapport PDF généré : {filename}")