import os
from datetime import datetime
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.enums import TA_LEFT, TA_CENTER
from reportlab.lib import colors


def generer_pdf_rapport(
    patient_id: str,
    contexte_patient: str,
    reponse_rag: str
):
    """
    Génère un rapport PDF d’aide à la décision clinique
    avec identité visuelle MediRAG.
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

    # ======================================================
    # Styles personnalisés MediRAG
    # ======================================================

    styles.add(ParagraphStyle(
        name="TitreMediRAG",
        fontSize=22,
        textColor=colors.HexColor("#0D9488"),
        alignment=TA_CENTER,
        spaceAfter=6,
        fontName="Helvetica-Bold"
    ))

    styles.add(ParagraphStyle(
        name="SousTitre",
        fontSize=10,
        textColor=colors.grey,
        alignment=TA_CENTER,
        spaceAfter=16
    ))

    styles.add(ParagraphStyle(
        name="Section",
        fontSize=13,
        fontName="Helvetica-Bold",
        spaceBefore=16,
        spaceAfter=8,
        textColor=colors.HexColor("#334155")
    ))

    styles.add(ParagraphStyle(
        name="Texte",
        fontSize=10,
        alignment=TA_LEFT,
        spaceAfter=8
    ))

    styles.add(ParagraphStyle(
        name="Disclaimer",
        fontSize=9,
        textColor=colors.red,
        spaceBefore=20,
        italic=True
    ))

    elements = []

    # ======================================================
    # En-tête MediRAG
    # ======================================================

    elements.append(
        Paragraph("🏥 MediRAG", styles["TitreMediRAG"])
    )

    elements.append(
        Paragraph(
            "Assistant d’aide à la décision médicale<br/>"
            "Basé sur les notices officielles ANSM (BDPM)",
            styles["SousTitre"]
        )
    )

    elements.append(
        Paragraph(
            f"Rapport généré le {datetime.now().strftime('%d/%m/%Y à %H:%M')}",
            styles["SousTitre"]
        )
    )

    elements.append(Spacer(1, 20))

    # ======================================================
    # Contexte patient
    # ======================================================

    elements.append(
        Paragraph("1. Contexte patient", styles["Section"])
    )

    elements.append(
        Paragraph(
            contexte_patient.replace("\n", "<br/>"),
            styles["Texte"]
        )
    )

    # ======================================================
    # Analyse RAG
    # ======================================================

    elements.append(
        Paragraph("2. Analyse issue de la base BDPM", styles["Section"])
    )

    elements.append(
        Paragraph(
            reponse_rag.replace("\n", "<br/>"),
            styles["Texte"]
        )
    )

    # ======================================================
    # Disclaimer
    # ======================================================

    elements.append(
        Paragraph(
            "⚠️ Ces informations ne remplacent pas l’avis d’un professionnel de santé. "
            "Toute décision thérapeutique relève de la responsabilité du praticien.",
            styles["Disclaimer"]
        )
    )

    doc.build(elements)

    print(f"📄 Rapport PDF MediRAG généré : {filename}")