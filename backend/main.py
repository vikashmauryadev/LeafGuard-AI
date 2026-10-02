import io
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Image as PDFImage,
    Table,
    TableStyle,
    KeepTogether,
)
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Image as PDFImage,
    Table,
    TableStyle,
    Spacer,
)


import numpy as np
import tensorflow as tf
from PIL import Image, UnidentifiedImageError
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from backend.gradcam import generate_gradcam_overlay
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# --------------------------------------------------
# Configuration
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "ml" / "models" / "leafguard_mobilenetv2.keras"
LABELS_PATH = BASE_DIR / "ml" / "models" / "class_labels.json"
DATABASE_PATH = BASE_DIR / "leafguard_history.db"
def init_database():
    with sqlite3.connect(DATABASE_PATH) as connection:
        connection.execute("""
            CREATE TABLE IF NOT EXISTS prediction_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                filename TEXT,
                predicted_class TEXT NOT NULL,
                display_name TEXT,
                confidence REAL NOT NULL,
                status TEXT,
                advisory TEXT,
                image_width INTEGER,
                image_height INTEGER
            )
        """)
        connection.commit()


init_database()

IMAGE_SIZE = (160, 160)
MAX_FILE_SIZE = 10 * 1024 * 1024

app = FastAPI(
    title="LeafGuard AI API",
    description="Plant leaf disease classification and advisory API",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:5174",
    "http://127.0.0.1:5174",
    "http://localhost:5175",
    "http://localhost:5176",
    "https://leaf-guard-ai-mu.vercel.app",
],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# --------------------------------------------------
# Load model and class labels
# --------------------------------------------------

model = None
class_labels = []


@app.on_event("startup")
def load_model():
    global model, class_labels

    if not MODEL_PATH.exists():
        raise RuntimeError(f"Trained model not found: {MODEL_PATH}")

    if not LABELS_PATH.exists():
        raise RuntimeError(f"Class labels not found: {LABELS_PATH}")

    model = tf.keras.models.load_model(MODEL_PATH)

    with open(LABELS_PATH, "r", encoding="utf-8") as file:
        class_labels = json.load(file)

    if not isinstance(class_labels, list) or not class_labels:
        raise RuntimeError("Class labels must be a non-empty JSON list.")

    if model.output_shape[-1] != len(class_labels):
        raise RuntimeError(
            "Model output count does not match the number of class labels."
        )

    print(f"LeafGuard AI model loaded: {len(class_labels)} classes")


# --------------------------------------------------
# Disease advisory knowledge base
# --------------------------------------------------

ADVISORIES = {
    "Bacterial_spot": {
        "summary": "The model identified a pattern associated with bacterial spot.",
        "actions": [
            "Inspect affected leaves and nearby plants.",
            "Avoid overhead watering and prolonged leaf wetness.",
            "Remove badly affected plant material where appropriate.",
            "Seek local agricultural guidance before applying treatments.",
        ],
    },
    "Early_blight": {
        "summary": "The model identified a pattern associated with early blight.",
        "actions": [
            "Inspect older leaves for expanding spots and yellowing.",
            "Avoid splashing soil onto leaves during watering.",
            "Remove severely affected leaves when appropriate.",
            "Use locally recommended management practices if confirmed.",
        ],
    },
    "Late_blight": {
        "summary": "The model identified a pattern associated with late blight.",
        "actions": [
            "Inspect the plant and nearby plants promptly.",
            "Avoid handling wet plants and spreading plant debris.",
            "Seek prompt advice from a local agricultural expert.",
            "Follow locally approved disease-management guidance.",
        ],
    },
    "Leaf_Mold": {
        "summary": "The model identified a pattern associated with leaf mold.",
        "actions": [
            "Improve airflow around plants.",
            "Avoid excessive humidity and prolonged leaf wetness.",
            "Inspect the undersides of affected leaves.",
            "Confirm the diagnosis before choosing a treatment.",
        ],
    },
    "Septoria_leaf_spot": {
        "summary": "The model identified a pattern associated with Septoria leaf spot.",
        "actions": [
            "Inspect lower leaves for small spots with darker margins.",
            "Avoid splashing water from soil onto foliage.",
            "Remove affected plant debris where appropriate.",
            "Use local agricultural guidance for further management.",
        ],
    },
    "Spider_mites": {
        "summary": "The model identified a pattern associated with spider-mite damage.",
        "actions": [
            "Inspect leaf undersides for mites and fine webbing.",
            "Check neighboring plants for similar symptoms.",
            "Use an appropriate integrated pest-management approach.",
            "Confirm the pest before selecting any treatment.",
        ],
    },
    "Target_Spot": {
        "summary": "The model identified a pattern associated with target spot.",
        "actions": [
            "Inspect leaves for circular or target-like lesions.",
            "Improve airflow and avoid prolonged leaf wetness.",
            "Remove affected debris where appropriate.",
            "Confirm the diagnosis with a local agricultural expert.",
        ],
    },
    "mosaic_virus": {
        "summary": "The model identified a pattern associated with mosaic virus.",
        "actions": [
            "Inspect for mottled leaf color and distorted growth.",
            "Clean tools after handling potentially affected plants.",
            "Check for insect vectors and follow local guidance.",
            "Seek expert confirmation; there is no universal cure for plant viruses.",
        ],
    },
    "YellowLeaf__Curl_Virus": {
        "summary": "The model identified a pattern associated with yellow leaf curl virus.",
        "actions": [
            "Inspect for leaf curling, yellowing, and stunted growth.",
            "Check for whiteflies and other possible insect vectors.",
            "Avoid moving potentially infected plant material.",
            "Seek local agricultural guidance for confirmation and management.",
        ],
    },
    "healthy": {
        "summary": "The model classified the leaf as belonging to a healthy class.",
        "actions": [
            "Continue regular watering and balanced crop care.",
            "Monitor the plant for changes in leaf color or texture.",
            "Inspect plants regularly for early signs of disease.",
            "A healthy classification does not rule out every plant problem.",
        ],
    },
}


def get_advisory(label: str) -> dict:
    for key, advisory in ADVISORIES.items():
        if key.lower() in label.lower():
            return advisory

    return {
        "summary": (
            "The model identified a class associated with a possible "
            "plant disease or pest condition."
        ),
        "actions": [
            "Inspect the leaf and the rest of the plant carefully.",
            "Compare symptoms with trusted local agricultural resources.",
            "Seek expert confirmation before applying a treatment.",
        ],
    }


# --------------------------------------------------
# API endpoints
# --------------------------------------------------

@app.get("/")
def root():
    return {
        "name": "LeafGuard AI API",
        "status": "ready",
        "model_loaded": model is not None,
        "classes": len(class_labels),
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
    }


@app.get("/classes")
def get_classes():
    return {"classes": class_labels}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="The trained model is not loaded.",
        )

    allowed_types = {"image/jpeg", "image/png", "image/webp"}

    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=415,
            detail="Upload a JPG, PNG, or WEBP image.",
        )

    contents = await file.read()

    if not contents:
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")

    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail="Image exceeds the 10 MB upload limit.",
        )

    try:
        with Image.open(io.BytesIO(contents)) as source:
            source.verify()

        with Image.open(io.BytesIO(contents)) as source:
            width, height = source.size

            if width < 32 or height < 32:
                raise HTTPException(
                    status_code=400,
                    detail="Image dimensions must be at least 32 x 32 pixels.",
                )

            image = source.convert("RGB")

    except UnidentifiedImageError:
        raise HTTPException(
            status_code=400,
            detail="The uploaded file is not a valid image.",
        )
    except OSError:
        raise HTTPException(
            status_code=400,
            detail="The image could not be read.",
        )

    image = image.resize(IMAGE_SIZE)

    image_array = np.asarray(image, dtype=np.float32)
    image_array = np.expand_dims(image_array, axis=0)
    image_array = preprocess_input(image_array)

    probabilities = model.predict(image_array, verbose=0)[0]

    predicted_index = int(np.argmax(probabilities))
    predicted_label = class_labels[predicted_index]
    confidence = float(probabilities[predicted_index])

    advisory = get_advisory(predicted_label)

    # This is a display threshold, not a calibrated probability guarantee.
    if confidence < 0.60:
        status = "uncertain"
        message = (
            "The model is not sufficiently confident for a dependable "
            "classification. Try a clearer leaf image or seek expert advice."
        )
    else:
        status = "prediction"
        message = (
            "This is an AI-generated prediction, not a confirmed diagnosis."
        )
    # Save the successful prediction to SQLite history.
    with sqlite3.connect(DATABASE_PATH) as connection:
        connection.execute(
            """
            INSERT INTO prediction_history (
                timestamp,
                filename,
                predicted_class,
                display_name,
                confidence,
                status,
                advisory,
                image_width,
                image_height
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                datetime.now(timezone.utc).isoformat(),
                file.filename,
                predicted_label,
                predicted_label.replace("___", " — ").replace("_", " "),
                confidence,
                status,
                json.dumps(advisory, ensure_ascii=False),
                width,
                height,
            ),
        )
        connection.commit()
    return {
        "status": status,
        "filename": file.filename,
        "image_width": width,
        "image_height": height,
        "predicted_class": predicted_label,
        "display_name": (
            predicted_label.replace("___", " — ")
            .replace("__", " ")
            .replace("_", " ")
        ),
        "confidence": round(confidence, 4),
        "confidence_percent": round(confidence * 100, 2),
        "advisory": advisory,
        "message": message,
        "disclaimer": (
            "Predictions can be wrong, especially for real-world images "
            "that differ from the training dataset. Confirm important "
            "decisions with a qualified agricultural expert."
        ),
    }


@app.post("/explain")
async def explain(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="The trained model is not loaded.",
        )

    allowed_types = {"image/jpeg", "image/png", "image/webp"}

    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=415,
            detail="Upload a JPG, PNG, or WEBP image.",
        )

    contents = await file.read()

    if not contents:
        raise HTTPException(
            status_code=400,
            detail="The uploaded file is empty.",
        )

    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail="Image exceeds the 10 MB upload limit.",
        )

    try:
        with Image.open(io.BytesIO(contents)) as source:
            source.verify()

        with Image.open(io.BytesIO(contents)) as source:
            width, height = source.size

            if width < 32 or height < 32:
                raise HTTPException(
                    status_code=400,
                    detail="Image dimensions must be at least 32 x 32 pixels.",
                )

            image = source.convert("RGB")

    except UnidentifiedImageError:
        raise HTTPException(
            status_code=400,
            detail="The uploaded file is not a valid image.",
        )
    except OSError:
        raise HTTPException(
            status_code=400,
            detail="The image could not be read.",
        )

    resized = image.resize(IMAGE_SIZE)
    image_array = np.asarray(resized, dtype=np.float32)
    image_array = np.expand_dims(image_array, axis=0)
    image_array = preprocess_input(image_array)

    probabilities = model.predict(image_array, verbose=0)[0]
    predicted_index = int(np.argmax(probabilities))
    confidence = float(probabilities[predicted_index])

    try:
        png_bytes = generate_gradcam_overlay(
            model=model,
            image=image,
            class_index=predicted_index,
        )
    except Exception as exc:
        print(f"Grad-CAM generation failed: {exc}")
        raise HTTPException(
            status_code=500,
            detail="Could not generate the Grad-CAM explanation.",
        ) from exc

    return Response(
        content=png_bytes,
        media_type="image/png",
        headers={
            "Cache-Control": "no-store",
            "X-Predicted-Class": class_labels[predicted_index],
            "X-Confidence": f"{confidence * 100:.2f}",
        },
    )

@app.get("/history")
async def get_prediction_history(limit: int = 20):
    """Return recent predictions saved in the SQLite database."""

    limit = max(1, min(limit, 100))

    with sqlite3.connect(DATABASE_PATH) as connection:
        connection.row_factory = sqlite3.Row

        rows = connection.execute(
            """
            SELECT
                id,
                timestamp,
                filename,
                predicted_class,
                display_name,
                confidence,
                status,
                advisory,
                image_width,
                image_height
            FROM prediction_history
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()

    history = []

    for row in rows:
        item = dict(row)

        try:
            item["advisory"] = json.loads(item["advisory"])
        except (json.JSONDecodeError, TypeError):
            item["advisory"] = {}

        item["confidence_percent"] = round(
            item["confidence"] * 100, 2
        )

        history.append(item)

    return {
        "total_returned": len(history),
        "history": history,
    }


import csv
import io

from fastapi.responses import StreamingResponse


@app.get("/history/export")
async def export_prediction_history():
    """Export saved prediction history as a downloadable CSV file."""

    with sqlite3.connect(DATABASE_PATH) as connection:
        connection.row_factory = sqlite3.Row

        rows = connection.execute(
            """
            SELECT
                id,
                timestamp,
                filename,
                predicted_class,
                display_name,
                confidence,
                status,
                advisory,
                image_width,
                image_height
            FROM prediction_history
            ORDER BY id DESC
            """
        ).fetchall()

    output = io.StringIO(newline="")
    fieldnames = [
        "id",
        "timestamp",
        "filename",
        "predicted_class",
        "display_name",
        "confidence_percent",
        "status",
        "advisory",
        "image_width",
        "image_height",
    ]

    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()

    for row in rows:
        writer.writerow({
            "id": row["id"],
            "timestamp": row["timestamp"],
            "filename": row["filename"],
            "predicted_class": row["predicted_class"],
            "display_name": row["display_name"],
            "confidence_percent": (
                round(row["confidence"] * 100, 2)
                if row["confidence"] is not None
                else ""
            ),
            "status": row["status"],
            "advisory": row["advisory"],
            "image_width": row["image_width"],
            "image_height": row["image_height"],
        })

    csv_content = "\ufeff" + output.getvalue()
    output.close()

    return StreamingResponse(
        iter([csv_content]),
        media_type="text/csv; charset=utf-8",
        headers={
            "Content-Disposition":
                'attachment; filename="leafguard_prediction_history.csv"'
        },
    )


@app.get("/stats")
async def get_dashboard_stats():
    """Calculate dashboard statistics from saved prediction records."""

    with sqlite3.connect(DATABASE_PATH) as connection:
        connection.row_factory = sqlite3.Row

        summary = connection.execute(
            """
            SELECT
                COUNT(*) AS total_predictions,
                COUNT(DISTINCT predicted_class) AS unique_classes,
                SUM(
                    CASE WHEN status = 'uncertain'
                    THEN 1 ELSE 0 END
                ) AS uncertain_predictions,
                AVG(confidence) AS average_confidence
            FROM prediction_history
            """
        ).fetchone()

        disease_rows = connection.execute(
            """
            SELECT
                predicted_class,
                display_name,
                COUNT(*) AS count
            FROM prediction_history
            GROUP BY predicted_class, display_name
            ORDER BY count DESC, display_name ASC
            """
        ).fetchall()

    total = summary["total_predictions"] or 0
    uncertain = summary["uncertain_predictions"] or 0
    average = summary["average_confidence"]

    return {
        "total_predictions": total,
        "unique_classes": summary["unique_classes"] or 0,
        "uncertain_predictions": uncertain,
        "average_confidence_percent": (
            round(average * 100, 2)
            if average is not None
            else None
        ),
        "disease_counts": [
            {
                "predicted_class": row["predicted_class"],
                "display_name": row["display_name"],
                "count": row["count"],
            }
            for row in disease_rows
        ],
    }



@app.post("/report")
async def generate_prediction_report(file: UploadFile = File(...)):
    """Generate a downloadable PDF report without adding a history record."""

    if model is None:
        raise HTTPException(
            status_code=503,
            detail="The trained model is not loaded.",
        )

    allowed_types = {"image/jpeg", "image/png", "image/webp"}

    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=415,
            detail="Upload a JPG, PNG, or WEBP image.",
        )

    contents = await file.read()

    if not contents:
        raise HTTPException(
            status_code=400,
            detail="The uploaded file is empty.",
        )

    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail="Image exceeds the 10 MB upload limit.",
        )

    try:
        with Image.open(io.BytesIO(contents)) as source:
            source.verify()

        with Image.open(io.BytesIO(contents)) as source:
            width, height = source.size

            if width < 32 or height < 32:
                raise HTTPException(
                    status_code=400,
                    detail="Image dimensions must be at least 32 x 32 pixels.",
                )

            original_image = source.convert("RGB")

    except UnidentifiedImageError:
        raise HTTPException(
            status_code=400,
            detail="The uploaded file is not a valid image.",
        )
    except OSError:
        raise HTTPException(
            status_code=400,
            detail="The image could not be read.",
        )

    # Run the existing model on the uploaded leaf image.
    resized = original_image.resize(IMAGE_SIZE)
    image_array = np.asarray(resized, dtype=np.float32)
    image_array = np.expand_dims(image_array, axis=0)
    image_array = preprocess_input(image_array)

    probabilities = model.predict(image_array, verbose=0)[0]
    predicted_index = int(np.argmax(probabilities))
    predicted_label = class_labels[predicted_index]
    confidence = float(probabilities[predicted_index])

    display_name = (
        predicted_label.replace("___", " — ")
        .replace("__", " ")
        .replace("_", " ")
    )

    advisory = get_advisory(predicted_label)

    if confidence < 0.60:
        status = "Uncertain"
        message = (
            "The model is not sufficiently confident. "
            "Try a clearer image or seek expert advice."
        )
    else:
        status = "AI prediction"
        message = (
            "This is an AI-generated prediction, "
            "not a confirmed diagnosis."
        )

    report_time = datetime.now(timezone.utc).strftime(
        "%d %B %Y, %H:%M UTC"
    )

    # Convert the uploaded image to PNG for embedding in the PDF.
    original_buffer = io.BytesIO()
    original_image.save(original_buffer, format="PNG")
    original_buffer.seek(0)

    # Grad-CAM is optional; the report can still be generated if it fails.
    gradcam_buffer = None

    try:
        gradcam_bytes = generate_gradcam_overlay(
            model=model,
            image=original_image,
            class_index=predicted_index,
        )
        gradcam_buffer = io.BytesIO(gradcam_bytes)
    except Exception as exc:
        print(f"Grad-CAM unavailable for PDF report: {exc}")

    # Build the PDF in memory.
    pdf_buffer = io.BytesIO()

    document = SimpleDocTemplate(
        pdf_buffer,
        pagesize=A4,
        rightMargin=18 * mm,
        leftMargin=18 * mm,
        topMargin=16 * mm,
        bottomMargin=16 * mm,
        title="LeafGuard AI Prediction Report",
        author="LeafGuard AI",
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "LeafGuardTitle",
        parent=styles["Title"],
        fontSize=22,
        leading=27,
        textColor=colors.HexColor("#176B45"),
        alignment=TA_CENTER,
        spaceAfter=4 * mm,
    )

    subtitle_style = ParagraphStyle(
        "LeafGuardSubtitle",
        parent=styles["Normal"],
        fontSize=10,
        leading=14,
        textColor=colors.HexColor("#52645A"),
        alignment=TA_CENTER,
        spaceAfter=5 * mm,
    )

    heading_style = ParagraphStyle(
        "LeafGuardHeading",
        parent=styles["Heading2"],
        fontSize=13,
        leading=17,
        textColor=colors.HexColor("#176B45"),
        spaceBefore=5 * mm,
        spaceAfter=2 * mm,
    )

    body_style = ParagraphStyle(
        "LeafGuardBody",
        parent=styles["BodyText"],
        fontSize=9,
        leading=13,
        spaceAfter=2 * mm,
    )

    story = [
        Paragraph("LeafGuard AI", title_style),
        Paragraph("Plant Leaf Prediction Report", subtitle_style),
        Paragraph(f"<b>Generated:</b> {escape(report_time)}", body_style),
        Spacer(1, 3 * mm),
        Paragraph("Prediction Summary", heading_style),
    ]

    details = [
        ["Report field", "Result"],
        ["Image filename", escape(file.filename or "Uploaded image")],
        ["Predicted class", escape(display_name)],
        ["Model confidence", f"{confidence * 100:.2f}%"],
        ["Result status", escape(status)],
        ["Original image dimensions", f"{width} × {height} pixels"],
    ]

    details_table = Table(
        details,
        colWidths=[52 * mm, 112 * mm],
        repeatRows=1,
    )

    details_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#176B45")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#F4F8F5")),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#D5E1D8")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 7),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
    ]))

    story.append(details_table)

    story.append(Paragraph("Uploaded Leaf Image", heading_style))
    story.append(
        PDFImage(
            original_buffer,
            width=72 * mm,
            height=72 * mm,
            kind="proportional",
        )
    )

    if gradcam_buffer is not None:
        story.append(
            Paragraph("Grad-CAM Visual Explanation", heading_style)
        )
        story.append(
            Paragraph(
                "The heatmap highlights image regions that influenced "
                "the model's prediction. It is an explanation aid, "
                "not proof of disease.",
                body_style,
            )
        )
        story.append(
            PDFImage(
                gradcam_buffer,
                width=72 * mm,
                height=72 * mm,
                kind="proportional",
            )
        )

    story.append(Paragraph("Advisory and Suggested Actions", heading_style))

    summary = advisory.get("summary", "No advisory summary is available.")
    story.append(
        Paragraph(f"<b>Summary:</b> {escape(str(summary))}", body_style)
    )

    for action in advisory.get("actions", []):
        story.append(
            Paragraph(f"&bull; {escape(str(action))}", body_style)
        )

    story.append(Paragraph("Model Assessment", heading_style))
    story.append(Paragraph(escape(message), body_style))

    disclaimer = (
        "Important: This report contains an AI-generated prediction, "
        "not a confirmed agricultural diagnosis. Confidence is the "
        "model's output score and may not represent real-world accuracy. "
        "Images, lighting, crop variety, and environmental conditions "
        "can affect results. Confirm important decisions with a "
        "qualified agricultural expert before applying treatments."
    )

    story.append(Paragraph("Limitations and Disclaimer", heading_style))
    story.append(Paragraph(escape(disclaimer), body_style))

    document.build(story)
    pdf_buffer.seek(0)

    return Response(
        content=pdf_buffer.getvalue(),
        media_type="application/pdf",
        headers={
            "Content-Disposition":
                'attachment; filename="leafguard_prediction_report.pdf"',
            "Cache-Control": "no-store",
        },
    )