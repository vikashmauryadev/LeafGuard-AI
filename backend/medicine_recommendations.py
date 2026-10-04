
"""
LeafGuard AI — Basic Safe Medicine Catalog
"""

MEDICINE_CATALOG = []

SUPPORTED_DISEASES = [
    "Pepper bell bacterial spot",
    "Pepper bell healthy",
    "Potato early blight",
    "Potato late blight",
    "Potato healthy",
    "Tomato bacterial spot",
    "Tomato early blight",
    "Tomato late blight",
    "Tomato leaf mold",
    "Tomato Septoria leaf spot",
    "Tomato spider mites",
    "Tomato target spot",
    "Tomato yellow leaf curl virus",
    "Tomato mosaic virus",
    "Tomato healthy",
]


def get_medicine_recommendations(predicted_label: str) -> dict:
    label = (predicted_label or "").strip().lower()

    if not label:
        status = "verification_required"
        message = "Disease could not be identified reliably."

    elif "healthy" in label:
        status = "no_medicine_needed"
        message = "No pesticide is recommended for a healthy leaf."

    elif "virus" in label:
        status = "non_chemical_management"
        message = (
            "Follow local agricultural guidance to manage "
            "the viral disease and reduce its spread."
        )

    else:
        status = "verification_required"
        message = (
            "No verified, crop-specific medicine is available yet. "
            "Consult an agricultural expert before applying pesticides."
        )

    return {
        "status": status,
        "message": message,
        "medicines": [],
    }
