import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

from backend.gradcam import generate_gradcam_overlay

MODEL_PATH = ROOT / "ml" / "models" / "leafguard_mobilenetv2.keras"
LABELS_PATH = ROOT / "ml" / "models" / "class_labels.json"
OUTPUT_PATH = ROOT / "ml" / "gradcam_helper_test.png"

IMAGE_PATH = Path(
    r"D:\Download\archive\PlantVillage\Potato___Late_blight"
    r"\e4a264b9-fbae-4f38-86ce-d0a7d70e92e1___RS_LB 5211.JPG"
)


def main():
    print("Step 1: Checking test image...")
    if not IMAGE_PATH.is_file():
        raise FileNotFoundError(f"Image not found: {IMAGE_PATH}")

    print("Step 2: Loading trained model...")
    model = tf.keras.models.load_model(MODEL_PATH)

    with open(LABELS_PATH, "r", encoding="utf-8") as file:
        labels = json.load(file)

    print("Step 3: Reading leaf image...")
    with Image.open(IMAGE_PATH) as source:
        image = source.convert("RGB")

    array = np.asarray(
        image.resize((160, 160)),
        dtype=np.float32,
    )
    array = preprocess_input(array)

    print("Step 4: Predicting disease class...")
    predictions = model.predict(array[None, ...], verbose=0)
    class_index = int(np.argmax(predictions[0]))
    confidence = float(predictions[0][class_index])

    print(f"Predicted class: {labels[class_index]}")
    print(f"Confidence: {confidence * 100:.2f}%")

    print("Step 5: Generating Grad-CAM overlay...")
    png_bytes = generate_gradcam_overlay(
        model=model,
        image=image,
        class_index=class_index,
    )

    OUTPUT_PATH.write_bytes(png_bytes)

    print(f"SUCCESS: Image saved to {OUTPUT_PATH}")
    print(f"PNG size: {len(png_bytes):,} bytes")


if __name__ == "__main__":
    main()