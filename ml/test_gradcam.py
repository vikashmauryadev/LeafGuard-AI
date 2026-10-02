
import sys
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "ml" / "models" / "leafguard_mobilenetv2.keras"
LABELS_PATH = BASE_DIR / "ml" / "models" / "class_labels.json"
OUTPUT_PATH = BASE_DIR / "ml" / "gradcam_test.png"

IMAGE_SIZE = (160, 160)


def main():
    if len(sys.argv) < 2:
        print('Usage: python ml/test_gradcam.py "FULL_IMAGE_PATH"')
        return

    image_path = Path(sys.argv[1])
    if not image_path.is_file():
        print(f"Image not found: {image_path}")
        return

    print("Loading trained model...")
    model = tf.keras.models.load_model(MODEL_PATH)

    with open(LABELS_PATH, "r", encoding="utf-8") as file:
        labels = json.load(file)

    # Access the nested MobileNetV2 backbone.
    backbone = model.get_layer("mobilenetv2_1.00_160")

    # Build an auxiliary model using the backbone's own input.
    # This avoids the disconnected-tensor error from the first attempt.
    feature_model = tf.keras.Model(
        inputs=backbone.input,
        outputs=backbone.output,
        name="gradcam_feature_model",
    )

    # Reuse the original trained classification head.
    pooling = model.get_layer("global_average_pooling2d")
    dropout = model.get_layer("dropout")
    classifier = model.get_layer("disease_probabilities")

    with Image.open(image_path) as source:
        original = source.convert("RGB")

    resized = original.resize(IMAGE_SIZE)
    image_array = np.asarray(resized, dtype=np.float32)
    image_array = preprocess_input(image_array)
    input_tensor = tf.convert_to_tensor(image_array[None, ...])

    # Calculate feature maps and class gradients.
    with tf.GradientTape() as tape:
        feature_maps = feature_model(input_tensor, training=False)
        tape.watch(feature_maps)

        pooled = pooling(feature_maps)
        dropped = dropout(pooled, training=False)
        predictions = classifier(dropped)

        class_index = tf.argmax(predictions[0])
        class_score = predictions[:, class_index]

    gradients = tape.gradient(class_score, feature_maps)

    if gradients is None:
        raise RuntimeError(
            "Gradients are unavailable for the feature maps."
        )

    weights = tf.reduce_mean(gradients, axis=(0, 1, 2))
    heatmap = tf.reduce_sum(feature_maps[0] * weights, axis=-1)
    heatmap = tf.maximum(heatmap, 0)

    maximum = float(tf.reduce_max(heatmap).numpy())
    if maximum <= 0:
        raise RuntimeError(
            "Grad-CAM heatmap is empty. No positive activation was found."
        )

    heatmap = heatmap / maximum
    heatmap = tf.image.resize(
        heatmap[..., tf.newaxis], IMAGE_SIZE
    ).numpy().squeeze()

    colored = plt.get_cmap("jet")(heatmap)[..., :3]
    colored = (colored * 255).astype(np.uint8)

    original_array = np.asarray(resized, dtype=np.float32)
    overlay = 0.55 * original_array + 0.45 * colored
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)

    comparison = Image.new("RGB", (320, 160))
    comparison.paste(resized, (0, 0))
    comparison.paste(Image.fromarray(overlay), (160, 0))
    comparison.save(OUTPUT_PATH)

    index = int(class_index.numpy())
    confidence = float(predictions[0, index].numpy())

    print(f"Predicted class: {labels[index]}")
    print(f"Model confidence: {confidence * 100:.2f}%")
    print(f"Heatmap maximum before normalization: {maximum:.6f}")
    print(f"Grad-CAM image saved to: {OUTPUT_PATH}")
    print("Left: original leaf | Right: Grad-CAM overlay")


if __name__ == "__main__":
    main()