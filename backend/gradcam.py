from io import BytesIO

import numpy as np
import tensorflow as tf
import matplotlib

from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


IMAGE_SIZE = (160, 160)


def generate_gradcam_overlay(
    model: tf.keras.Model,
    image: Image.Image,
    class_index: int,
) -> bytes:
    """
    Generate a Grad-CAM overlay for a selected class.

    Args:
        model: The loaded LeafGuard AI classification model.
        image: The original leaf image as a PIL Image.
        class_index: Index of the class to explain.

    Returns:
        PNG image bytes containing the Grad-CAM overlay.
    """
    backbone = model.get_layer("mobilenetv2_1.00_160")

    feature_model = tf.keras.Model(
        inputs=backbone.input,
        outputs=backbone.output,
        name="gradcam_feature_model",
    )

    pooling = model.get_layer("global_average_pooling2d")
    dropout = model.get_layer("dropout")
    classifier = model.get_layer("disease_probabilities")

    original = image.convert("RGB")
    resized = original.resize(IMAGE_SIZE)

    image_array = np.asarray(resized, dtype=np.float32)
    image_array = preprocess_input(image_array)
    input_tensor = tf.convert_to_tensor(image_array[None, ...])

    with tf.GradientTape() as tape:
        feature_maps = feature_model(input_tensor, training=False)
        tape.watch(feature_maps)

        pooled = pooling(feature_maps)
        dropped = dropout(pooled, training=False)
        predictions = classifier(dropped)
        class_score = predictions[:, class_index]

    gradients = tape.gradient(class_score, feature_maps)

    if gradients is None:
        raise RuntimeError("Grad-CAM gradients could not be calculated.")

    weights = tf.reduce_mean(gradients, axis=(0, 1, 2))
    heatmap = tf.reduce_sum(feature_maps[0] * weights, axis=-1)
    heatmap = tf.maximum(heatmap, 0)

    maximum = float(tf.reduce_max(heatmap).numpy())
    if maximum <= 0:
        raise RuntimeError("Grad-CAM produced an empty heatmap.")

    heatmap = heatmap / maximum
    heatmap = tf.image.resize(
        heatmap[..., tf.newaxis],
        IMAGE_SIZE,
    ).numpy().squeeze()

    colormap = matplotlib.colormaps["jet"]
    colored_heatmap = (colormap(heatmap)[..., :3] * 255).astype(np.uint8)

    original_array = np.asarray(resized, dtype=np.float32)
    overlay_array = (
        0.55 * original_array + 0.45 * colored_heatmap.astype(np.float32)
    )
    overlay_array = np.clip(overlay_array, 0, 255).astype(np.uint8)

    overlay = Image.fromarray(overlay_array)

    output = BytesIO()
    overlay.save(output, format="PNG")
    return output.getvalue()