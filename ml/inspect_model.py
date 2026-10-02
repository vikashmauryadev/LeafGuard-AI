
from pathlib import Path
import tensorflow as tf

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "ml" / "models" / "leafguard_mobilenetv2.keras"

print("Loading LeafGuard AI model...")

model = tf.keras.models.load_model(MODEL_PATH)

print("\n=== TOP-LEVEL LAYERS ===")
for layer in model.layers:
    print(f"{layer.name}: {type(layer).__name__}")

print("\n=== CONVOLUTIONAL LAYERS ===")

def inspect_layers(container, prefix=""):
    for layer in container.layers:
        name = f"{prefix}{layer.name}"

        if isinstance(layer, tf.keras.layers.Conv2D):
            print(f"{name} | filters={layer.filters}")

        if isinstance(layer, tf.keras.Model):
            inspect_layers(layer, prefix=f"{name}/")

inspect_layers(model)

print("\nInspection complete.")