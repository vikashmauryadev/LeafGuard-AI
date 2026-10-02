
import json
import random
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

# --------------------------------------------------
# Configuration
# --------------------------------------------------
SEED = 42
IMAGE_SIZE = (160, 160)
BATCH_SIZE = 32
EPOCHS = 5

DATASET_DIR = Path(
    r"D:\Download\archive\PlantVillage\PlantVillage"
)

PROJECT_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_DIR / "ml" / "models"

MODEL_PATH = MODEL_DIR / "leafguard_mobilenetv2.keras"
LABELS_PATH = MODEL_DIR / "class_labels.json"
METRICS_PATH = MODEL_DIR / "evaluation_metrics.json"

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

MODEL_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------
# 1. Find images and class labels
# --------------------------------------------------
if not DATASET_DIR.is_dir():
    raise FileNotFoundError(
        f"Dataset folder not found: {DATASET_DIR}"
    )

supported_extensions = {".jpg", ".jpeg", ".png", ".bmp"}

class_names = sorted(
    folder.name
    for folder in DATASET_DIR.iterdir()
    if folder.is_dir()
    and any(
        file.is_file()
        and file.suffix.lower() in supported_extensions
        for file in folder.iterdir()
    )
)

if len(class_names) < 2:
    raise ValueError(
        "Expected at least two class folders containing images."
    )

class_to_index = {
    name: index for index, name in enumerate(class_names)
}

image_paths = []
image_labels = []

for class_name in class_names:
    class_dir = DATASET_DIR / class_name

    for file_path in class_dir.rglob("*"):
        if (
            file_path.is_file()
            and file_path.suffix.lower() in supported_extensions
        ):
            image_paths.append(str(file_path))
            image_labels.append(class_to_index[class_name])

image_paths = np.asarray(image_paths)
image_labels = np.asarray(image_labels, dtype=np.int32)

if len(image_paths) == 0:
    raise ValueError("No supported image files were found.")

print(f"Classes: {len(class_names)}")
print(f"Images found: {len(image_paths)}")

for index, name in enumerate(class_names):
    print(f"  {name}: {np.sum(image_labels == index)}")

# --------------------------------------------------
# 2. Stratified train / validation / test split
# --------------------------------------------------
train_paths, remaining_paths, train_labels, remaining_labels = (
    train_test_split(
        image_paths,
        image_labels,
        test_size=0.30,
        random_state=SEED,
        stratify=image_labels,
    )
)

val_paths, test_paths, val_labels, test_labels = (
    train_test_split(
        remaining_paths,
        remaining_labels,
        test_size=0.50,
        random_state=SEED,
        stratify=remaining_labels,
    )
)

print(f"\nTraining images:   {len(train_paths)}")
print(f"Validation images: {len(val_paths)}")
print(f"Test images:       {len(test_paths)}")

# --------------------------------------------------
# 3. Build efficient image pipelines
# --------------------------------------------------
AUTOTUNE = tf.data.AUTOTUNE


def load_image(path, label):
    image_bytes = tf.io.read_file(path)

    image = tf.io.decode_image(
        image_bytes,
        channels=3,
        expand_animations=False,
    )

    image.set_shape([None, None, 3])

    image = tf.image.resize(image, IMAGE_SIZE)
    image = tf.keras.applications.mobilenet_v2.preprocess_input(
        image
    )

    return image, label


def make_dataset(paths, labels, training=False):
    dataset = tf.data.Dataset.from_tensor_slices(
        (paths, labels)
    )

    if training:
        dataset = dataset.shuffle(
            buffer_size=min(len(paths), 3000),
            seed=SEED,
            reshuffle_each_iteration=True,
        )

    dataset = dataset.map(
        load_image,
        num_parallel_calls=AUTOTUNE,
    )

    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(AUTOTUNE)

    return dataset


train_ds = make_dataset(
    train_paths, train_labels, training=True
)
val_ds = make_dataset(val_paths, val_labels)
test_ds = make_dataset(test_paths, test_labels)

# --------------------------------------------------
# 4. Calculate class weights
# --------------------------------------------------
weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(train_labels),
    y=train_labels,
)

class_weights = {
    int(class_index): float(weight)
    for class_index, weight in zip(
        np.unique(train_labels), weights
    )
}

# --------------------------------------------------
# 5. Create MobileNetV2 transfer-learning model
# --------------------------------------------------
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(*IMAGE_SIZE, 3),
    include_top=False,
    weights="imagenet",
)

base_model.trainable = False

inputs = tf.keras.Input(
    shape=(*IMAGE_SIZE, 3),
    name="leaf_image",
)

features = base_model(inputs, training=False)
features = tf.keras.layers.GlobalAveragePooling2D()(features)
features = tf.keras.layers.Dropout(0.25)(features)

outputs = tf.keras.layers.Dense(
    len(class_names),
    activation="softmax",
    name="disease_probabilities",
)(features)

model = tf.keras.Model(inputs, outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

model.summary()

# --------------------------------------------------
# 6. Train the classification head
# --------------------------------------------------
callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        filepath=str(MODEL_PATH),
        monitor="val_loss",
        save_best_only=True,
        verbose=1,
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=2,
        restore_best_weights=True,
        verbose=1,
    ),
]

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    class_weight=class_weights,
    callbacks=callbacks,
)

# Save the best weights restored by early stopping.
model.save(MODEL_PATH)

# --------------------------------------------------
# 7. Evaluate on the held-out test set
# --------------------------------------------------
best_model = tf.keras.models.load_model(MODEL_PATH)

true_labels = []
predicted_labels = []

for batch_images, batch_labels in test_ds:
    probabilities = best_model.predict(
        batch_images, verbose=0
    )

    true_labels.extend(batch_labels.numpy().tolist())
    predicted_labels.extend(
        np.argmax(probabilities, axis=1).tolist()
    )

test_accuracy = accuracy_score(
    true_labels, predicted_labels
)

report = classification_report(
    true_labels,
    predicted_labels,
    labels=list(range(len(class_names))),
    target_names=class_names,
    output_dict=True,
    zero_division=0,
)

matrix = confusion_matrix(
    true_labels,
    predicted_labels,
    labels=list(range(len(class_names))),
)

print("\n========== TEST RESULTS ==========")
print(f"Test accuracy: {test_accuracy:.4f}")
print(
    classification_report(
        true_labels,
        predicted_labels,
        labels=list(range(len(class_names))),
        target_names=class_names,
        zero_division=0,
    )
)

# --------------------------------------------------
# 8. Save labels and evaluation results
# --------------------------------------------------
with open(LABELS_PATH, "w", encoding="utf-8") as file:
    json.dump(class_names, file, indent=2)

metrics = {
    "test_accuracy": float(test_accuracy),
    "test_image_count": len(true_labels),
    "class_count": len(class_names),
    "class_names": class_names,
    "classification_report": report,
    "confusion_matrix": matrix.tolist(),
    "image_size": list(IMAGE_SIZE),
    "model_architecture": "MobileNetV2",
    "epochs_configured": EPOCHS,
    "dataset_path": str(DATASET_DIR),
    "split_seed": SEED,
}

with open(METRICS_PATH, "w", encoding="utf-8") as file:
    json.dump(metrics, file, indent=2)

print("\nTraining and evaluation completed.")
print(f"Model:   {MODEL_PATH}")
print(f"Labels:  {LABELS_PATH}")
print(f"Metrics: {METRICS_PATH}")