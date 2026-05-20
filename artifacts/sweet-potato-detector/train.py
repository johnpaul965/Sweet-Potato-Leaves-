"""
Training script for Sweet Potato Leaf Health Detection System.
Fine-tunes MobileNetV2 on the sorted dataset using transfer learning.

Class mapping (must match model.py CLASS_NAMES):
  0 → Healthy            (folder: Healthy)
  1 → Leaf Curl Virus    (folder: Leaf_Curl_Virus)
  2 → Fusarium Wilt      (folder: Fusarium_Wilt)
  3 → Cercospora Leaf Spot (folder: Cercospora_Leaf_Spot)
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, Input
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.utils.class_weight import compute_class_weight

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(__file__)
DATASET_DIR = os.path.join(BASE_DIR, "sorted_dataset_jpeg", "sorted_dataset")
MODEL_OUT   = os.path.join(BASE_DIR, "sweet_potato_model.h5")

# ── Config ─────────────────────────────────────────────────────────────────────
IMG_SIZE    = (224, 224)
BATCH_SIZE  = 16
EPOCHS_P1   = 20   # frozen base
EPOCHS_P2   = 15   # fine-tune last 30 layers
SEED        = 42

# Folder order MUST match model.py CLASS_NAMES order
CLASSES = ["Healthy", "Leaf_Curl_Virus", "Fusarium_Wilt", "Cercospora_Leaf_Spot"]

print("=" * 60)
print("Sweet Potato Leaf Health Detection — Training")
print("=" * 60)
print(f"Dataset  : {DATASET_DIR}")
print(f"Output   : {MODEL_OUT}")
print(f"Classes  : {CLASSES}")
print()

# ── Data generators ────────────────────────────────────────────────────────────
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    validation_split=0.2,
    rotation_range=30,
    width_shift_range=0.15,
    height_shift_range=0.15,
    shear_range=0.15,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    brightness_range=[0.75, 1.25],
    fill_mode="reflect",
)

val_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    validation_split=0.2,
)

train_gen = train_datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    classes=CLASSES,
    class_mode="categorical",
    subset="training",
    seed=SEED,
    shuffle=True,
)

val_gen = val_datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    classes=CLASSES,
    class_mode="categorical",
    subset="validation",
    seed=SEED,
    shuffle=False,
)

print(f"\nTraining samples  : {train_gen.samples}")
print(f"Validation samples: {val_gen.samples}")
print(f"Class indices     : {train_gen.class_indices}")

# ── Class weights (handle imbalance) ──────────────────────────────────────────
labels = train_gen.classes
cw = compute_class_weight("balanced", classes=np.unique(labels), y=labels)
class_weights = dict(enumerate(cw))
print(f"\nClass weights: {class_weights}")

# ── Build model ────────────────────────────────────────────────────────────────
inputs    = Input(shape=(224, 224, 3))
base      = MobileNetV2(weights="imagenet", include_top=False, input_tensor=inputs)
base.trainable = False   # Phase 1: freeze all base layers

x       = base.output
x       = GlobalAveragePooling2D()(x)
x       = Dropout(0.3)(x)
outputs = Dense(4, activation="softmax", name="predictions")(x)
model   = Model(inputs=base.input, outputs=outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

print(f"\nModel parameters: {model.count_params():,}")
print(f"Trainable params (phase 1): {sum(v.numpy().size for v in model.trainable_variables):,}")

# ── Phase 1: Train top layers only ────────────────────────────────────────────
print("\n--- Phase 1: Training classifier head (base frozen) ---")

callbacks_p1 = [
    ModelCheckpoint(MODEL_OUT, monitor="val_accuracy", save_best_only=True,
                    verbose=1, mode="max"),
    EarlyStopping(monitor="val_accuracy", patience=8, restore_best_weights=True,
                  verbose=1),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4,
                      min_lr=1e-6, verbose=1),
]

history1 = model.fit(
    train_gen,
    epochs=EPOCHS_P1,
    validation_data=val_gen,
    class_weight=class_weights,
    callbacks=callbacks_p1,
    verbose=1,
)

# ── Phase 2: Fine-tune last 30 layers ─────────────────────────────────────────
print("\n--- Phase 2: Fine-tuning last 30 layers ---")

base.trainable = True
for layer in base.layers[:-30]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

callbacks_p2 = [
    ModelCheckpoint(MODEL_OUT, monitor="val_accuracy", save_best_only=True,
                    verbose=1, mode="max"),
    EarlyStopping(monitor="val_accuracy", patience=10, restore_best_weights=True,
                  verbose=1),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5,
                      min_lr=1e-7, verbose=1),
]

history2 = model.fit(
    train_gen,
    epochs=EPOCHS_P2,
    validation_data=val_gen,
    class_weight=class_weights,
    callbacks=callbacks_p2,
    verbose=1,
)

# ── Evaluate on validation set ────────────────────────────────────────────────
print("\n--- Final Evaluation ---")
val_gen.reset()
loss, acc = model.evaluate(val_gen, verbose=1)
print(f"\nValidation Accuracy : {acc:.4f} ({acc*100:.2f}%)")
print(f"Validation Loss     : {loss:.4f}")

# ── Per-class metrics ─────────────────────────────────────────────────────────
from sklearn.metrics import classification_report, confusion_matrix

val_gen.reset()
y_pred = model.predict(val_gen, verbose=1)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = val_gen.classes

class_labels = [
    "Healthy",
    "Sweet Potato Leaf Curl Virus",
    "Fusarium Wilt",
    "Cercospora Leaf Spot",
]

print("\nClassification Report:")
print(classification_report(y_true, y_pred_classes, target_names=class_labels))

print("Confusion Matrix:")
print(confusion_matrix(y_true, y_pred_classes))

print(f"\n✅ Model saved to: {MODEL_OUT}")
print("Place sweet_potato_model.h5 in the app folder and restart Streamlit.")
