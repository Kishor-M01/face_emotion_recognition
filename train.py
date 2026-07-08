import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import matplotlib.pyplot as plt

# ── Config ──────────────────────────────────────────────────────────────────
DATA_DIR   = "data"                  # expects data/train & data/test
IMG_SIZE   = 96                      # MobileNetV2 min is 96
BATCH      = 64
EPOCHS     = 50
EMOTIONS   = ['angry','disgust','fear','happy','neutral','sad','surprise']
NUM_CLASSES = len(EMOTIONS)
MODEL_PATH = "emotion_model.keras"

# ── Data generators ──────────────────────────────────────────────────────────
train_gen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    zoom_range=0.1,
)
val_gen = ImageDataGenerator(rescale=1./255)

train_ds = train_gen.flow_from_directory(
    os.path.join(DATA_DIR, "train"),
    target_size=(IMG_SIZE, IMG_SIZE),
    color_mode="rgb",
    batch_size=BATCH,
    class_mode="categorical",
    shuffle=True,
)
val_ds = val_gen.flow_from_directory(
    os.path.join(DATA_DIR, "test"),
    target_size=(IMG_SIZE, IMG_SIZE),
    color_mode="rgb",
    batch_size=BATCH,
    class_mode="categorical",
    shuffle=False,
)

# ── Model: MobileNetV2 + custom head ─────────────────────────────────────────
def build_model():
    base = MobileNetV2(input_shape=(IMG_SIZE, IMG_SIZE, 3),
                       include_top=False, weights="imagenet")
    # Freeze base initially
    base.trainable = False

    inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = base(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)
    return Model(inputs, outputs), base

model, base = build_model()
model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
              loss="categorical_crossentropy",
              metrics=["accuracy"])
model.summary()

callbacks = [
    EarlyStopping(patience=7, restore_best_weights=True, monitor="val_accuracy"),
    ReduceLROnPlateau(factor=0.5, patience=3, monitor="val_loss"),
    ModelCheckpoint(MODEL_PATH, save_best_only=True, monitor="val_accuracy"),
]

# ── Phase 1: Train head only ──────────────────────────────────────────────────
print("\n=== Phase 1: Training head ===")
history1 = model.fit(train_ds, validation_data=val_ds,
                     epochs=20, callbacks=callbacks)

# ── Phase 2: Fine-tune top layers of base ────────────────────────────────────
print("\n=== Phase 2: Fine-tuning ===")
base.trainable = True
# Freeze all except last 30 layers
for layer in base.layers[:-30]:
    layer.trainable = False

model.compile(optimizer=tf.keras.optimizers.Adam(1e-4),
              loss="categorical_crossentropy",
              metrics=["accuracy"])

history2 = model.fit(train_ds, validation_data=val_ds,
                     epochs=EPOCHS, callbacks=callbacks)

# ── Plot ──────────────────────────────────────────────────────────────────────
def plot_history(h1, h2):
    acc  = h1.history["accuracy"]  + h2.history["accuracy"]
    vacc = h1.history["val_accuracy"] + h2.history["val_accuracy"]
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(acc, label="train"); plt.plot(vacc, label="val")
    plt.title("Accuracy"); plt.legend()

    loss  = h1.history["loss"]  + h2.history["loss"]
    vloss = h1.history["val_loss"] + h2.history["val_loss"]
    plt.subplot(1, 2, 2)
    plt.plot(loss, label="train"); plt.plot(vloss, label="val")
    plt.title("Loss"); plt.legend()
    plt.tight_layout()
    plt.savefig("training_history.png")
    plt.show()

plot_history(history1, history2)
print(f"\nModel saved to {MODEL_PATH}")
