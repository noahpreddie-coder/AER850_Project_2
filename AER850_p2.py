# AER850_p2.py

# --- 2.1: data loading and preprocessing ---
import os, json, numpy as np, tensorflow as tf
from tensorflow.keras.utils import image_dataset_from_directory
from tensorflow.keras import layers

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

DATA_ROOT = "Data"
IMG_SIZE = (500, 500)
BATCH_SIZE = 32
CLASS_NAMES = ["crack", "missing-head", "paint-off"]

train_dir = os.path.join(DATA_ROOT, "train")
valid_dir = os.path.join(DATA_ROOT, "valid")
test_dir  = os.path.join(DATA_ROOT, "test")

# datasets
train_ds = image_dataset_from_directory(
    train_dir, labels="inferred", label_mode="categorical",
    class_names=CLASS_NAMES, image_size=IMG_SIZE, batch_size=BATCH_SIZE,
    shuffle=True, seed=SEED
)
valid_ds = image_dataset_from_directory(
    valid_dir, labels="inferred", label_mode="categorical",
    class_names=CLASS_NAMES, image_size=IMG_SIZE, batch_size=BATCH_SIZE,
    shuffle=False
)
test_ds = image_dataset_from_directory(
    test_dir, labels="inferred", label_mode="categorical",
    class_names=CLASS_NAMES, image_size=IMG_SIZE, batch_size=BATCH_SIZE,
    shuffle=False
)

# preprocessing/augmentation
rescale = layers.Rescaling(1./255)
augment = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.02),
    layers.RandomZoom(0.08),
    layers.RandomTranslation(0.04, 0.04),
])

train_ds = train_ds.map(lambda x, y: (augment(rescale(x)), y),
                        num_parallel_calls=tf.data.AUTOTUNE)
valid_ds = valid_ds.map(lambda x, y: (rescale(x), y),
                        num_parallel_calls=tf.data.AUTOTUNE)
test_ds  = test_ds.map(lambda x, y: (rescale(x), y),
                        num_parallel_calls=tf.data.AUTOTUNE)

# pipeline performance
train_ds = train_ds.cache().prefetch(tf.data.AUTOTUNE)
valid_ds = valid_ds.cache().prefetch(tf.data.AUTOTUNE)
test_ds  = test_ds.cache().prefetch(tf.data.AUTOTUNE)

# summary and class map
def count_images(folder):
    return sum(len(files) for _, _, files in os.walk(folder))

print("\n=== Dataset Summary ===")
print("Train samples     :", count_images(train_dir))
print("Validation samples:", count_images(valid_dir))
print("Test samples      :", count_images(test_dir))
print("Image size        :", IMG_SIZE, "channels=3")
print("Batch size        :", BATCH_SIZE)
print("Classes (order)   :", CLASS_NAMES)

os.makedirs("models/meta", exist_ok=True)
class_indices = {name: i for i, name in enumerate(CLASS_NAMES)}
with open("models/meta/class_indices.json", "w") as f:
    json.dump(class_indices, f, indent=2)
print('Saved class indices to models/meta/class_indices.json')

# --- 2.2–2.4: Model 1 (baseline CNN), training, and plots ---
from tensorflow.keras import models
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
import matplotlib.pyplot as plt

# model
base_filters = 24
dropout_rate = 0.5

model = models.Sequential([
    layers.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),

    layers.Conv2D(base_filters, 3, padding="same", activation="relu"),
    layers.Conv2D(base_filters, 3, padding="same", activation="relu"),
    layers.MaxPooling2D(),

    layers.Conv2D(base_filters*2, 3, padding="same", activation="relu"),
    layers.Conv2D(base_filters*2, 3, padding="same", activation="relu"),
    layers.MaxPooling2D(),

    layers.Conv2D(base_filters*4, 3, padding="same", activation="relu"),
    layers.Conv2D(base_filters*4, 3, padding="same", activation="relu"),
    layers.MaxPooling2D(),

    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(dropout_rate),
    layers.Dense(3, activation="softmax")
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
    steps_per_execution=64
)

os.makedirs("models/model1", exist_ok=True)
os.makedirs("plots", exist_ok=True)

callbacks = [
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6),
    ModelCheckpoint("models/model1/best.h5", monitor="val_accuracy", save_best_only=True)
]

history = model.fit(
    train_ds,
    validation_data=valid_ds,
    epochs=20,
    callbacks=callbacks
)

# plots
plt.figure()
plt.plot(history.history["accuracy"], label="train_acc")
plt.plot(history.history["val_accuracy"], label="val_acc")
plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.legend(); plt.title("Model 1 Accuracy")
plt.savefig("plots/model1_acc.png", dpi=150, bbox_inches="tight")
plt.close()

plt.figure()
plt.plot(history.history["loss"], label="train_loss")
plt.plot(history.history["val_loss"], label="val_loss")
plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend(); plt.title("Model 1 Loss")
plt.savefig("plots/model1_loss.png", dpi=150, bbox_inches="tight")
plt.close()

print("Saved best weights to models/model1/best.h5")
print("Saved plots to plots/model1_acc.png and plots/model1_loss.png")
