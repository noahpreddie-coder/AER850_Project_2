# part_2_1_data.py
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
    layers.RandomRotation(0.05),
    layers.RandomZoom(0.15),
    layers.RandomTranslation(0.1, 0.1),
])

train_ds = train_ds.map(lambda x, y: (augment(rescale(x)), y))
valid_ds = valid_ds.map(lambda x, y: (rescale(x), y))
test_ds  = test_ds.map(lambda x, y: (rescale(x), y))

# performance
train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
valid_ds = valid_ds.prefetch(tf.data.AUTOTUNE)
test_ds  = test_ds.prefetch(tf.data.AUTOTUNE)

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

