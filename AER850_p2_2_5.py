# part_2_5_test
import os, json, numpy as np, tensorflow as tf
import matplotlib.pyplot as plt
from glob import glob
from tensorflow.keras.preprocessing.image import load_img, img_to_array

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

DATA_ROOT = "Data"
IMG_SIZE = (500, 500)
BATCH_SIZE = 32
CLASS_NAMES = ["crack", "missing-head", "paint-off"]

# load label map
with open("models/meta/class_indices.json", "r") as f:
    class_indices = json.load(f)
idx_to_class = {v: k for k, v in class_indices.items()}

# load trained model
model = tf.keras.models.load_model("models/model1/best.h5")

# test dataset (for accuracy)
test_dir = os.path.join(DATA_ROOT, "test")
test_ds = tf.keras.utils.image_dataset_from_directory(
    test_dir, labels="inferred", label_mode="categorical",
    class_names=CLASS_NAMES, image_size=IMG_SIZE, batch_size=BATCH_SIZE,
    shuffle=False
).map(lambda x, y: (x/255.0, y)).prefetch(tf.data.AUTOTUNE)

loss, acc = model.evaluate(test_ds, verbose=0)
print(f"Test accuracy: {acc:.3f} | Test loss: {loss:.3f}")

# pick one test image per class and visualize predictions
os.makedirs("outputs/model1", exist_ok=True)

def preprocess(path):
    img = load_img(path, target_size=IMG_SIZE)
    x = img_to_array(img) / 255.0
    return np.expand_dims(x, 0), img

samples = []
for cls in CLASS_NAMES:
    files = sorted(
        [p for p in glob(os.path.join(test_dir, cls, "*"))
         if p.lower().endswith((".jpg", ".jpeg", ".png"))]
    )
    if files:
        samples.append(files[0])

for path in samples:
    x, pil_img = preprocess(path)
    probs = model.predict(x, verbose=0)[0]
    pred_idx = int(np.argmax(probs))
    pred_label = idx_to_class[pred_idx]
    conf = float(probs[pred_idx])

    plt.figure()
    plt.imshow(pil_img); plt.axis("off")
    plt.title(f"Pred: {pred_label}  ({conf:.2f})")
    base = os.path.splitext(os.path.basename(path))[0]
    out = f"outputs/model1/pred_{base}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()

    print(f"{os.path.basename(path)} -> {pred_label} ({conf:.2f})")

print("Saved labeled images to outputs/model1/")
