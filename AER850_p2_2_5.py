# part_2_5_test_model1.py
import os, json, numpy as np, tensorflow as tf
import matplotlib.pyplot as plt
from glob import glob
from tensorflow.keras.preprocessing.image import load_img, img_to_array

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

#model 1
DATA_ROOT = "Data"
IMG_SIZE = (500, 500) 
BATCH_SIZE = 32
MODEL_PATH = "models/model1/best.h5"
CLASS_META_PATH = "models/meta/class_indices.json"  # mapping used during training
OUTPUT_DIR = "outputs/model1"
# ==========================================================================

# ---- Load label map ---
with open(CLASS_META_PATH, "r") as f:
    class_indices = json.load(f)  # e.g. {"crack":0,"missing-head":1,"paint-off":2}
idx_to_class = {v: k for k, v in class_indices.items()}
CLASS_NAMES = [c for c, _ in sorted(class_indices.items(), key=lambda x: x[1])]

# ---- Load trained Model 1 ----
model = tf.keras.models.load_model(MODEL_PATH)

# ---- Build test dataset (for accuracy/loss only) ----
test_dir = os.path.join(DATA_ROOT, "test")
test_ds = (
    tf.keras.utils.image_dataset_from_directory(
        test_dir, labels="inferred", label_mode="categorical",
        class_names=CLASS_NAMES, image_size=IMG_SIZE, batch_size=BATCH_SIZE,
        shuffle=False
    )
    .map(lambda x, y: (x/255.0, y))
    .prefetch(tf.data.AUTOTUNE)
)

loss, acc = model.evaluate(test_ds, verbose=0)
print(f"Test accuracy: {acc:.3f} | Test loss: {loss:.3f}")

def preprocess(path):
    img = load_img(path, target_size=IMG_SIZE, color_mode="rgb")
    x = img_to_array(img) / 255.0
    return np.expand_dims(x, 0), img  # (1,H,W,3), PIL image

# NEW: normalize model outputs to a probability distribution if needed
def to_probs(raw_vec):
    raw = np.asarray(raw_vec).ravel()
    s = float(np.sum(raw))
    # if already looks like probs that sum to ~1, keep as-is
    if np.all(raw >= 0) and np.all(raw <= 1) and 0.98 <= s <= 1.02:
        return raw
    # otherwise treat as logits/sigmoids and softmax them
    m = np.max(raw)
    exps = np.exp(raw - m)
    return exps / np.sum(exps)

def draw_overlay(pil_img, true_label, probs, save_path, pred_label, pred_conf):
    """
    Overlays:
      - Title lines: True label / Predicted label
      - On-image, bottom-left: per-class probabilities in green
    """
    plt.figure(figsize=(6.5, 6.8))
    plt.suptitle(
        f"True Crack Classification Label: {true_label}\n"
        f"Predicted Crack Classification Label: {pred_label}",
        fontsize=12, y=0.98
    )

    plt.imshow(pil_img)
    plt.axis("off")

    lines = []
    for i in range(len(probs)):
        cls = idx_to_class[i]
        lines.append(f"{cls.replace('-', ' ').title()}: {probs[i]*100:.1f}%")
    txt = "\n".join(lines)

    ax = plt.gca()
    ax.text(
        0.03, 0.06, txt,
        transform=ax.transAxes,
        fontsize=18, color="green",
        va="bottom", ha="left",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="none", edgecolor="none")
    )

    ax.text(
        0.03, 0.01, f"Top-1: {pred_label} ({pred_conf*100:.1f}%)",
        transform=ax.transAxes, fontsize=10, color="black",
        va="bottom", ha="left"
    )

    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

# ---- Pick one test image per class and visualize predictions with overlays ----
os.makedirs(OUTPUT_DIR, exist_ok=True)

samples = []
for cls in CLASS_NAMES:
    files = sorted(
        [p for p in glob(os.path.join(test_dir, cls, "*"))
         if p.lower().endswith((".jpg", ".jpeg", ".png"))]
    )
    if files:
        samples.append(files[0])  # first example of each class

for path in samples:
    true_label = os.path.basename(os.path.dirname(path))  # folder name == ground truth
    x, pil_img = preprocess(path)

    raw = model.predict(x, verbose=0)[0]       # shape (num_classes,)
    probs = to_probs(raw)                      # <-- normalize here
    pred_idx = int(np.argmax(probs))
    pred_label = idx_to_class[pred_idx]
    pred_conf = float(probs[pred_idx])

    base = os.path.splitext(os.path.basename(path))[0]
    out_path = os.path.join(OUTPUT_DIR, f"pred_{base}.png")
    draw_overlay(pil_img, true_label, probs, out_path, pred_label, pred_conf)

    print(f"{os.path.basename(path)} | true={true_label} -> pred={pred_label} ({pred_conf:.2f}) | saved: {out_path}")

print(f"Saved labeled images with overlays to {OUTPUT_DIR}/")
