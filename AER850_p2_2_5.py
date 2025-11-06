# part_2_5_test_model2.py
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

# ---- Load label map exactly as model was trained ----
with open("models/meta/class_indices.json", "r") as f:
    class_indices = json.load(f)                  # e.g. {"crack":0,"missing-head":1,"paint-off":2}
idx_to_class = {v: k for k, v in class_indices.items()}
CLASS_NAMES = [c for c, _ in sorted(class_indices.items(), key=lambda x: x[1])]  # ordered by index

# ---- Load trained Model 2 ----
model = tf.keras.models.load_model("models/model2/best.h5")

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

# ---- Helpers ----
def preprocess(path):
    img = load_img(path, target_size=IMG_SIZE, color_mode="rgb")
    x = img_to_array(img) / 255.0
    return np.expand_dims(x, 0), img  # (1,H,W,3), PIL image (for plotting)

def draw_overlay(pil_img, true_label, probs, save_path, pred_label, pred_conf):
    """
    Overlays:
      - Title lines: True label / Predicted label
      - On-image, bottom-left: per-class probabilities in green (like your example)
    """
    # Figure sized to keep text readable in saved PNG
    plt.figure(figsize=(6.5, 6.8))
    # Title block (two lines)
    plt.suptitle(
        f"True Crack Classification Label: {true_label}\n"
        f"Predicted Crack Classification Label: {pred_label}",
        fontsize=12, y=0.98
    )

    # Show image
    plt.imshow(pil_img)
    plt.axis("off")

    # Compose per-class lines (sorted by class_indices order)
    lines = []
    for i in range(len(probs)):
        cls = idx_to_class[i]
        lines.append(f"{cls.replace('-', ' ').title()}: {probs[i]*100:.1f}%")

    # Draw multiline text near bottom-left
    txt = "\n".join(lines)
    # Position in axes fraction coords (0..1), nudge up from bottom
    plt.gca().text(
        0.03, 0.06, txt,
        transform=plt.gca().transAxes,
        fontsize=18, color="green",
        va="bottom", ha="left",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="none", edgecolor="none")  # no box, like example
    )

    # Small caption-like footnote with top-1 confidence if you want it
    plt.gca().text(
        0.03, 0.01, f"Top-1: {pred_label} ({pred_conf*100:.1f}%)",
        transform=plt.gca().transAxes, fontsize=10, color="black",
        va="bottom", ha="left"
    )

    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

# ---- Pick one test image per class and visualize predictions with overlays ----
os.makedirs("outputs/model2", exist_ok=True)

samples = []
for cls in CLASS_NAMES:
    files = sorted(
        [p for p in glob(os.path.join(test_dir, cls, "*"))
         if p.lower().endswith((".jpg", ".jpeg", ".png"))]
    )
    if files:
        samples.append(files[0])   # take first example of each class

for path in samples:
    true_label = os.path.basename(os.path.dirname(path))  # folder name == ground truth
    x, pil_img = preprocess(path)

    probs = model.predict(x, verbose=0)[0]               # shape (num_classes,)
    pred_idx = int(np.argmax(probs))
    pred_label = idx_to_class[pred_idx]
    pred_conf = float(probs[pred_idx])

    base = os.path.splitext(os.path.basename(path))[0]
    out_path = f"outputs/model2/pred_{base}.png"
    draw_overlay(pil_img, true_label, probs, out_path, pred_label, pred_conf)

    print(f"{os.path.basename(path)} | true={true_label} -> pred={pred_label} ({pred_conf:.2f}) | saved: {out_path}")

print("Saved labeled images with overlays to outputs/model2/")
