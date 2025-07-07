from pathlib import Path
import tensorflow as tf
import numpy as np
import cv2

# ── EDIT THESE TWO PATHS ───────────────────────────────────
img_path   = Path(r"C:\Users\Asus\Desktop\Fire Detection\fire_dataset\no_fire\non_fire.178.png")
model_path = Path(r"C:\Users\Asus\Desktop\Fire Detection\fire_exp\models\final_full_model.keras")
# ───────────────────────────────────────────────────────────

IMG_SIZE = (128, 128)
THRESH   = 0.5

# 1. Load model
model = tf.keras.models.load_model(model_path)

# 2. Preprocess image
img_bgr = cv2.imread(str(img_path))
if img_bgr is None:
    raise FileNotFoundError(img_path)
img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, IMG_SIZE)
img = img.astype("float32") / 255.0
img = np.expand_dims(img, axis=0)   # shape (1, 128, 128, 3)

# 3. Predict
prob = model.predict(img)[0, 0]
label = "🔥 FIRE DETECTED" if prob > THRESH else "✅ No Fire"
print(f"{img_path.name}: {label}  (prob = {prob:.3f})")
