import tensorflow as tf
import numpy as np
import cv2
import json

# ==========================
# Load Model
# ==========================
model = tf.keras.models.load_model("face_mask_model.keras")

# ==========================
# Load Classes
# ==========================
with open("classes.json", "r") as f:
    class_names = json.load(f)

print("Classes:", class_names)

# ==========================
# Image Path
# ==========================
image_path = "test.jpg"

# ==========================
# Read Image
# ==========================
image = cv2.imread(image_path)

if image is None:
    print("Image not found!")
    exit()

# ==========================
# Resize Image
# ==========================
img = cv2.resize(image, (128, 128))

# ==========================
# BGR to RGB
# ==========================
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# ==========================
# Normalize
# ==========================
img = img.astype("float32") / 255.0

# ==========================
# Add Batch Dimension
# ==========================
img = np.expand_dims(img, axis=0)

# ==========================
# Prediction
# ==========================
prediction = model.predict(img, verbose=0)

print("\nPrediction Probabilities:")
print(prediction)

# ==========================
# Get Result
# ==========================
index = np.argmax(prediction[0])

label = class_names[index]

confidence = prediction[0][index] * 100

# ==========================
# Print Result
# ==========================
print("\n==============================")
print("Prediction :", label)
print("Confidence :", round(confidence, 2), "%")
print("==============================")

# ==========================
# Display Result
# ==========================
cv2.putText(
    image,
    f"{label} ({confidence:.2f}%)",
    (20, 40),
    cv2.FONT_HERSHEY_SIMPLEX,
    1,
    (0, 255, 0),
    2
)

cv2.imshow("Face Mask Prediction", image)

cv2.waitKey(0)
cv2.destroyAllWindows()