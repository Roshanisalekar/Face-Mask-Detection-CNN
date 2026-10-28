import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import json

# ==========================
# Dataset Path
# ==========================
dataset_path = "data"

# ==========================
# Training Data
# ==========================
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    validation_split=0.2,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True
)

train_data = train_datagen.flow_from_directory(
    dataset_path,
    target_size=(128, 128),
    batch_size=32,
    class_mode="categorical",
    subset="training",
    shuffle=True
)

# ==========================
# Validation Data
# ==========================
validation_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    validation_split=0.2
)

validation_data = validation_datagen.flow_from_directory(
    dataset_path,
    target_size=(128, 128),
    batch_size=32,
    class_mode="categorical",
    subset="validation",
    shuffle=False
)

# ==========================
# Classes
# ==========================
print("\nClass Indices:")
print(train_data.class_indices)

class_names = list(train_data.class_indices.keys())

with open("classes.json", "w") as f:
    json.dump(class_names, f)

print("Classes:", class_names)

# ==========================
# CNN Model
# ==========================
model = Sequential([
    
    Conv2D(
        32,
        (3, 3),
        activation="relu",
        input_shape=(128, 128, 3)
    ),
    MaxPooling2D((2, 2)),

    Conv2D(
        64,
        (3, 3),
        activation="relu"
    ),
    MaxPooling2D((2, 2)),

    Conv2D(
        128,
        (3, 3),
        activation="relu"
    ),
    MaxPooling2D((2, 2)),

    Flatten(),

    Dense(
        256,
        activation="relu"
    ),

    Dropout(0.5),

    Dense(
        train_data.num_classes,
        activation="softmax"
    )
])

# ==========================
# Compile
# ==========================
model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# ==========================
# Model Summary
# ==========================
model.summary()

# ==========================
# Training
# ==========================
print("\nTraining Started...\n")

history = model.fit(
    train_data,
    validation_data=validation_data,
    epochs=10
)

# ==========================
# Save Model
# ==========================
model.save("face_mask_model.keras")

print("\n==============================")
print("Model Saved Successfully!")
print("==============================")
print("Model: face_mask_model.keras")
print("Classes: classes.json")