# Face Mask Detection Using CNN

## 📌 Project Overview

Face Mask Detection is a Deep Learning project that uses a **Convolutional Neural Network (CNN)** to classify whether a person is wearing a face mask or not.

The model predicts two classes:

- 😷 **With Mask**
- ❌ **Without Mask**

The project uses **TensorFlow/Keras** for model training and **OpenCV** for image prediction.

---

## 🎯 Project Objective

The main objective of this project is to build an image classification model that can automatically detect whether a person is wearing a face mask.

---

## 🛠️ Technologies Used

- Python
- TensorFlow
- Keras
- CNN (Convolutional Neural Network)
- OpenCV
- NumPy
- Matplotlib
- ImageDataGenerator

---

## 🔄 Machine Learning Workflow

**Dataset → Image Preprocessing → Data Augmentation → CNN Model → Training → Prediction → Confidence Score**

### 1. Image Preprocessing

Images are resized to:

**128 × 128 pixels**

Pixel values are normalized using:

**1 / 255**

Data augmentation is also applied using:

- Rotation
- Zoom
- Horizontal Flip

### 2. CNN Model

The model contains:

- Conv2D
- MaxPooling2D
- Conv2D
- MaxPooling2D
- Conv2D
- MaxPooling2D
- Flatten
- Dense Layer
- Dropout
- Softmax Output Layer

The output layer predicts two classes:

**With Mask / Without Mask**

### 3. Model Training

The model is trained using:

- Optimizer: **Adam**
- Loss Function: **Categorical Crossentropy**
- Epochs: **10**
- Batch Size: **32**
- Validation Split: **20%**

The trained model is saved as:

`face_mask_model.keras`

Class labels are saved in:

`classes.json`

---

## 🔍 Prediction

The `predict.py` file:

1. Loads the trained CNN model.
2. Loads the class labels.
3. Reads the test image.
4. Resizes the image to 128 × 128.
5. Converts the image into the required format.
6. Normalizes pixel values.
7. Predicts the class.
8. Displays the prediction with confidence score.

Example:

**Prediction: With Mask**

**Confidence: XX%**

---

## 📁 Project Structure

```text
Face-Mask-Detection-CNN/
│
├── train.py
├── predict.py
├── classes.json
├── test.jpg
├── README.md
└── face_mask_model.keras
```

> The trained `.keras` model is provided separately because of GitHub file-size limitations.

---

## 🚀 How to Run

### Step 1: Install Required Libraries

```bash
pip install tensorflow opencv-python numpy matplotlib
```

### Step 2: Train the Model

```bash
python train.py
```

This will create:

```text
face_mask_model.keras
classes.json
```

### Step 3: Add Test Image

Place your test image in the project folder with the name:

```text
test.jpg
```

### Step 4: Run Prediction

```bash
python predict.py
```

The program will display the predicted class and confidence score.

---

## 📊 Classes

| Class | Description |
|---|---|
| With Mask | Person is wearing a face mask |
| Without Mask | Person is not wearing a face mask |

---

## 💡 Key Learning

Through this project, I learned:

- Image preprocessing
- Data augmentation
- CNN architecture
- Image classification
- Model training
- Model prediction
- Confidence score
- TensorFlow/Keras implementation
- OpenCV image processing

---

## 🎯 Project Goal

The goal of this project is to demonstrate the practical implementation of **Deep Learning and CNN for image classification**.

---

## 👩‍💻 Author

**Roshani Salekar**

Data Science | Machine Learning | AI

GitHub:  
https://github.com/Roshanisalekar

drive link data :
https://drive.google.com/file/d/1v2asbi8ognfVxlKRXWJ7EjV6GHVBLVQD/view?usp=sharing

