# 🩻 CardiX - Chest X-Ray Diagnosis with Deep Learning

**CardiX** is an AI-powered diagnostic tool that classifies chest X-ray images into four categories: **COVID-19**, **Normal**, **Pneumonia**, and **Lung Opacity**. It uses a custom-built deep Convolutional Neural Network (CNN) and features a responsive frontend and backend for real-time medical image analysis.

---

## 🚀 Results

- ✅ **94.91% Accuracy** on combined **training + validation** dataset *(26,309 images)*
- ✅ **96.35% Accuracy** on a **completely unseen test set** *(1,288 images)*
- 🧠 Built **entirely from scratch** — no pre-trained models
- 💻 Optimized for Apple Silicon (M1/M2) using PyTorch’s MPS backend

---

## 📚 Datasets Used

CardiX uses data aggregated from the following publicly available sources:

1. [COVID-19 Radiography Database](https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database)
2. [Chest X-ray: COVID19, Pneumonia, Normal Dataset](https://www.kaggle.com/datasets/prashant268/chest-xray-covid19-pneumonia)

> Both datasets were cleaned and balanced before training.

---

## 🧠 Model Overview

- Custom CNN architecture with:
  - **Batch Normalization**
  - **Dropout Regularization**
  - **Adaptive Average Pooling**
  - **Multi-layer Dense Classifier Head**
- Trained using:
  - **CrossEntropyLoss**
  - **Adam Optimizer**
  - **Learning Rate Scheduler**
  - **Early Stopping** on 95% validation accuracy
- Data Augmentation:
  - Random Rotation
  - Horizontal Flip
  - Gaussian Blur

---

## 🖥️ Full-Stack Web Application

- **Frontend**: React.js 
- **Backend**: FastAPI (Python)
- **Functionality**:
  - Upload X-ray image
  - Model predicts class label
  - Displays real-time prediction & diagnosis

---


