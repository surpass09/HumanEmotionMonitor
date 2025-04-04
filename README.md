# 🧠 Human Emotion Prediction Model (WIP)

A deep learning project that detects **human emotions** from facial expressions using **PyTorch**, **TensorFlow**, and a layered CNN architecture. Built for fun, learning, and to push boundaries in human-centered AI.

![Status](https://img.shields.io/badge/status-in%20progress-yellow?style=for-the-badge)
![PyTorch](https://img.shields.io/badge/framework-pytorch-red?style=for-the-badge&logo=pytorch)
![TensorFlow](https://img.shields.io/badge/also_using-tensorflow-orange?style=for-the-badge&logo=tensorflow)
![Emotions](https://img.shields.io/badge/emotions-happy%20%7C%20sad%20%7C%20angry-blue?style=for-the-badge)

---

## 🚀 Overview

This repo is a work-in-progress human emotion prediction model. The idea is simple: feed in an image of a face, and the model will tell you what emotion it sees — happy, sad, angry, etc.

Currently using **CNNs** with **PyTorch**, but also experimenting with **TensorFlow/Keras** to test different architectures.

---

## ⚙️ Tech Stack

| Tool           | Purpose                         |
|----------------|---------------------------------|
| PyTorch        | Core model building             |
| TensorFlow     | Alternate model experiments     |
| OpenCV         | Image preprocessing + face detection |
| NumPy & Pandas | Data handling                   |
| Scikit-learn   | Metrics + confusion matrix      |
| Matplotlib     | Training visualization          |

---

## 🧠 Model Architecture (Early Stage)

I'm stacking layers like this (in PyTorch):

```python
Conv2d ➡ ReLU ➡ BatchNorm ➡ MaxPooling ➡ Dropout
(Repeat 3-4x) ➡ Flatten ➡ Dense ➡ Softmax
