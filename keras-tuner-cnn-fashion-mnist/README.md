# Fashion MNIST Classification using CNN with Keras Tuner

This project builds and optimizes a Convolutional Neural Network (CNN) to classify Fashion MNIST images using TensorFlow and Keras Tuner.

## 📌 Features
- Uses TensorFlow/Keras for deep learning
- Hyperparameter optimization with Keras Tuner (`RandomSearch`)
- Validation accuracy up to **91.6%**
- Automatically searches best:
  - Number of filters in Conv layers
  - Kernel size
  - Dense units
  - Learning rate

## 📊 Dataset
**Fashion MNIST** (60,000 training + 10,000 testing images, 28x28 grayscale)

## ⚙️ Technologies Used
- Python
- TensorFlow / Keras
- Keras Tuner
- NumPy

## 🧪 Results
Best model summary:
- Conv2D: 80 filters, 3x3 kernel
- Conv2D: 32 filters, 5x5 kernel
- Dense: 96 units
- Learning Rate: 0.001  
→ **Val Accuracy:** 91.6%

## 🚀 How to Run

```bash
git clone https://github.com/yourusername/keras-tuner-cnn-fashion-mnist.git
cd keras-tuner-cnn-fashion-mnist
pip install -r requirements.txt
python cnn_keras_tuner.py
