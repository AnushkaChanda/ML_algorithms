# cnn_keras_tuner.py
# Fashion MNIST classification using CNN and Keras Tuner

import tensorflow as tf
from tensorflow import keras
import numpy as np

from kerastuner import RandomSearch
from kerastuner.engine.hyperparameters import HyperParameters

print("TensorFlow Version:", tf.__version__)

# Load and preprocess Fashion MNIST dataset
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

train_images = train_images / 255.0
test_images = test_images / 255.0

# Reshape images to fit CNN input
train_images = train_images.reshape(len(train_images), 28, 28, 1)
test_images = test_images.reshape(len(test_images), 28, 28, 1)

# Build model with hyperparameters
def build_model(hp):  
    model = keras.Sequential([
        keras.layers.Conv2D(
            filters=hp.Int('conv_1_filter', min_value=32, max_value=128, step=16),
            kernel_size=hp.Choice('conv_1_kernel', values=[3, 5]),
            activation='relu',
            input_shape=(28, 28, 1)
        ),
        keras.layers.Conv2D(
            filters=hp.Int('conv_2_filter', min_value=32, max_value=64, step=16),
            kernel_size=hp.Choice('conv_2_kernel', values=[3, 5]),
            activation='relu'
        ),
        keras.layers.Flatten(),
        keras.layers.Dense(
            units=hp.Int('dense_1_units', min_value=32, max_value=128, step=16),
            activation='relu'
        ),
        keras.layers.Dense(10, activation='softmax')
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(
            hp.Choice('learning_rate', values=[1e-2, 1e-3])
        ),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# Define tuner
tuner_search = RandomSearch(
    build_model,
    objective='val_accuracy',
    max_trials=5,
    directory='output',
    project_name="Mnist_Fashion"
)

# Start hyperparameter search
tuner_search.search(train_images, train_labels, epochs=3, validation_split=0.1)

# Get the best model
best_model = tuner_search.get_best_models(num_models=1)[0]

# Show model summary
best_model.summary()

# Continue training the best model
best_model.fit(train_images, train_labels, epochs=10, validation_split=0.1, initial_epoch=3)
