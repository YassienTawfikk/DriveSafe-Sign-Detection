import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

from src.__00__paths import figures_dir

classes = 43


def train_validation_split(data, labels):
    x_train, x_val, y_train, y_val = train_test_split(data, labels, test_size=0.2, random_state=42)

    y_train = to_categorical(y_train, classes)
    y_val = to_categorical(y_val, classes)

    return x_train, x_val, y_train, y_val


def return_CNN_model():
    model = Sequential([
        Conv2D(32, (5, 5), activation='relu', input_shape=(30, 30, 3)),
        Conv2D(64, (5, 5), activation='relu'),
        MaxPool2D(pool_size=(2, 2)),
        Dropout(0.15),

        Conv2D(128, (3, 3), activation='relu'),
        Conv2D(256, (3, 3), activation='relu'),
        MaxPool2D(pool_size=(2, 2)),
        Dropout(0.20),

        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.25),
        Dense(classes, activation='softmax')
    ])

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    model.summary()

    return model


def save_training_performance(history):
    plt.figure(figsize=(16, 5))

    # Accuracy plot
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train')
    plt.plot(history.history['val_accuracy'], label='Validation')
    plt.title('Model Accuracy', fontsize=16)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid()

    # Loss plot
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train')
    plt.plot(history.history['val_loss'], label='Validation')
    plt.title('Model Loss', fontsize=16)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()

    plt.savefig(figures_dir / "training_performance.png", dpi=300)
