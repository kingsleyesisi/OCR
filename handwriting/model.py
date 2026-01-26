import tensorflow as tf
from tensorflow.keras import layers, models

def build_digit_cnn(input_shape=(28, 28, 1), num_classes=10):
    """
    Builds a simple CNN for digit recognition.

    Args:
        input_shape (tuple): Shape of the input image (height, width, channels).
        num_classes (int): Number of output classes (digits 0-9).

    Returns:
        model: Compiled Keras model.
    """
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model
