import argparse
import os
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from handwriting.model import build_digit_cnn

def train(epochs, batch_size, save_path):
    # Load and preprocess data
    print("Loading MNIST data...")
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

    # Normalize to 0-1 and add channel dimension
    train_images = train_images.reshape((-1, 28, 28, 1)).astype('float32') / 255
    test_images = test_images.reshape((-1, 28, 28, 1)).astype('float32') / 255

    # Build model
    print("Building model...")
    model = build_digit_cnn()
    model.summary()

    # Ensure directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Callbacks
    checkpoint = ModelCheckpoint(save_path, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')

    # Data Augmentation
    # Simulating variations like rotation, zoom, and shifts which are common in handwriting
    datagen = ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        shear_range=0.1,
        fill_mode='nearest'
    )

    datagen.fit(train_images)

    # Train
    print("Starting training with augmentation...")
    model.fit(datagen.flow(train_images, train_labels, batch_size=batch_size),
              steps_per_epoch=len(train_images) // batch_size,
              epochs=epochs,
              validation_data=(test_images, test_labels),
              callbacks=[checkpoint])

    print(f"Training complete. Best model saved to {save_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train MNIST CNN')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--batch', type=int, default=64, help='Batch size')
    parser.add_argument('--save', type=str, default='models/digit_cnn.h5', help='Path to save model')

    args = parser.parse_args()
    train(args.epochs, args.batch, args.save)
