import argparse
import os
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.callbacks import ModelCheckpoint
from handwriting.model import build_digit_cnn

def train(epochs, batch_size, save_path):
    # Load and preprocess data
    print("Loading MNIST data...")
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

    # Normalize to 0-1 and add channel dimension
    train_images = train_images.reshape((60000, 28, 28, 1)).astype('float32') / 255
    test_images = test_images.reshape((10000, 28, 28, 1)).astype('float32') / 255

    # Build model
    print("Building model...")
    model = build_digit_cnn()
    model.summary()

    # Ensure directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Callbacks
    checkpoint = ModelCheckpoint(save_path, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')

    # Train
    print("Starting training...")
    model.fit(train_images, train_labels, epochs=epochs, batch_size=batch_size,
              validation_data=(test_images, test_labels), callbacks=[checkpoint])

    print(f"Training complete. Best model saved to {save_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train MNIST CNN')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs')
    parser.add_argument('--batch', type=int, default=64, help='Batch size')
    parser.add_argument('--save', type=str, default='models/digit_cnn.h5', help='Path to save model')

    args = parser.parse_args()
    train(args.epochs, args.batch, args.save)
