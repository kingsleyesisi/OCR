import argparse
import os
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from handwriting.model import build_digit_cnn


def augment(image, label):
    """Apply random augmentations to simulate real handwriting variations.
    Uses tf.image functions — no scipy dependency needed."""
    # Random rotation (up to ~10 degrees via affine transform workaround)
    # TF doesn't have native rotate, so we use random shifts and zoom instead.

    # Random brightness/contrast variation
    image = tf.image.random_brightness(image, max_delta=0.1)
    image = tf.image.random_contrast(image, lower=0.9, upper=1.1)

    # Random shift via padding + cropping (simulates translation)
    # Pad by 4 pixels on each side, then random crop back to 28x28
    image = tf.image.resize_with_crop_or_pad(image, 36, 36)
    image = tf.image.random_crop(image, size=[28, 28, 1])

    # Random zoom via resize
    if tf.random.uniform([]) > 0.5:
        zoom = tf.random.uniform([], 0.85, 1.15)
        new_size = tf.cast(28 * zoom, tf.int32)
        image = tf.image.resize(image, [new_size, new_size])
        image = tf.image.resize_with_crop_or_pad(image, 28, 28)

    # Clip values to valid range
    image = tf.clip_by_value(image, 0.0, 1.0)
    return image, label


def train(epochs, batch_size, save_path):
    # Load and preprocess data
    print("Loading MNIST data...")
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

    # Normalize to 0-1 and add channel dimension
    train_images = train_images.reshape((-1, 28, 28, 1)).astype('float32') / 255
    test_images = test_images.reshape((-1, 28, 28, 1)).astype('float32') / 255

    # Create augmented training dataset
    print("Preparing augmented dataset...")
    train_ds = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
    train_ds = train_ds.shuffle(10000).map(augment, num_parallel_calls=tf.data.AUTOTUNE)
    train_ds = train_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    test_ds = tf.data.Dataset.from_tensor_slices((test_images, test_labels))
    test_ds = test_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    # Build model
    print("Building model...")
    model = build_digit_cnn()
    model.summary()

    # Ensure directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Callbacks
    checkpoint = ModelCheckpoint(
        save_path, monitor='val_accuracy', verbose=1,
        save_best_only=True, mode='max'
    )
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss', factor=0.5, patience=3,
        min_lr=1e-6, verbose=1
    )
    early_stop = EarlyStopping(
        monitor='val_accuracy', patience=5,
        restore_best_weights=True, verbose=1
    )

    # Train with augmented data
    print("Starting training with data augmentation...")
    model.fit(
        train_ds,
        epochs=epochs,
        validation_data=test_ds,
        callbacks=[checkpoint, reduce_lr, early_stop]
    )

    # Final evaluation
    loss, accuracy = model.evaluate(test_ds, verbose=0)
    print(f"\nFinal test accuracy: {accuracy * 100:.2f}%")
    print(f"Training complete. Best model saved to {save_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train MNIST CNN')
    parser.add_argument('--epochs', type=int, default=15, help='Number of epochs (default: 15)')
    parser.add_argument('--batch', type=int, default=64, help='Batch size (default: 64)')
    parser.add_argument('--save', type=str, default='models/digit_cnn.h5', help='Path to save model')

    args = parser.parse_args()
    train(args.epochs, args.batch, args.save)
