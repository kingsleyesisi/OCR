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

    # Random Dilation (Thicken strokes simulate markers/bold pens)
    if tf.random.uniform([]) > 0.6:
        # tf.nn.max_pool2d acts as dilation
        image = tf.expand_dims(image, 0)
        image = tf.nn.max_pool2d(image, ksize=3, strides=1, padding='SAME')
        image = tf.squeeze(image, 0)

    # Random Erosion (Thin strokes simulate biro/thin pens)
    if tf.random.uniform([]) > 0.6:
        # Erosion is negative max pool of negative image
        image = tf.expand_dims(image, 0)
        image = -tf.nn.max_pool2d(-image, ksize=3, strides=1, padding='SAME')
        image = tf.squeeze(image, 0)


    # Clip values to valid range
    image = tf.clip_by_value(image, 0.0, 1.0)
    return image, label


def train(epochs, batch_size, save_path):
    # Load and preprocess data
    print("Loading MNIST data...")
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

    # Normalize to 0-1 and add channel dimension
    train_images = train_images[:10000].reshape((-1, 28, 28, 1)).astype('float32') / 255
    train_labels = train_labels[:10000]
    test_images = test_images[:2000].reshape((-1, 28, 28, 1)).astype('float32') / 255
    test_labels = test_labels[:2000]

    import numpy as np
    
    # Generate Synthetic Noise images (Class 10)
    print("Generating synthetic noise data...")
    num_noise_train = 2000
    num_noise_test = 500
    
    def generate_noise(num_samples):
        # 1. Complete random noise
        # 2. Random lines/squiggles
        # 3. Blank images with slight noise
        noise = []
        for _ in range(num_samples):
            choice = np.random.rand()
            if choice < 0.3:
                # Random pixels
                img = np.random.rand(28, 28, 1) * 0.5
            elif choice < 0.6:
                # Random lines (simulating random contour boundaries)
                img = np.zeros((28, 28, 1))
                for _ in range(np.random.randint(1, 5)):
                    x1, y1 = np.random.randint(0, 28, 2)
                    x2, y2 = np.random.randint(0, 28, 2)
                    import cv2
                    cv2.line(img, (x1, y1), (x2, y2), (np.random.rand() * 0.7 + 0.3), np.random.randint(1, 4))
            elif choice < 0.8:
                # Blocks
                img = np.zeros((28, 28, 1))
                x, y = np.random.randint(0, 20, 2)
                w, h = np.random.randint(4, 15, 2)
                import cv2
                cv2.rectangle(img, (x, y), (x+w, y+h), (np.random.rand() * 0.7 + 0.3), -1)
            else:
                # Emptyish or very dark noise
                img = np.random.rand(28, 28, 1) * 0.1
            noise.append(img.astype('float32'))
        return np.array(noise)

    train_noise = generate_noise(num_noise_train)
    test_noise = generate_noise(num_noise_test)
    train_labels_noise = np.full((num_noise_train,), 10, dtype=train_labels.dtype)
    test_labels_noise = np.full((num_noise_test,), 10, dtype=test_labels.dtype)

    train_images = np.concatenate([train_images, train_noise], axis=0)
    train_labels = np.concatenate([train_labels, train_labels_noise], axis=0)
    test_images = np.concatenate([test_images, test_noise], axis=0)
    test_labels = np.concatenate([test_labels, test_labels_noise], axis=0)


    # Create augmented training dataset
    print("Preparing augmented dataset...")
    train_ds = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
    train_ds = train_ds.shuffle(72000).map(augment, num_parallel_calls=tf.data.AUTOTUNE)
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
