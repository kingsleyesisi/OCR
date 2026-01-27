#!/bin/bash
echo "Starting Model Training..."
export PYTHONPATH=$PYTHONPATH:.
python handwriting/train_mnist.py --epochs 5 --save models/digit_cnn.h5
echo "Training finished."
