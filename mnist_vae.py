import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import tensorflow as tf 
import tensorflow_probability as tfp
import sklearn as sk 
import time

import conv_vae as cv
from conv_vae import CVAE as CVAE

# Load the MNIST dataset
(train_images, _), (test_images, _) = tf.keras.datasets.mnist.load_data()


def preprocess_images(images):
    images = images.reshape((images.shape[0], 28, 28, 1)) / 255.0
    return np.where(images > .5, 1.0, 0.0).astype('float32')

train_images = preprocess_images(train_images)
test_images = preprocess_images(test_images)

train_size = train_images.shape[0]
batch_size = 32
test_size = test_images.shape[0]

train_dataset = (tf.data.Dataset.from_tensor_slices(train_images).shuffle(train_size).batch(batch_size))


# Training

epochs = 10
latent_dim = 2
num_example_to_generate = 16

random_vector_for_generation = tf.random.normal(
    shape = [num_example_to_generate, latent_dim]
)

model = CVAE(latent_dim)

