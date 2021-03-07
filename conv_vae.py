import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import tensorflow as tf 
import tensorflow_probability as tfp
import sklearn as sk 

# CVAE Class
class CVAE(tf.keras.Model):
    """Convolutional variational autoencoder."""

    def __init__(self, latent_dim):
        super(CVAE, self).__init__()
        self.latent_dim = latent_dim
        
        # Encoder network
        self.encoder = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape = (28, 28, 1)),
                tf.keras.layers.Conv2D(
                    filters = 32, kernel_size = 3, strides = (2, 2), activation = 'relu'),
                tf.keras.layers.Conv2D(
                    filters = 64, kernel_size = 3, strides = (2, 2), activation = 'relu'),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(latent_dim + latent_dim) # two times latent_dim for z-distribution mean and variance I think?   
            ]
        )

        # Decoder network
        self.decoder = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape = (latent_dim, )),
                tf.keras.layers.Dense(units = 7*7*32, activation = tf.nn.relu),
                tf.keras.layers.Reshape(target_shape = (7, 7, 32)),
                tf.keras.layers.Conv2DTranspose(
                    filters = 64, kernel_size = 3, strides = 2, padding = 'same', 
                    activation = 'relu'),
                tf.keras.layers.Conv2DTranspose(
                    filters = 32, kernel_size = 3, strides = 2, padding = 'same',
                    activation = 'relu'),
                tf.keras.layers.Conv2DTranspose(
                    filters = 1, kernel_size = 3, strides = 1, padding = 'same'),
            ]
        )

        @tf.function
        def sample(self, eps = None):
            if eps is None:
                eps = tf.random.normal(shape = (100, self.latent_dim))
            return self.decode(eps, apply_sigmoid = True)
        
        def encode(self, x):
            mean, logvar = tf.split(self.encoder(x), num_or_size_splits = 2, axis = 1)
            return mean, logvar
        
        def reparameterize(self, mean, logvar):
            eps = tf.random.normal(shape = mean.shape)
            return eps * tf.exp(logvar * .5) + mean

        def decode(self, z, apply_sigmoid = False):
            logits = self.decoder(z)
            if apply_sigmoid:
                probs = tf.sigmoid(logits)
                return probs
            return logits



# Define loss function and optimizer
optimizer = tf.keras.optimizers.Adam(1e-4)

def log_normal_pdf(sample, mean, logvar, raxis = 1):
    log2pi = tf.math.log(2., np.pi)
    return tf.reduce_sum(
        -.5 *((sample - mean) ** 2 * tf.exp(-logvar) + logvar + log2pi)
    )

def compute_loss(model, x):
    mean, logvar = model.encode(x)
    z = model.reparameterize(mean, logvar)
    x_logit = model.decode(z)
    cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits = x_logit, labels = x)
    logpx_z = -tf.reduce_sum(cross_ent, axis = [1, 2, 3])
    logpz = log_normal_pdf(z, 0., 0.)
    logqz_x = log_normal_pdf(z, mean, logvar)
    return -tf.reduce_mean(logpx_z + logpz - logqz_x)

@tf.function
def train_step(model, x, optimizer):
    """ Execute one trainingg step: return the loss. Update parameters with gradient"""

    with tf.GradientTape() as tape:
        loss = compute_loss(model, x)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))