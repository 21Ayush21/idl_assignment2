#!/usr/bin/env python
# coding: utf-8

# # Introduction to Deep Learning, Assignment 2, Task 1
# 
# <div style="text-align: right">   </div>
# 
# In this notebook we are going to cover two generative models for generating novel images:
# 
# 1. Variational Autoencoders (**VAEs**)
# 2. Generative adversarial networks (**GANs**)
# 
# 
# <img src="https://lilianweng.github.io/lil-log/assets/images/three-generative-models.png" width="500">
# 
# 
# Your main goal will be to retrain these models on a dataset of your choice and do some experiments on the learned latent space.
# 
# When using the Lab Computers, download the Jupyter Notebook to one of the machines first.
# 
# If you want to use Google Colab, you should first copy this notebook and enable GPU runtime in 'Runtime -> Change runtime type -> Hardware acceleration -> GPU **OR** TPU'.

# In[1]:


### If you are running on LIACS Lab machines, run the command below to reproduce the same package environment as Colab

#!pip install tf_keras==2.17.0 # Uncomment this to set up the right version


# In[1]:


import os
os.environ['TF_USE_LEGACY_KERAS'] = '1'

from tqdm import tqdm
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

#from google.colab import drive
# drive.mount('/content/drive', force_remount=True) ## If you want to use your google drive


# In[3]:


get_ipython().system('wget https://surfdrive.surf.nl/s/wxStiBetTKDJC32/download -O face_dataset_64x64.npy')


# 
# ### Dataset
# 
# This dataset is called [Flickr-Faces-HQ Dataset](https://github.com/NVlabs/ffhq-dataset). Here we will use a downsampled version of it (64x64x3) that will speed up all the experiments. [Download](https://surfdrive.surf.nl/s/wxStiBetTKDJC32).
# 
# After downloading you should copy it to your google drive's main directory (or modify the code to load it from elsewhere).
# 
# After running the notebook on this default dataset you then need to find a dataset of your own.

# In[2]:


def load_real_samples(scale=False):
    # We load 20,000 samples only to avoid memory issues, you can  change this value
    X = np.load('face_dataset_64x64.npy',  fix_imports=True,encoding='latin1')[:20000, :, :, :]
    # Scale samples in range [-127, 127]
    if scale:
        X = (X - 127.5) * 2
    return X / 255.

# We will use this function to display the output of our models throughout this notebook
def grid_plot(images, epoch='', name='', n=3, save=False, scale=False):
    if scale:
        images = (images + 1) / 2.0
    for index in range(n * n):
        plt.subplot(n, n, 1 + index)
        plt.axis('off')
        plt.imshow(images[index])
    fig = plt.gcf()
    fig.suptitle(name + '  '+ str(epoch), fontsize=14)
    if save:
        filename = 'results/generated_plot_e%03d_f.png' % (epoch+1)
        plt.savefig(filename)
        plt.close()
    plt.show()


dataset = load_real_samples()
grid_plot(dataset[np.random.randint(0, 1000, 9)], name='Fliqr dataset (64x64x3)', n=3)


# ## 2.1. Introduction
# 
# The generative models that we are going to cover both have the following components:
# 
# 1. A downsampling architecture (encoder in case of VAE, and discriminator in case of GAN) to either extract features from the data or model its distribution.
# 2. An upsampling architecture (decoder for VAE, generator for GAN) that will use some kind of latent vector to generate new samples that resemble the data that it was trained on.
# 
# Since we are going to be dealing with images, we are going to use convolutional networks for upsampling and downsampling, similar to what you see below.
# 
# <img src="https://i2.wp.com/sefiks.com/wp-content/uploads/2018/03/convolutional-autoencoder.png" width="500">
# 
# 
# #### Code for building these components:

# In[3]:


from tensorflow.keras.layers import Dense, Flatten, Conv2D, Conv2DTranspose, Reshape

def build_conv_net(in_shape, out_shape, n_downsampling_layers=4, filters=128, out_activation='sigmoid'):
    """
    Build a basic convolutional network
    """
    default_args=dict(kernel_size=(3,3), strides=(2,2), padding='same', activation='relu')

    input = tf.keras.Input(shape=in_shape)
    x = Conv2D(filters=filters, name='enc_input', **default_args)(input) #** is the unpacking argument

    for _ in range(n_downsampling_layers): # _ if we dont care about the varible itself
        x = Conv2D(**default_args, filters=filters)(x)

    x = Flatten()(x)
    x = Dense(out_shape, activation=out_activation, name='enc_output')(x) # out_shape gives the shape of the latent space

    model = tf.keras.Model(inputs=input, outputs=x, name='Encoder')

    model.summary()
    return model


def build_deconv_net(latent_dim, n_upsampling_layers=4, filters=128, activation_out='sigmoid'):
    """
    Build a deconvolutional network for decoding/upscaling latent vectors

    When building the deconvolutional architecture, usually it is best to use the same layer sizes that
    were used in the downsampling network and the Conv2DTranspose layers are used instead of Conv2D layers.
    Using identical layers and hyperparameters ensures that the dimensionality of our output matches the
    shape of our input images.
    """
    input = tf.keras.Input(shape=(latent_dim,))
    x = Dense(4 * 4 * 64, input_dim=latent_dim, name='dec_input')(input)
    x = Reshape((4, 4, 64))(x) # This matches the output size of the downsampling architecture

    default_args=dict(kernel_size=(3,3), strides=(2,2), padding='same', activation='relu')

    for i in range(n_upsampling_layers):
        x = Conv2DTranspose(filters=filters, **default_args)(x) # reverse of a Conv layer, takes 1 input and maps kernel to output, addind where overlap

    # This last convolutional layer converts back to 3 channel RGB image
    x = Conv2D(filters=3, kernel_size=(3,3), padding='same', activation=activation_out, name='dec_output')(x)

    model = tf.keras.Model(inputs=input, outputs=x, name='Decoder')
    model.summary()
    return model


# ### Convolutional Autoencoder example
# 
# Using these two basic building blocks we can now build a Convolutional Autoencoder (CAE).
# 
# <img src="https://lilianweng.github.io/lil-log/assets/images/autoencoder-architecture.png" width="500">
# 
# 
# 
# Even though it's not a generative model, CAE is a great way to illustrate how these two components (convolutional and deconvolutional networks) can be used together to reconstruct images.
# 
# You can view such model as a compression/dimensionality reduction method as each image gets compressed to a vector of 256 numbers by the encoder and gets decompressed back into an image using the decoder network.

# In[4]:


def build_convolutional_autoencoder(data_shape, latent_dim, filters=128):
    encoder = build_conv_net(in_shape=data_shape, out_shape=latent_dim, filters=filters)
    decoder = build_deconv_net(latent_dim, activation_out='sigmoid', filters=filters)

    # We connect encoder and decoder into a single model
    autoencoder = tf.keras.Sequential([encoder, decoder])

    # Binary crossentropy loss - pairwise comparison between input and output pixels
    autoencoder.compile(loss='binary_crossentropy', optimizer='adam')

    return autoencoder


# In[5]:


# Defining the model dimensions and building it
image_size = dataset.shape[1:]
latent_dim = 512
num_filters = 128
cae = build_convolutional_autoencoder(image_size, latent_dim, num_filters)


## Training the Convolutional autoencoder to reconstruct images
for epoch in range(50):
    print('\nEpoch: ', epoch)

    # Note that (X=y) when training autoencoders!
    # In this case we only care about qualitative performance, we don't split into train/test sets
    cae.fit(x=dataset, y=dataset, epochs=1, batch_size=64)

    samples = dataset[:9]
    reconstructed = cae.predict(samples)
    grid_plot(samples, epoch, name='Original', n=3, save=False)
    grid_plot(reconstructed, epoch, name='Reconstructed', n=3, save=False)


# *Note: You may experiment with the latent dimensionality and number of filters in your convolutional network to see how it affects the reconstruction quality. Remember that this also affects the size of the model and time it takes to train.*

# ---
# ---
# 
# 
# ## 2. 2. Variational Autoencoders (VAEs)
# 
# <img src="https://lilianweng.github.io/lil-log/assets/images/vae-gaussian.png" width="500">
# 
# #### Encoder network
# This defines the approximate posterior distribution, which takes as input an observation and outputs a set of parameters for specifying the conditional distribution of the latent representation. In this example, we simply model the distribution as a diagonal Gaussian, and the network outputs the mean and log-variance parameters of a factorized Gaussian. We output log-variance instead of the variance directly for numerical stability.
# 
# #### Decoder network
# This defines the conditional distribution of the observation $z$, which takes a latent sample $z$ as input and outputs the parameters for a conditional distribution of the observation. We model the latent distribution prior  as a unit Gaussian.
# 
# #### Reparameterization trick
# To generate a sample  for the decoder during training, we can sample from the latent distribution defined by the parameters outputted by the encoder, given an input observation $z$. However, this sampling operation creates a bottleneck because backpropagation cannot flow through a random node.
# 
# To address this, we use a reparameterization trick. In our example, we approximate  using the decoder parameters and another parameter  as follows:
# 
# $$z = \mu + \sigma\epsilon$$
# 
# where $\mu$ and $\sigma$  represent the mean and standard deviation of a Gaussian distribution respectively. They can be derived from the decoder output. The  can be thought of as a random noise used to maintain stochasticity of $z$. We generate  from a standard normal distribution.
# 
# The latent variable  is now generated by a function of $\mu$ and $\sigma$ which would enable the model to backpropagate gradients in the encoder through $\mu$ and $\sigma$ respectively, while maintaining stochasticity through $\epsilon$.
# 
# #### Implementation
# 
# You can see how this trick is implemented below by creating a custom layer by sublassing tf.keras.layers.Layer.
# It is then connected to the output of the original encoder architecture and an additional [KL](https://en.wikipedia.org/wiki/Kullback–Leibler_divergence) loss term is introduced.
# 

# In[5]:


class Sampling(tf.keras.layers.Layer):
    """
    Custom layer for the variational autoencoder
    It takes two vectors as input - one for means and other for variances of the latent variables described by a multimodal gaussian
    Its output is a latent vector randomly sampled from this distribution
    """
    def call(self, inputs):
        z_mean, z_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim)) # e for each of the pixels 64x64x3 in each image
        return z_mean + tf.exp(0.5 * z_var) * epsilon # log variance here

def build_vae(data_shape, latent_dim, filters=128):

    # Building the encoder - starts with a simple downsampling convolutional network
    encoder = build_conv_net(data_shape, latent_dim*2, filters=filters) # 2 since have both mu and sigma

    # Adding special sampling layer that uses the reparametrization trick
    z_mean = Dense(latent_dim)(encoder.output)
    z_var = Dense(latent_dim)(encoder.output)
    z = Sampling()([z_mean, z_var]) # z is our latent varible vector

    # Connecting the two encoder parts
    encoder = tf.keras.Model(inputs=encoder.input, outputs=z)

    # Defining the decoder which is a regular upsampling deconvolutional network
    decoder = build_deconv_net(latent_dim, activation_out='sigmoid', filters=filters)
    vae = tf.keras.Model(inputs=encoder.input, outputs=decoder(z))

    # Define a custom layer for the KL loss calculation
    class KLLossLayer(tf.keras.layers.Layer):
        def call(self, inputs):
            z_mean, z_var = inputs
            kl_loss = -0.5 * tf.reduce_sum(z_var - tf.square(z_mean) - tf.exp(z_var) + 1)
            # Add the KL loss to the model's losses
            self.add_loss(kl_loss / tf.cast(tf.keras.backend.prod(data_shape), tf.float32))
            return inputs  # Pass through the inputs unchanged

    # Apply the custom layer to z_mean and z_var
    _, _ = KLLossLayer()([z_mean, z_var])

    vae.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss='binary_crossentropy')

    return encoder, decoder, vae


# In[42]:


# Training the VAE model

latent_dim = 64
encoder, decoder, vae = build_vae(dataset.shape[1:], latent_dim, filters=64) # three seperate ones since want to be able to input latent varibles

for epoch in range(20):
    vae.fit(x=dataset, y=dataset, epochs=1, batch_size=8)

    # Generate random vectors that we will use to sample from the learned latent space
    coefficient = 6                                 # You can tweak this coefficient to increase/decrease the std of the sampled vectors
    latent_vectors = np.random.randn(9, latent_dim) # Generate 9 random points in the latent space
    images = decoder(latent_vectors / coefficient)  # Feed the vectors scaled by the coefficient to the model
    grid_plot(images, epoch, name='VAE generated images (randomly sampled from the latent space)', n=3, save=False)


# *Note: again, you might experiment with the latent dimensionality, batch size and the architecture of your convolutional nets to see how it affects the generative capabilities of this model.*

# ---
# 
# ## 2.3 Generative Adversarial Networks (GANs)
# 
# ### GAN architecture
# Generative adversarial networks consist of two models: a generative model and a discriminative model.
# 
# <img src="https://media.springernature.com/original/springer-static/image/chp%3A10.1007%2F978-1-4842-3679-6_8/MediaObjects/463582_1_En_8_Fig1_HTML.jpg" width="500">
# 
# **The discriminator** model is a classifier that determines whether a given image looks like a real image from the dataset or like an artificially created image. This is basically a binary classifier that will take the form of a normal convolutional neural network (CNN).
# As an input this network will get samples both from the dataset that it is trained on and on the samples generated by the **generator**.
# 
# The **generator** model takes random input values (noise) and transforms them into images through a deconvolutional neural network.
# 
# Over the course of many training iterations, the weights and biases in the discriminator and the generator are trained through backpropagation. The discriminator learns to tell "real" images of handwritten digits apart from "fake" images created by the generator. At the same time, the generator uses feedback from the discriminator to learn how to produce convincing images that the discriminator can't distinguish from real images.
# 
# 
# 

# In[6]:


from tensorflow.keras.optimizers.legacy import Adam

def build_gan(data_shape, latent_dim, filters=128, lr=0.0002, beta_1=0.5):
    optimizer = Adam(learning_rate=lr, beta_1=beta_1)

    # Usually the GAN generator has tanh activation function in the output layer
    generator = build_deconv_net(latent_dim, activation_out='tanh', filters=filters)

    # Build and compile the discriminator
    discriminator = build_conv_net(in_shape=data_shape, out_shape=1, filters=filters) # Single output for binary classification
    discriminator.compile(loss='binary_crossentropy', optimizer=optimizer)

    # End-to-end GAN model for training the generator
    discriminator.trainable = False
    true_fake_prediction = discriminator(generator.output)
    GAN = tf.keras.Model(inputs=generator.input, outputs=true_fake_prediction)
    GAN.compile(loss='binary_crossentropy', optimizer=optimizer)

    return discriminator, generator, GAN


# ### Definining custom functions for training your GANs
# 
# ---
# 
# 
# 

# In[7]:


def get_batch(generator, dataset, batch_size=64):
    """
    Fetches one batch of data and ensures no memory leaks by using TensorFlow operations.
    """
    half_batch = batch_size // 2

    # Generate fake images
    latent_vectors = tf.random.normal(shape=(half_batch, latent_dim))
    fake_data = generator(latent_vectors, training=False)

    # Select real images
    idx = np.random.randint(0, dataset.shape[0], half_batch)
    real_data = dataset[idx]

    # Combine
    X = tf.concat([real_data, fake_data], axis=0)
    y = tf.concat([tf.ones((half_batch, 1)), tf.zeros((half_batch, 1))], axis=0)

    return X, y


def train_gan(generator, discriminator, gan, dataset, latent_dim, n_epochs=20, batch_size=64):
    """
    Train the GAN with memory-efficient updates and clear session management.
    """
    batches_per_epoch = dataset.shape[0] // batch_size

    for epoch in range(n_epochs):
        for batch in tqdm(range(batches_per_epoch)):
            # Train Discriminator
            X, y = get_batch(generator, dataset, batch_size)
            discriminator_loss = discriminator.train_on_batch(X, y)

            # Train Generator
            latent_vectors = tf.random.normal(shape=(batch_size, latent_dim))
            y_gan = tf.ones((batch_size, 1))
            generator_loss = gan.train_on_batch(latent_vectors, y_gan)

        # Generate and visualize after each epoch
        noise = tf.random.normal(shape=(16, latent_dim))
        generated_images = generator(noise, training=False)
        grid_plot(generated_images.numpy(), epoch, name='Generated Images', n=3, scale=True)

        # Clear backend session to free memory
        tf.keras.backend.clear_session()


# In[8]:


## Build and train the model (need around 10 epochs to start seeing some results)

latent_dim = 256
discriminator, generator, gan = build_gan(dataset.shape[1:], latent_dim, filters=128)
dataset_scaled = load_real_samples(scale=True)

train_gan(generator, discriminator, gan, dataset_scaled, latent_dim, n_epochs=20)


# *Note: the samples generated by small GANs are more diverse, when compared to VAEs, however some samples might look strange and do not resemble the data the model was trained on.

# # New Code

# ### Importing new dataset

# In[9]:


import os
os.environ['TF_USE_LEGACY_KERAS'] = '1'

from tqdm import tqdm
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


# ### 64x64x3

# In[10]:


full_ds = tfds.load('stanford_dogs', split='train+test', as_supervised=True)
# 'train+test' includes all 20,580

resized_images = []  # want 64x64 images


for image, label in tfds.as_numpy(full_ds): 
    # image a NumPy array of shape (H, W, 3), label is an integer 

    img = Image.fromarray(image)              # Have to change to a PIL image
    img = img.resize((64, 64), Image.BILINEAR)  # resize to 64x64 ,bilinear interpolation, takes the weighted average of the 4 nearest pixels
    img_array = np.array(img, dtype=np.uint8)  # change back to array as in other notebook 

    resized_images.append(img_array) 


resized_images = np.stack(resized_images, axis=0)  # was a list beforehadn
print("Combined array shape:", resized_images.shape)


np.save('stanford_dogs_64x64.npy', resized_images)


# In[11]:


def load_real_samples_64(scale=False):
    X = np.load('stanford_dogs_64x64.npy')[:20000]  # or remove slicing to load all
    if scale:
        X = (X - 127.5) * 2
    return X / 255.0

dataset = load_real_samples_64()

grid_plot(dataset[np.random.randint(0, 1000, 9)], name='Stanford Dogs 64x64x3', n=3)


# ### 128x128x64

# In[12]:


full_ds = tfds.load('stanford_dogs', split='train+test', as_supervised=True)
# 'train+test' includes all 20,580

resized_images = []  # want 128x128 images


for image, label in tfds.as_numpy(full_ds): 
    # image a NumPy array of shape (H, W, 3), label is an integer 

    img = Image.fromarray(image)              # Have to change to a PIL image
    img = img.resize((128, 128), Image.BILINEAR)  # resize to 64x64 ,bilinear interpolation, takes the weighted average of the 4 nearest pixels
    img_array = np.array(img, dtype=np.uint8)  # change back to array as in other notebook 

    resized_images.append(img_array) 


resized_images = np.stack(resized_images, axis=0)  # was a list beforehadn
print("Combined array shape:", resized_images.shape)


np.save('stanford_dogs_128x128.npy', resized_images)


# In[13]:


def load_real_samples_128(scale=False):
    X = np.load('stanford_dogs_128x128.npy')[:20000]  # or remove slicing to load all
    if scale:
        X = (X - 127.5) * 2
    return X / 255.0

dataset = load_real_samples_128()
grid_plot(dataset[np.random.randint(0, 1000, 9)], name='Stanford Dogs 128x128x3', n=3)


# In[14]:


dataset = load_real_samples_64()


# ## CAE Generation

# In[9]:


image_size = dataset.shape[1:]
latent_dim = 512
num_filters = 128
cae = build_convolutional_autoencoder(image_size, latent_dim, num_filters)


## Training the Convolutional autoencoder to reconstruct images
for epoch in range(50):
    print('\nEpoch: ', epoch)

    # Note that (X=y) when training autoencoders!
    # In this case we only care about qualitative performance, we don't split into train/test sets
    cae.fit(x=dataset, y=dataset, epochs=1, batch_size=64)

    samples = dataset[:9]
    reconstructed = cae.predict(samples)
    grid_plot(samples, epoch, name='Original', n=3, save=False)
    grid_plot(reconstructed, '', name='CAE-Reconstructed', n=3, save=True)


# ## Previous models on New dataset

# ### VAE

# In[21]:


# Training the VAE model

latent_dim = 64
encoder, decoder, vae = build_vae(dataset.shape[1:], latent_dim, filters=64) # three seperate ones since want to be able to input latent varibles

for epoch in range(20):
    vae.fit(x=dataset, y=dataset, epochs=1, batch_size=8)

    # Generate random vectors that we will use to sample from the learned latent space
    coefficient = 6                                 # You can tweak this coefficient to increase/decrease the std of the sampled vectors
    latent_vectors = np.random.randn(9, latent_dim) # Generate 9 random points in the latent space
    images = decoder(latent_vectors / coefficient)  # Feed the vectors scaled by the coefficient to the model
    grid_plot(images, epoch, name='VAE generated images (randomly sampled from the latent space)', n=3, save=False)


# ### GAN

# In[19]:


latent_dim = 256
discriminator, generator, gan = build_gan(dataset.shape[1:], latent_dim, filters=128)
dataset_scaled = load_real_samples(scale=True)

train_gan(generator, discriminator, gan, dataset_scaled, latent_dim, n_epochs=20)


# # GAN

# ## Gridsearch Code

# In[15]:


import os
os.environ['TF_USE_LEGACY_KERAS'] = '1'

from tqdm import tqdm
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.model_selection import ParameterGrid


# In[16]:


# Updated gridplot allowing to save results
def grid_plot(images, epoch='', name='', n=3, save=False, scale=False):
    if scale:
        images = (images + 1) / 2.0
    for index in range(n * n):
        plt.subplot(n, n, 1 + index)
        plt.axis('off')
        plt.imshow(images[index])
    fig = plt.gcf()
    fig.suptitle(name + '  '+ str(epoch), fontsize=14)
    if save:
        plt.savefig(f'Graphs/{name + '  '+ str(epoch)}.png')

    plt.show()
    plt.close()


# In[17]:


# Fixed hyperparamers
n_epochs = 30
N_layers = 4


# In[18]:


from tensorflow.keras.layers import Dense, Flatten, Conv2D, Conv2DTranspose, Reshape

def build_conv_net(in_shape, out_shape, filters, n_downsampling_layers=N_layers, out_activation='sigmoid'):
    """
    Build a basic convolutional network
    """
    default_args=dict(kernel_size=(3,3), strides=(2,2), padding='same', activation='relu')

    input = tf.keras.Input(shape=in_shape)
    x = Conv2D(filters=filters, name='enc_input', **default_args)(input) #** is the unpacking argument

    for _ in range(n_downsampling_layers): # _ if we dont care about the varible itself
        x = Conv2D(**default_args, filters=filters)(x)

    x = Flatten()(x)
    x = Dense(out_shape, activation=out_activation, name='enc_output')(x) # out_shape gives the shape of the latent space

    model = tf.keras.Model(inputs=input, outputs=x, name='Encoder')

    model.summary()
    return model


def build_deconv_net(latent_dim, filters, n_upsampling_layers=N_layers, activation_out='sigmoid'):
    """
    Build a deconvolutional network for decoding/upscaling latent vectors

    When building the deconvolutional architecture, usually it is best to use the same layer sizes that
    were used in the downsampling network and the Conv2DTranspose layers are used instead of Conv2D layers.
    Using identical layers and hyperparameters ensures that the dimensionality of our output matches the
    shape of our input images.
    """
    input = tf.keras.Input(shape=(latent_dim,))
    x = Dense(4 * 4 * 64, input_dim=latent_dim, name='dec_input')(input)
    x = Reshape((4, 4, 64))(x) # This matches the output size of the downsampling architecture

    default_args=dict(kernel_size=(3,3), strides=(2,2), padding='same', activation='relu')

    for i in range(n_upsampling_layers):
        x = Conv2DTranspose(filters=filters, **default_args)(x) # reverse of a Conv layer, takes 1 input and maps kernel to output, addind where overlap

    # This last convolutional layer converts back to 3 channel RGB image
    x = Conv2D(filters=3, kernel_size=(3,3), padding='same', activation=activation_out, name='dec_output')(x)

    model = tf.keras.Model(inputs=input, outputs=x, name='Decoder')
    model.summary()
    return model


# In[19]:


def get_batch(generator, dataset, batch_size=64):
    """
    Fetches one batch of data (includes both real and generated images) and ensures no memory leaks by using TensorFlow operations.
    """
    half_batch = batch_size // 2

    # Generate fake images
    latent_vectors = tf.random.normal(shape=(half_batch, latent_dim))
    fake_data = generator(latent_vectors, training=False)

    # Select real images
    idx = np.random.randint(0, dataset.shape[0], half_batch)
    real_data = dataset[idx]

    # Combine
    X = tf.concat([real_data, fake_data], axis=0)
    y = tf.concat([tf.ones((half_batch, 1)), tf.zeros((half_batch, 1))], axis=0)

    return X, y


# In[20]:


from tensorflow.keras.optimizers.legacy import Adam

def build_gan(data_shape, latent_dim, filters, lr=0.0002, beta_1=0.5):
    optimizer = Adam(learning_rate=lr, beta_1=beta_1)
    print(filters)
    # Usually the GAN generator has tanh activation function in the output layer
    generator = build_deconv_net(latent_dim, filters, activation_out='tanh')

    # Build and compile the discriminator
    discriminator = build_conv_net(in_shape=data_shape, out_shape=1,  filters=filters) # Single output for binary classification
    discriminator.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    # End-to-end GAN model for training the generator
    discriminator.trainable = False
    true_fake_prediction = discriminator(generator.output)
    GAN = tf.keras.Model(inputs=generator.input, outputs=true_fake_prediction)
    GAN.compile(loss='binary_crossentropy', optimizer=optimizer)

    return discriminator, generator, GAN


# In[21]:


# Updated train gan model, allows for grid search and also saves learning plots and final generated images
def train_gan(generator, discriminator, gan, dataset, latent_dim, n_f, n_epochs, batch_size=64):
    """
    Train the GAN with memory-efficient updates and clear session management.
    """
    filters = n_f
    Title = f"(latent={latent_dim}, filters={filters}, layers={N_layers})"
    batches_per_epoch = dataset.shape[0] // batch_size
    history = {'disc_loss': [], 'gen_loss': [], 'disc_acc': []}
    i=0


    for epoch in range(n_epochs):
        epoch_disc_loss = 0.0
        epoch_gen_loss = 0.0
        epoch_disc_acc = 0.0


        for batch in tqdm(range(batches_per_epoch)):
            # Train Discriminator
            X, y = get_batch(generator, dataset, batch_size)
            discriminator_loss = discriminator.train_on_batch(X, y) # trains the discriminator aswell as records the loss
            disc_loss = discriminator_loss[0]
            disc_acc = discriminator_loss[1]

            # Train Generator
            latent_vectors = tf.random.normal(shape=(batch_size, latent_dim))
            y_gan = tf.ones((batch_size, 1))
            gen_loss = gan.train_on_batch(latent_vectors, y_gan)

            epoch_disc_loss += disc_loss
            epoch_disc_acc += disc_acc
            epoch_gen_loss += gen_loss


        epoch_disc_loss /= batches_per_epoch
        epoch_disc_acc /= batches_per_epoch
        epoch_gen_loss /= batches_per_epoch
        history['disc_loss'].append(epoch_disc_loss)
        history['disc_acc'].append(epoch_disc_acc)
        history['gen_loss'].append(epoch_gen_loss)

        noise = tf.random.normal(shape=(16, latent_dim))
        generated_images = generator(noise, training=False)
        grid_plot(generated_images.numpy(), epoch, name='Generated Images', n=3, save=False, scale=True)

        i+=1
        if i == n_epochs:
            noise = tf.random.normal(shape=(1, latent_dim))
            generated_images = generator(noise, training=False)
            grid_plot(generated_images.numpy(),epoch=Title, name='Final Image', n=1, save=True, scale=True)

        # Clear backend session to free memory
        tf.keras.backend.clear_session()


    fig, ax1 = plt.subplots()
    ax1.plot(history['disc_loss'], 'r-', label='Discriminator Loss')
    ax1.plot(history['gen_loss'], 'b-', label='Generator Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend(loc='upper left')
    ax2 = ax1.twinx()
    ax2.plot(history['disc_acc'], 'g--', label='Discriminator Accuracy')
    ax2.set_ylabel('Accuracy')
    ax2.legend(loc='lower left')
    plt.title(Title)
    plt.savefig(f"Graphs/{Title}.png")
    plt.show()

    # Clear backend session to free memory
    tf.keras.backend.clear_session()


# In[22]:


# Grid search code
param_grid = {
    'latent_dim': [128,256,512],
    'filters': [128, 256, 512],
}


for params in ParameterGrid(param_grid):
    latent_dim = params['latent_dim']
    filters = params['filters']


    print(f"\nTraining GAN with latent_dim={latent_dim}, filters={filters}, layers={N_layers}")


    discriminator, generator, gan = build_gan(dataset.shape[1:], latent_dim, filters)
    dataset_scaled = load_real_samples(scale=True)

    train_gan(generator, discriminator, gan, dataset_scaled, latent_dim, filters, n_epochs)


# ## Saving Final Model

# In[23]:


def train_gan(generator, discriminator, gan, dataset, latent_dim, n_f, n_epochs, batch_size=64, save=False):
    """
    Train the GAN with memory-efficient updates and clear session management.
    """
    filters = n_f
    Title = f"Endgame-(latent={latent_dim}, filters={filters}, layers={N_layers})"
    batches_per_epoch = dataset.shape[0] // batch_size
    history = {'disc_loss': [], 'gen_loss': [], 'disc_acc': []}
    i=0


    for epoch in range(n_epochs):
        epoch_disc_loss = 0.0
        epoch_gen_loss = 0.0
        epoch_disc_acc = 0.0


        for batch in tqdm(range(batches_per_epoch)):
            # Train Discriminator
            X, y = get_batch(generator, dataset, batch_size)
            discriminator_loss = discriminator.train_on_batch(X, y) # trains the discriminator aswell as records the loss
            disc_loss = discriminator_loss[0]
            disc_acc = discriminator_loss[1]

            # Train Generator
            latent_vectors = tf.random.normal(shape=(batch_size, latent_dim))
            y_gan = tf.ones((batch_size, 1))
            gen_loss = gan.train_on_batch(latent_vectors, y_gan)

            epoch_disc_loss += disc_loss
            epoch_disc_acc += disc_acc
            epoch_gen_loss += gen_loss


        epoch_disc_loss /= batches_per_epoch
        epoch_disc_acc /= batches_per_epoch
        epoch_gen_loss /= batches_per_epoch
        history['disc_loss'].append(epoch_disc_loss)
        history['disc_acc'].append(epoch_disc_acc)
        history['gen_loss'].append(epoch_gen_loss)

        noise = tf.random.normal(shape=(16, latent_dim))
        generated_images = generator(noise, training=False)
        grid_plot(generated_images.numpy(), epoch, name='Generated Images', n=3, save=False, scale=True)

        i+=1
        if i == n_epochs:
            noise = tf.random.normal(shape=(1, latent_dim))
            generated_images = generator(noise, training=False)
            grid_plot(generated_images.numpy(),epoch=Title, name='Final Image', n=1, save=True, scale=True)

        if save==True:
            generator.save(f"Models/{Title}.keras")

        # Clear backend session to free memory
        tf.keras.backend.clear_session()


    fig, ax1 = plt.subplots()
    ax1.plot(history['disc_loss'], 'r-', label='Discriminator Loss')
    ax1.plot(history['gen_loss'], 'b-', label='Generator Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend(loc='upper left')
    ax2 = ax1.twinx()
    ax2.plot(history['disc_acc'], 'g--', label='Discriminator Accuracy')
    ax2.set_ylabel('Accuracy')
    ax2.legend(loc='lower left')
    plt.title(Title)
    plt.savefig(f"Graphs/{Title}.png")
    plt.show()

    # Clear backend session to free memory
    tf.keras.backend.clear_session()

## Build and train the model (need around 10 epochs to start seeing some results)

param_grid = {
    'latent_dim': [512],
    'filters': [256],
}


for params in ParameterGrid(param_grid):
    latent_dim = params['latent_dim']
    filters = params['filters']


    print(f"\nTraining GAN with latent_dim={latent_dim}, filters={filters}, layers={N_layers}")


    discriminator, generator, gan = build_gan(dataset.shape[1:], latent_dim, filters)
    dataset_scaled = load_real_samples(scale=True)

    train_gan(generator, discriminator, gan, dataset_scaled, latent_dim, filters, n_epochs, save=True)


# See attached other notebooks for 128x128x3 Code

# # VAE

# ## VAE GridSearch

# In[24]:


N_layers = 4


# In[25]:


from tensorflow.keras.layers import Dense, Flatten, Conv2D, Conv2DTranspose, Reshape

def build_conv_net(in_shape, out_shape, n_downsampling_layers=4, filters=128, out_activation='sigmoid'):
    """
    Build a basic convolutional network
    """
    default_args=dict(kernel_size=(3,3), strides=(2,2), padding='same', activation='relu')

    input = tf.keras.Input(shape=in_shape)
    x = Conv2D(filters=filters, name='enc_input', **default_args)(input) #** is the unpacking argument

    for _ in range(n_downsampling_layers): # _ if we dont care about the varible itself
        x = Conv2D(**default_args, filters=filters)(x)

    x = Flatten()(x)
    x = Dense(out_shape, activation=out_activation, name='enc_output')(x) # out_shape gives the shape of the latent space

    model = tf.keras.Model(inputs=input, outputs=x, name='Encoder')

    model.summary()
    return model


def build_deconv_net(latent_dim, n_upsampling_layers=4, filters=128, activation_out='sigmoid'):
    """
    Build a deconvolutional network for decoding/upscaling latent vectors

    When building the deconvolutional architecture, usually it is best to use the same layer sizes that
    were used in the downsampling network and the Conv2DTranspose layers are used instead of Conv2D layers.
    Using identical layers and hyperparameters ensures that the dimensionality of our output matches the
    shape of our input images.
    """
    input = tf.keras.Input(shape=(latent_dim,))
    x = Dense(4 * 4 * 64, input_dim=latent_dim, name='dec_input')(input)
    x = Reshape((4, 4, 64))(x) # This matches the output size of the downsampling architecture

    default_args=dict(kernel_size=(3,3), strides=(2,2), padding='same', activation='relu')

    for i in range(n_upsampling_layers):
        x = Conv2DTranspose(filters=filters, **default_args)(x) # reverse of a Conv layer, takes 1 input and maps kernel to output, addind where overlap

    # This last convolutional layer converts back to 3 channel RGB image
    x = Conv2D(filters=3, kernel_size=(3,3), padding='same', activation=activation_out, name='dec_output')(x)

    model = tf.keras.Model(inputs=input, outputs=x, name='Decoder')
    model.summary()
    return model


# In[26]:


class Sampling(tf.keras.layers.Layer):
    """
    Custom layer for the variational autoencoder
    It takes two vectors as input - one for means and other for variances of the latent variables described by a multimodal gaussian
    Its output is a latent vector randomly sampled from this distribution
    """
    def call(self, inputs):
        z_mean, z_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim)) # e for each of the pixels 64x64x3 in each image
        return z_mean + tf.exp(0.5 * z_var) * epsilon # log variance here

def build_vae(data_shape, latent_dim, filters=128):

    # Building the encoder - starts with a simple downsampling convolutional network
    encoder = build_conv_net(data_shape, latent_dim*2, filters=filters) # 2 since have both mu and sigma

    # Adding special sampling layer that uses the reparametrization trick
    z_mean = Dense(latent_dim)(encoder.output)
    z_var = Dense(latent_dim)(encoder.output)
    z = Sampling()([z_mean, z_var]) # z is our latent varible vector

    # Connecting the two encoder parts
    encoder = tf.keras.Model(inputs=encoder.input, outputs=z)

    # Defining the decoder which is a regular upsampling deconvolutional network
    decoder = build_deconv_net(latent_dim, activation_out='sigmoid', filters=filters)
    vae = tf.keras.Model(inputs=encoder.input, outputs=decoder(z))

    # Define a custom layer for the KL loss calculation
    class KLLossLayer(tf.keras.layers.Layer):
        def call(self, inputs):
            z_mean, z_var = inputs
            kl_loss = -0.5 * tf.reduce_sum(z_var - tf.square(z_mean) - tf.exp(z_var) + 1)
            # Add the KL loss to the model's losses
            self.add_loss(kl_loss / tf.cast(tf.keras.backend.prod(data_shape), tf.float32))
            return inputs  # Pass through the inputs unchanged

    # Apply the custom layer to z_mean and z_var
    _, _ = KLLossLayer()([z_mean, z_var])

    vae.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss='binary_crossentropy')

    return encoder, decoder, vae


# In[27]:


# Grid Search code

param_grid = {
    'latent_dim': [32,64,128,256],
    'filters': [32,64,128,256],
    'batch_size': [8],
    'n_epochs': [30],
}


for params in ParameterGrid(param_grid):
    latent_dim = params['latent_dim']
    n_f = params['filters']
    b_s = params['batch_size']
    n_epochs = params['n_epochs']
    saveM= False

    encoder, decoder, vae = build_vae(dataset.shape[1:], latent_dim, filters=n_f) # three seperate ones since want to be able to input latent varibles
    Title = f"z-VAE-(latent={latent_dim}, filters={n_f}, layers={N_layers})"
    history = {'VAE_loss': []}
    i=0
    for epoch in range(n_epochs):

        epoch_VAE_loss = 0.0


        h = vae.fit(x=dataset, y=dataset, epochs=1, batch_size=b_s)
        loss_value = h.history['loss'][0]   # reconstruction loss + KL
        history['VAE_loss'].append(loss_value)


        i+=1 
        if i == n_epochs:
            coefficient = 6                                 # You can tweak this coefficient to increase/decrease the std of the sampled vectors
            latent_vectors = np.random.randn(1, latent_dim) # Generate 9 random points in the latent space
            images = decoder(latent_vectors / coefficient)  # Feed the vectors scaled by the coefficient to the model
            grid_plot(images, epoch=Title, name='zimg', n=1, save=True)

        if saveM==True:
            decoder.save(f"Models/{Title}.keras")

        tf.keras.backend.clear_session()

    plt.figure(figsize=(8,5))
    plt.plot(history['VAE_loss'], 'r', label='VAE Loss')
    plt.title("VAE Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.savefig(f"Graphs/{Title}.png")
    plt.show()


# ## Saving Final Model

# In[28]:


# Training the VAE model

latent_dim = 64
n_f = 64
b_s = 16
N_layers = 4
n_epochs = 60
saveM= True

encoder, decoder, vae = build_vae(dataset.shape[1:], latent_dim, filters=n_f) # three seperate ones since want to be able to input latent varibles
Title = f"VAE-(latent={latent_dim}, filters={n_f}, layers={N_layers})"
history = {'VAE_loss': []}
i=0
for epoch in range(n_epochs):

    epoch_VAE_loss = 0.0


    h = vae.fit(x=dataset, y=dataset, epochs=1, batch_size=b_s)
    loss_value = h.history['loss'][0]   # reconstruction loss + KL
    history['VAE_loss'].append(loss_value)


    i+=1 
    if i == n_epochs:
        coefficient = 3                                # You can tweak this coefficient to increase/decrease the std of the sampled vectors
        latent_vectors = np.random.randn(1, latent_dim) # Generate 9 random points in the latent space
        images = decoder(latent_vectors / coefficient)  # Feed the vectors scaled by the coefficient to the model
        grid_plot(images, epoch=Title, name='Final Image', n=1, save=True)

    if saveM==True:
        decoder.save(f"Models/{Title}.keras")

    tf.keras.backend.clear_session()

plt.figure(figsize=(8,5))
plt.plot(history['VAE_loss'], 'r', label='VAE Loss')
plt.title("VAE Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.savefig(f"Graphs/{Title}.png")
plt.show()


# ## Latent Space interpolation

# ### GAN

# In[29]:


latent_dim = 256
noise = tf.random.normal(shape=(2, latent_dim))


# In[30]:


def interpolationVec(v1,v2,steps):
    dif = v2-v1
    latent_vec = []
    for i in range(steps+1):
        latent_vec.append(v1+ (v2-v1)*(i/steps))
    return latent_vec


# In[31]:


interVec = interpolationVec(noise[0],noise[1],15)
interVec = tf.stack(interVec, axis=0)


# In[32]:


from tensorflow import keras


# In[14]:


# Note here we import saved models
GAN_generator = keras.models.load_model("Models/Endgame-(latent=512, filters=256, layers=4).keras")
latent_dim = 512
noise = tf.random.normal(shape=(2, latent_dim))
interVec = interpolationVec(noise[0],noise[1],8)
interVec = tf.stack(interVec, axis=0)

epoch='test'
# After trained generator
generated_images = GAN_generator(interVec, training=False)
grid_plot(generated_images.numpy(), epoch, name='Interpolated Images', n=3, scale=True)


# In[8]:


# We will use this function to display the output of our models throughout this notebook
def grid_plot(images, epoch='', name='', n=3, save=False, scale=False):
    if scale:
        images = (images + 1) / 2.0
    for index in range(n * n):
        plt.subplot(n, n, 1 + index)
        plt.axis('off')
        plt.imshow(images[index])
    fig = plt.gcf()
    fig.suptitle(name + '  '+ str(epoch), fontsize=14)
    if save:
        plt.savefig(f'Graphs/{name + '  '+ str(epoch)}.png')

    plt.show()
    plt.close()


# In[22]:


GAN_generator = keras.models.load_model("Models/GAN-(latent=512, filters=256, layers=4).keras")
latent_dim = 512
noise = tf.random.normal(shape=(2, latent_dim))
interVec = interpolationVec(noise[0],noise[1],8)
interVec = tf.stack(interVec, axis=0)

epoch='-'
# After trained generator
generated_images = GAN_generator(interVec, training=False)
grid_plot(generated_images.numpy(), epoch, name='GAN-Interpolated Images, 4 layers', n=3,save=True, scale=True)
print(noise)


# In[62]:


GAN_generator = keras.models.load_model("Models/GAN5-(latent=512, filters=256, layers=5).keras")
latent_dim = 512
noise = tf.random.normal(shape=(2, latent_dim))
interVec = interpolationVec(noise[0],noise[1],8)
interVec = tf.stack(interVec, axis=0)

epoch='-'
# After trained generator
generated_images = GAN_generator(interVec, training=False)
grid_plot(generated_images.numpy(), epoch, name='GAN-Interpolated Images, 5 layers', n=3,save=True, scale=True)


# ## VAE

# In[73]:


VAE_generator = keras.models.load_model("Models/VAE-(latent=64, filters=64, layers=4).keras")
latent_dim = 64

end_vec = np.random.randn(2, latent_dim) 
epoch=''
interVec = interpolationVec(end_vec[0],end_vec[1],8)
interVec = tf.stack(interVec, axis=0)
images = VAE_generator(interVec)  
grid_plot(images, epoch, name='VAE-Interpolated images', n=3, save=True)


# ## Getting Outliers

# In[33]:


def grid_plot_outliers(images, epoch='', name='', n=3, save=False, scale=False):
    if scale:
        images = (images + 1) / 2.0
    for index in range(n ):
        plt.subplot(n,n, 1 + index)
        plt.axis('off')
        plt.imshow(images[index])
    fig = plt.gcf()
    fig.suptitle(name + '  '+ str(epoch), fontsize=14)
    if save:
        plt.savefig(f'Graphs/{name + '  '+ str(epoch)}.png')

    plt.show()
    plt.close()

grid_plot_outliers(dataset[np.array([164,253,284])], name='Outliers', n=3, save=True)
#140
#192
#289


# In[ ]:


#!jupyter nbconvert --to python A2_GenerativeModels.ipynb


# In[ ]:




