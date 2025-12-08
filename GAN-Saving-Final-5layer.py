#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
os.environ['TF_USE_LEGACY_KERAS'] = '1'

from tqdm import tqdm
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.model_selection import ParameterGrid


# In[2]:


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


# In[11]:


def load_real_samples(scale=False):
    X = np.load('stanford_dogs_128x128.npy')[:20000]  # or remove slicing to load all
    if scale:
        X = (X - 127.5) * 2
    return X / 255.0

dataset = load_real_samples()


# In[12]:


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


# In[13]:


# HyperHyper param
n_epochs = 15
N_layers = 5


# In[14]:


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


# In[15]:


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


# In[16]:


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


# In[17]:


def train_gan(generator, discriminator, gan, dataset, latent_dim, n_f, n_epochs, batch_size=64, save=False):
    """
    Train the GAN with memory-efficient updates and clear session management.
    """
    filters = n_f
    Title = f"Endgame_5-(latent={latent_dim}, filters={filters}, layers={N_layers})"
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


# In[18]:


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


# In[1]:


#!jupyter nbconvert --to python GAN-Saving-Final-5layer.ipynb


# In[ ]:




