# %% [markdown]
# # Introduction to Deep Learning, Assignment 2, Task 2
# 
# # Introduction
# 
# 
# The goal of this assignment is to learn how to use encoder-decoder recurrent neural networks (RNNs). Specifically we will be dealing with a sequence to sequence problem and try to build recurrent models that can learn the principles behind simple arithmetic operations (**integer addition, subtraction and multiplication.**).
# 
# <img src="https://i.ibb.co/5Ky5pbk/Screenshot-2023-11-10-at-07-51-21.png" alt="Screenshot-2023-11-10-at-07-51-21" border="0" width="500"></a>
# 
# In this assignment you will be working with three different kinds of models, based on input/output data modalities:
# 1. **Text-to-text**: given a text query containing two integers and an operand between them (+ or -) the model's output should be a sequence of integers that match the actual arithmetic result of this operation
# 2. **Image-to-text**: same as above, except the query is specified as a sequence of images containing individual digits and an operand.
# 3. **Text-to-image**: the query is specified in text format as in the text-to-text model, however the model's output should be a sequence of images corresponding to the correct result.
# 
# 
# ### Description
# Let us suppose that we want to develop a neural network that learns how to add or subtract
# two integers that are at most two digits long. For example, given input strings of 5 characters: ‘81+24’ or
# ’41-89’ that consist of 2 two-digit long integers and an operand between them, the network should return a
# sequence of 3 characters: ‘105 ’ or ’-48 ’ that represent the result of their respective queries. Additionally,
# we want to build a model that generalizes well - if the network can extract the underlying principles behind
# the ’+’ and ’-’ operands and associated operations, it should not need too many training examples to generate
# valid answers to unseen queries. To represent such queries we need 13 unique characters: 10 for digits (0-9),
# 2 for the ’+’ and ’-’ operands and one for whitespaces ’ ’ used as padding.
# The example above describes a text-to-text sequence mapping scenario. However, we can also use different
# modalities of data to represent our queries or answers. For that purpose, the MNIST handwritten digit
# dataset is going to be used again, however in a slightly different format. The functions below will be used to create our datasets.
# 
# ---
# 
# *To work on this notebook you should create a copy of it.*
# 
# When using the Lab Computers, download the Jupyter Notebook to one of the machines first.
# 
# If you want to use Google Colab, you should first copy this notebook and enable GPU runtime in 'Runtime -> Change runtime type -> Hardware acceleration -> GPU **OR** TPU'.
# 

# %% [markdown]
# # Function definitions for creating the datasets
# 
# First we need to create our datasets that are going to be used for training our models.
# 
# In order to create image queries of simple arithmetic operations such as '15+13' or '42-10' we need to create images of '+' and '-' signs using ***open-cv*** library. We will use these operand signs together with the MNIST dataset to represent the digits.

# %%
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
import numpy as np
import tensorflow as tf
import random
from sklearn.model_selection import train_test_split

from tensorflow.keras.layers import Dense, RNN, LSTM, Flatten, TimeDistributed, LSTMCell, InputLayer
from tensorflow.keras.layers import RepeatVector, Conv2D, SimpleRNN, GRU, Reshape, ConvLSTM2D, Conv2DTranspose

# %%
from scipy.ndimage import rotate


# Create plus/minus operand signs
def generate_images(number_of_images=50, sign='-'):
    blank_images = np.zeros([number_of_images, 28, 28])  # Dimensionality matches the size of MNIST images (28x28)
    x = np.random.randint(12, 16, (number_of_images, 2)) # Randomized x coordinates
    y1 = np.random.randint(6, 10, number_of_images)       # Randomized y coordinates
    y2 = np.random.randint(18, 22, number_of_images)     # -||-

    for i in range(number_of_images): # Generate n different images
        cv2.line(blank_images[i], (y1[i], x[i,0]), (y2[i], x[i, 1]), (255,0,0), 2, cv2.LINE_AA)     # Draw lines with randomized coordinates
        if sign == '+':
            cv2.line(blank_images[i], (x[i,0], y1[i]), (x[i, 1], y2[i]), (255,0,0), 2, cv2.LINE_AA) # Draw lines with randomized coordinates

    return blank_images

def show_generated(images, n=5):
    plt.figure(figsize=(2, 2))
    for i in range(n**2):
        plt.subplot(n, n, i+1)
        plt.axis('off')
        plt.imshow(images[i])
    plt.show()

show_generated(generate_images())
show_generated(generate_images(sign='+'))

# %%
def create_data(highest_integer, num_addends=2, operands=['+', '-']):
    """
    Creates the following data for all pairs of integers up to [1:highest integer][+/-][1:highest_integer]:

    @return:
    X_text: '51+21' -> text query of an arithmetic operation (5)
    X_img : Stack of MNIST images corresponding to the query (5 x 28 x 28) -> sequence of 5 images of size 28x28
    y_text: '72' -> answer of the arithmetic text query
    y_img :  Stack of MNIST images corresponding to the answer (3 x 28 x 28)

    Images for digits are picked randomly from the whole MNIST dataset.
    """

    num_indices = [np.where(MNIST_labels==x) for x in range(10)]
    num_data = [MNIST_data[inds] for inds in num_indices]
    image_mapping = dict(zip(unique_characters[:10], num_data))
    image_mapping['-'] = generate_images()
    image_mapping['+'] = generate_images(sign='+')
    image_mapping['*'] = generate_images(sign='*')
    image_mapping[' '] = np.zeros([1, 28, 28])

    X_text, X_img, y_text, y_img = [], [], [], []

    for i in range(highest_integer + 1):      # First addend
        for j in range(highest_integer + 1):  # Second addend
            for sign in operands: # Create all possible combinations of operands
                query_string = to_padded_chars(str(i) + sign + str(j), max_len=max_query_length, pad_right=True)
                query_image = []
                for n, char in enumerate(query_string):
                    image_set = image_mapping[char]
                    index = np.random.randint(0, len(image_set), 1)
                    query_image.append(image_set[index].squeeze())

                result = eval(query_string)
                result_string = to_padded_chars(result, max_len=max_answer_length, pad_right=True)
                result_image = []
                for n, char in enumerate(result_string):
                    image_set = image_mapping[char]
                    index = np.random.randint(0, len(image_set), 1)
                    result_image.append(image_set[index].squeeze())

                X_text.append(query_string)
                X_img.append(np.stack(query_image))
                y_text.append(result_string)
                y_img.append(np.stack(result_image))

    return np.stack(X_text), np.stack(X_img)/255., np.stack(y_text), np.stack(y_img)/255.

def to_padded_chars(integer, max_len=3, pad_right=False):
    """
    Returns a string of len()=max_len, containing the integer padded with ' ' on either right or left side
    """
    length = len(str(integer))
    padding = (max_len - length) * ' '
    if pad_right:
        return str(integer) + padding
    else:
        return padding + str(integer)


# %% [markdown]
# # Creating our data
# 
# The dataset consists of 20000 samples that (additions and subtractions between all 2-digit integers) and they have two kinds of inputs and label modalities:
# 
#   **X_text**: strings containing queries of length 5: ['  1+1  ', '11-18', ...]
# 
#   **X_image**: a stack of images representing a single query, dimensions: [5, 28, 28]
# 
#   **y_text**: strings containing answers of length 3: ['  2', '156']
# 
#   **y_image**: a stack of images that represents the answer to a query, dimensions: [3, 28, 28]

# %%
# Illustrate the generated query/answer pairs

unique_characters = '0123456789+- '       # All unique characters that are used in the queries (13 in total: digits 0-9, 2 operands [+, -], and a space character ' '.)
highest_integer = 99                      # Highest value of integers contained in the queries

max_int_length = len(str(highest_integer))# Maximum number of characters in an integer
max_query_length = max_int_length * 2 + 1 # Maximum length of the query string (consists of two integers and an operand [e.g. '22+10'])
max_answer_length = 3    # Maximum length of the answer string (the longest resulting query string is ' 1-99'='-98')

# Create the data (might take around a minute)
(MNIST_data, MNIST_labels), _ = tf.keras.datasets.mnist.load_data()
X_text, X_img, y_text, y_img = create_data(highest_integer)
print(X_text.shape, X_img.shape, y_text.shape, y_img.shape)


## Display the samples that were created
def display_sample(n):
    labels = ['X_img:', 'y_img:']
    for i, data in enumerate([X_img, y_img]):
        plt.subplot(1,2,i+1)
        # plt.set_figheight(15)
        plt.axis('off')
        plt.title(labels[i])
        plt.imshow(np.hstack(data[n]), cmap='gray')
    print('='*50, f'\nQuery #{n}\n\nX_text: "{X_text[n]}" = y_text: "{y_text[n]}"')
    plt.show()

for _ in range(10):
    display_sample(np.random.randint(0, 10000, 1)[0])

# %% [markdown]
# ## Helper functions
# 
# The functions below will help with input/output of the data.

# %%
# One-hot encoding/decoding the text queries/answers so that they can be processed using RNNs
# You should use these functions to convert your strings and read out the output of your networks

def encode_labels(labels, max_len=3):
  n = len(labels)
  length = len(labels[0])
  char_map = dict(zip(unique_characters, range(len(unique_characters))))
  one_hot = np.zeros([n, length, len(unique_characters)])
  for i, label in enumerate(labels):
      m = np.zeros([length, len(unique_characters)])
      for j, char in enumerate(label):
          m[j, char_map[char]] = 1
      one_hot[i] = m

  return one_hot


def decode_labels(labels):
    pred = np.argmax(labels, axis=2)
    predicted = [''.join([unique_characters[i] for i in j]) for j in pred]

    return predicted

X_text_onehot = encode_labels(X_text)
y_text_onehot = encode_labels(y_text)

print(X_text_onehot.shape, y_text_onehot.shape)

# %% [markdown]
# ---
# ---
# 
# ## I. Text-to-text RNN model
# 
# The following code showcases how Recurrent Neural Networks (RNNs) are built using Keras. Several new layers are going to be used:
# 
# 1. LSTM
# 2. TimeDistributed
# 3. RepeatVector
# 
# The code cell below explains each of these new components.
# 
# <img src="https://i.ibb.co/NY7FFTc/Screenshot-2023-11-10-at-09-27-25.png" alt="Screenshot-2023-11-10-at-09-27-25" border="0" width="500"></a>
# 

# %%
def build_text2text_model():

    # We start by initializing a sequential model
    text2text = tf.keras.Sequential()

    # "Encode" the input sequence using an RNN, producing an output of size 256.
    # In this case the size of our input vectors is [5, 13] as we have queries of length 5 and 13 unique characters. Each of these 5 elements in the query will be fed to the network one by one,
    # as shown in the image above (except with 5 elements).
    # Hint: In other applications, where your input sequences have a variable length (e.g. sentences), you would use input_shape=(None, unique_characters).
    text2text.add(LSTM(256, input_shape=(None, len(unique_characters))))

    # As the decoder RNN's input, repeatedly provide with the last output of RNN for each time step. Repeat 3 times as that's the maximum length of the output (e.g. '  1-99' = '-98')
    # when using 2-digit integers in queries. In other words, the RNN will always produce 3 characters as its output.
    text2text.add(RepeatVector(max_answer_length))

    # By setting return_sequences to True, return not only the last output but all the outputs so far in the form of (num_samples, timesteps, output_dim). This is necessary as TimeDistributed in the below expects
    # the first dimension to be the timesteps.
    text2text.add(LSTM(256, return_sequences=True))

    # Apply a dense layer to the every temporal slice of an input. For each of step of the output sequence, decide which character should be chosen.
    text2text.add(TimeDistributed(Dense(len(unique_characters), activation='softmax')))

    # Next we compile the model using categorical crossentropy as our loss function.
    text2text.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    text2text.summary()

    return text2text

# %%
## Your code (look at the assignment description for your tasks for text-to-text model):
## Your first task is to fit the text2text model using X_text and y_text
train = 0.4
valid = 0.1
test = 0.5

# shuffle indices
shuffle = np.arange(len(X_img))
np.random.shuffle(shuffle)

n = len(X_img)
train_end = int(train * n)
valid_end = int((train + valid) * n)
train_idx = shuffle[:train_end]
valid_idx = shuffle[train_end:valid_end]
test_idx  = shuffle[valid_end:]

# apply indices to image data
X_img_train, y_img_train = X_img[train_idx], y_img[train_idx]
X_img_valid, y_img_valid = X_img[valid_idx], y_img[valid_idx]
X_img_test,  y_img_test  = X_img[test_idx],  y_img[test_idx]

# apply indices to one-hot text data
X_text_onehot_train, y_text_onehot_train = X_text_onehot[train_idx], y_text_onehot[train_idx]
X_text_onehot_valid, y_text_onehot_valid = X_text_onehot[valid_idx], y_text_onehot[valid_idx]
X_text_onehot_test,  y_text_onehot_test  = X_text_onehot[test_idx],  y_text_onehot[test_idx]

# %%
text2text = build_text2text_model()

text2text_history = text2text.fit(X_text_onehot_train, y_text_onehot_train, epochs=30,
                                    validation_data=(X_text_onehot_valid,y_text_onehot_valid))

text2text.evaluate(X_text_onehot_test, y_text_onehot_test)

# %%
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

y_pred_probs = text2text.predict(X_text_onehot_test)
y_pred = np.argmax(y_pred_probs, axis=-1)
y_true = np.argmax(y_text_onehot_test, axis=-1)

digit_labels = list(range(13))
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for pos in range(3):
    cm = confusion_matrix(y_true[:, pos], y_pred[:, pos], labels=digit_labels)
    
    sns.heatmap(
        cm,
        annot=False,
        cmap="Blues",
        xticklabels=digit_labels,
        yticklabels=digit_labels,
        ax=axes[pos]
    )
    
    axes[pos].set_title(f"Output Digit Position {pos + 1}")
    axes[pos].set_xlabel("Predicted Digit")
    axes[pos].set_ylabel("True Digit")

plt.tight_layout()
plt.show()


# %% [markdown]
# 
# ---
# ---
# 
# ## II. Image to text RNN Model
# 
# Hint: There are two ways of building the encoder for such a model - again by using the regular LSTM cells (with flattened images as input vectors) or recurrect convolutional layers [ConvLSTM2D](https://keras.io/api/layers/recurrent_layers/conv_lstm2d/).
# 
# The goal here is to use **X_img** as inputs and **y_text** as outputs.

# %%
## Your code
def build_image2text_model():

    # We start by initializing a sequential model
    image2text = tf.keras.Sequential()

    image2text.add(InputLayer(input_shape=(5, 28*28)))

    # "Encode" the input sequence using an RNN, producing an output of size 256.
    # In this case the size of our input vectors is [5, 13] as we have queries of length 5 and 13 unique characters. Each of these 5 elements in the query will be fed to the network one by one,
    # as shown in the image above (except with 5 elements).
    # Hint: In other applications, where your input sequences have a variable length (e.g. sentences), you would use input_shape=(None, unique_characters).
    image2text.add(LSTM(256))

    # As the decoder RNN's input, repeatedly provide with the last output of RNN for each time step. Repeat 3 times as that's the maximum length of the output (e.g. '  1-99' = '-98')
    # when using 2-digit integers in queries. In other words, the RNN will always produce 3 characters as its output.
    image2text.add(RepeatVector(max_answer_length))

    # By setting return_sequences to True, return not only the last output but all the outputs so far in the form of (num_samples, timesteps, output_dim). This is necessary as TimeDistributed in the below expects
    # the first dimension to be the timesteps.
    image2text.add(LSTM(256, return_sequences=True))

    # Apply a dense layer to the every temporal slice of an input. For each of step of the output sequence, decide which character should be chosen.
    image2text.add(TimeDistributed(Dense(len(unique_characters), activation='softmax')))

    # Next we compile the model using categorical crossentropy as our loss function.
    image2text.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    image2text.summary()

    return image2text




# %%
## Your code
def build_image2text_cnn_model():

    # We start by initializing a sequential model
    image2text = tf.keras.Sequential()

    image2text.add(InputLayer(input_shape=(5, 28, 28, 1)))

    # "Encode" the input sequence using an RNN, producing an output of size 256.
    # In this case the size of our input vectors is [5, 13] as we have queries of length 5 and 13 unique characters. Each of these 5 elements in the query will be fed to the network one by one,
    # as shown in the image above (except with 5 elements).
    # Hint: In other applications, where your input sequences have a variable length (e.g. sentences), you would use input_shape=(None, unique_characters).
    image2text.add(ConvLSTM2D(32, (3,3)))

    image2text.add(Flatten())

    # As the decoder RNN's input, repeatedly provide with the last output of RNN for each time step. Repeat 3 times as that's the maximum length of the output (e.g. '  1-99' = '-98')
    # when using 2-digit integers in queries. In other words, the RNN will always produce 3 characters as its output.
    image2text.add(RepeatVector(max_answer_length))

    # By setting return_sequences to True, return not only the last output but all the outputs so far in the form of (num_samples, timesteps, output_dim). This is necessary as TimeDistributed in the below expects
    # the first dimension to be the timesteps.
    image2text.add(LSTM(256, return_sequences=True))

    # Apply a dense layer to the every temporal slice of an input. For each of step of the output sequence, decide which character should be chosen.
    image2text.add(TimeDistributed(Dense(len(unique_characters), activation='softmax')))

    # Next we compile the model using categorical crossentropy as our loss function.
    image2text.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    image2text.summary()

    return image2text




# %%
train = 0.4
valid = 0.1
test = 0.5

shuffle = np.arange(len(X_img))
np.random.shuffle(shuffle)
train_idx = shuffle[:int(train*len(X_img))]
valid_idx = shuffle[int(train*len(X_img)):int((train+valid)*len(X_img))]
test_idx = shuffle[int((1-test)*len(X_img)):]

X_img_train, y_img_train = X_img[train_idx], y_img[train_idx]
X_img_valid, y_img_valid = X_img[valid_idx], y_img[valid_idx]
X_img_test, y_img_test = X_img[test_idx], y_img[test_idx]

X_text_onehot_train, y_text_onehot_train = X_text_onehot[train_idx], y_text_onehot[train_idx]
X_text_onehot_valid, y_text_onehot_valid = X_text_onehot[valid_idx], y_text_onehot[valid_idx]
X_text_onehot_test, y_text_onehot_test = X_text_onehot[test_idx], y_text_onehot[test_idx]

# %%
image2text_flat = build_image2text_model()

X_img_train_flat = X_img_train.reshape(X_img_train.shape[0],X_img_train.shape[1],X_img_train.shape[2]*X_img_train.shape[3])
X_img_valid_flat = X_img_valid.reshape(X_img_valid.shape[0],X_img_valid.shape[1],X_img_valid.shape[2]*X_img_valid.shape[3])
X_img_test_flat = X_img_test.reshape(X_img_test.shape[0],X_img_test.shape[1],X_img_test.shape[2]*X_img_test.shape[3])

image2text_flat_history = image2text_flat.fit(X_img_train_flat, y_text_onehot_train, epochs=30,
                                    validation_data=(X_img_valid_flat,y_text_onehot_valid))

image2text_flat.evaluate(X_img_test_flat, y_text_onehot_test)

# %%
image2text = build_image2text_cnn_model()

image2text_history = image2text.fit(X_img_train, y_text_onehot_train, epochs=30,
                                    validation_data=(X_img_valid,y_text_onehot_valid))

image2text.evaluate(X_img_test, y_text_onehot_test)

# %%
## Your code
def build_image2text_model_2():

    # We start by initializing a sequential model
    image2text = tf.keras.Sequential()

    image2text.add(InputLayer(input_shape=(5, 28*28)))

    # "Encode" the input sequence using an RNN, producing an output of size 256.
    # In this case the size of our input vectors is [5, 13] as we have queries of length 5 and 13 unique characters. Each of these 5 elements in the query will be fed to the network one by one,
    # as shown in the image above (except with 5 elements).
    # Hint: In other applications, where your input sequences have a variable length (e.g. sentences), you would use input_shape=(None, unique_characters).
    image2text.add(LSTM(256, return_sequences=True))

    image2text.add(LSTM(256, return_sequences=True))

    image2text.add(LSTM(256, return_sequences=True))

    image2text.add(LSTM(256))

    # As the decoder RNN's input, repeatedly provide with the last output of RNN for each time step. Repeat 3 times as that's the maximum length of the output (e.g. '  1-99' = '-98')
    # when using 2-digit integers in queries. In other words, the RNN will always produce 3 characters as its output.
    image2text.add(RepeatVector(max_answer_length))

    # By setting return_sequences to True, return not only the last output but all the outputs so far in the form of (num_samples, timesteps, output_dim). This is necessary as TimeDistributed in the below expects
    # the first dimension to be the timesteps.
    image2text.add(LSTM(256, return_sequences=True))

    # Apply a dense layer to the every temporal slice of an input. For each of step of the output sequence, decide which character should be chosen.
    image2text.add(TimeDistributed(Dense(len(unique_characters), activation='softmax')))

    # Next we compile the model using categorical crossentropy as our loss function.
    image2text.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    image2text.summary()

    return image2text

# %%
image2text_2 = build_image2text_model_2()

image2text_2_history = image2text_2.fit(X_img_train_flat, y_text_onehot_train, epochs=30,
                                        validation_data=(X_img_valid_flat,y_text_onehot_valid))

image2text_2.evaluate(X_img_test_flat, y_text_onehot_test)

# %%
histories = [image2text_flat_history,image2text_history,image2text_2_history]
labels = {'loss':'Training loss', 'val_loss':'Validation loss', 'accuracy':'Training acc.', 'val_accuracy':'Validation acc.'}
titles = ['Flattened','Convolutional','Flattened, more LSTM']
fig, ax = plt.subplots(2,3,figsize=(6,3),sharex=True,sharey='row',layout='constrained')
for i, history in enumerate(histories):
    for j, param in enumerate(['loss','accuracy']):
        for val in ['','val_']:
            ax[j,i].plot(np.arange(1,31,1),history.history[val+param],label=labels[val+param])
    ax[0,i].set_title(titles[i])
    ax[1,i].set_xlabel('Epoch')
ax[0,0].set_ylabel('Loss')
ax[1,0].set_ylabel('Accuracy')
for j in [0,1]:
    ax[j,2].legend()
plt.savefig('image2text_performance.pdf',bbox_inches='tight')
plt.show()

# %%
from sklearn.metrics import confusion_matrix
from mpl_toolkits.axes_grid1 import make_axes_locatable

y_text_test = decode_labels(y_text_onehot_test)
X_img_test_list = [X_img_test_flat,X_img_test,X_img_test_flat]
save_names = ['flat','cnn','flat_2']
y_text_pred_list = []

for i, model in enumerate([image2text_flat,image2text,image2text_2]):
    # Calculate and decode predictions
    model_pred = model.predict(X_img_test_list[i])
    y_text_pred = decode_labels(model_pred)
    y_text_pred_list.append(y_text_pred)

    # Sort labels by number, with improper labels (not a number) at start
    labels_init = np.unique(np.append(y_text_test,y_text_pred))
    not_ints = []
    for label in labels_init:
        try:
            label_int = int(label)
        except ValueError:
            not_ints.append(label)
    labels_nums = [label for label in labels_init if label not in not_ints]
    label_sorting = np.argsort(np.array([int(label) for label in labels_nums]))
    labels = not_ints
    labels.extend(list(np.array(labels_nums)[label_sorting]))

    # Plot confusion matrix
    confusion = confusion_matrix(y_text_test,y_text_pred,labels=labels,normalize='true')
    fig, ax = plt.subplots(1,1,figsize=(20,20))
    plot = ax.imshow(confusion)
    ax.set_xlabel('Predicted',fontsize=25)
    ax.set_ylabel('Actual',fontsize=25)
    ax.set_xticks(ticks=np.arange(len(confusion)),labels=labels,fontsize=4,rotation=90)
    ax.set_yticks(ticks=np.arange(len(confusion)),labels=labels,fontsize=4)
    cax = make_axes_locatable(ax).append_axes("right", size="2.5%", pad=0.1)
    cbar = fig.colorbar(plot,cax=cax,aspect=50)
    cbar.set_label('Predicted fraction',fontsize=25)
    cbar.set_ticks(ticks=np.arange(0,1.2,0.2),labels=[0.0,0.2,0.4,0.6,0.8,1.0],fontsize=25)
    plt.savefig(f'RNN_confusion_matrix_{save_names[i]}.pdf',bbox_inches='tight')
    plt.show()

# %%
X_text_test = decode_labels(X_text_onehot_test)

fig, ax = plt.subplots(3,1,figsize=(6,3),sharex=True,layout='constrained')
for i, y_text_pred in enumerate(y_text_pred_list):
    incorrect_mask = (np.array(y_text_pred) != np.array(y_text_test))

    print(np.sum(incorrect_mask)/len(X_text_test))

    X_text_test_incorrect = np.array(X_text_test)[incorrect_mask]

    incorrect_counts = np.zeros(len(unique_characters))
    total_counts = np.zeros(len(unique_characters))
    for j, char in enumerate(unique_characters):
        for string in X_text_test_incorrect:
            if char in string:
                incorrect_counts[j] = incorrect_counts[j] + 1
        for string in X_text_test:
            if char in string:
                total_counts[j] = total_counts[j] + 1

    ax[i].plot((incorrect_counts/len(X_text_test_incorrect))/(total_counts/len(X_text_test)))
    ax[i].axhline(1,c='k',ls='--')
    ax[i].set_xticks(ticks=np.arange(len(total_counts)),labels=list(unique_characters.replace(' ','_')))
    ax[i].annotate(titles[i],(0.99,0.97),xycoords='axes fraction',ha='right',va='top')
ax[2].set_xlabel('Character')
fig.supylabel('Relative occurrence')
plt.savefig(f'RNN_symbol_occurrence.pdf',bbox_inches='tight')
plt.show()

# %% [markdown]
# ---
# ---
# 
# ## III. Text to image RNN Model
# 
# Hint: to make this model work really well you could use deconvolutional layers in your decoder (you might need to look up ***Conv2DTranspose*** layer). However, regular vector-based decoder will work as well.
# 
# The goal here is to use **X_text** as inputs and **y_img** as outputs.

# %%
# Your code
latent_dim = 256

def build_text2image_model():
    text2image = tf.keras.Sequential()
    #Encoder
    text2image.add(LSTM(latent_dim, input_shape=(None, len(unique_characters))))
    text2image.add(RepeatVector(max_answer_length))

    #Decoder
    text2image.add(LSTM(latent_dim, return_sequences=True))

    text2image.add(TimeDistributed(Dense(128,activation="relu")))
    text2image.add(TimeDistributed(Dense(28 * 28, activation="sigmoid")))
    text2image.add(Reshape((max_answer_length, 28, 28, 1)))

    text2image.compile(loss='mse', optimizer='adam', metrics=['mae'])
    text2image.summary()


    return text2image

# %%
y_img.shape

# %%
model = build_text2image_model()

# %%
X_text_train, X_text_test,X_img_train,X_img_test, y_text_test, y_text_train, y_img_train, y_img_test = train_test_split(
    X_text, X_img, y_text, y_img, test_size=0.2, random_state=42
)

X_text_train_onehot = encode_labels(X_text_train)
X_text_test_onehot = encode_labels(X_text_test)

y_text_train_onehot = encode_labels(y_text_train)
y_text_test_onehot = encode_labels(y_text_test)

print(X_text_train.shape)

# %%
X_train, X_test, y_train, y_test = train_test_split(
    X_text_onehot, y_img, train_size=(0.8)
)

# %%
print(X_text_onehot.shape)
print(y_img_train.shape)

# %%
history_text2image_single = model.fit(
    X_text_train_onehot,
    y_img_train,
    validation_split=0.1,
    epochs=70,
    batch_size=32
)


# %%
test_loss, test_accuracy = model.evaluate(X_test, y_test)

# %%
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

ax1.plot(history_text2image_single.history['loss'], label='train loss')
ax1.plot(history_text2image_single.history['val_loss'], label='validation loss')
ax1.set_title("Model loss")
ax1.set_xlabel("Epochs")
ax1.set_ylabel("Loss")
ax1.legend()

ax2.plot(history_text2image_single.history['mae'], label='train mae')
ax2.plot(history_text2image_single.history['val_mae'], label='validation mae')
ax2.set_title("Model MAE")
ax2.set_xlabel("Epochs")
ax2.set_ylabel("MAE")   
ax2.legend()

plt.tight_layout()
plt.show()


# %%
def generate_image_from_text(n):
    labels = ['y_pred_img','y_test_img']

    for i, data in enumerate([pred, y_test]):
        plt.subplot(1,2,i+1)
        plt.axis('off')
        plt.title(labels[i])
        plt.imshow(np.hstack(data[n]), cmap='gray')
    plt.show()


# %%
pred = model.predict(X_test)
print(pred.shape)

# %%
for i in range(10):
  generate_image_from_text(i)

# %% [markdown]
# ### Additional LSTM layers

# %%
latent_dim = 256

def build_text2image_model_adl():
    text2image = tf.keras.Sequential()
    #Encoder
    text2image.add(LSTM(latent_dim,return_sequences=True, input_shape=(None, len(unique_characters))))
    text2image.add(LSTM(latent_dim,return_sequences=True))
    text2image.add(LSTM(latent_dim))
    text2image.add(RepeatVector(max_answer_length))

    #Decoder
    text2image.add(LSTM(latent_dim, return_sequences=True))

    text2image.add(TimeDistributed(Dense(128,activation="relu")))
    text2image.add(TimeDistributed(Dense(28 * 28, activation="sigmoid")))
    text2image.add(Reshape((max_answer_length, 28, 28, 1)))

    text2image.compile(loss='mse', optimizer='adam', metrics=['mae'])
    text2image.summary()


    return text2image

# %%
adl_model = build_text2image_model_adl()

# %%
history_text2image_adl = adl_model.fit(
    X_text_train_onehot,
    y_img_train,
    validation_split=0.1,
    epochs=70,
    batch_size=32
)

# %%
test_loss, test_accuracy = adl_model.evaluate(X_text_test_onehot, y_img_test)

# %%
def generate_image_from_text_adl(n):
    labels = ['y_pred_img','y_test_img']

    for i, data in enumerate([pred, y_test]):
        plt.subplot(1,2,i+1)
        plt.axis('off')
        plt.title(labels[i])
        plt.imshow(np.hstack(data[n]), cmap='gray')
    plt.show()

pred = adl_model.predict(X_test)


# %%
for i in range(5):
  generate_image_from_text_adl(i)

# %%
latent_dim = 256

def build_text2image_model():
    text2image = tf.keras.Sequential()
    #Encoder
    text2image.add(LSTM(latent_dim, input_shape=(None, len(unique_characters))))
    text2image.add(RepeatVector(max_answer_length))

    #Decoder
    text2image.add(LSTM(latent_dim, return_sequences=True))


    text2image.add(TimeDistributed(Dense(128,activation="relu")))
    text2image.add(TimeDistributed(Dense(14 * 14 * 32, activation="relu")))
    text2image.add(Reshape((max_answer_length, 14, 14, 32)))

    # Deconvolution layer to upsample and refine
    text2image.add(TimeDistributed(Conv2DTranspose(16, kernel_size=3, strides=2, padding='same', activation='relu')))
    text2image.add(TimeDistributed(Conv2DTranspose(1, kernel_size=3, strides=1, padding='same', activation='sigmoid')))


    text2image.compile(loss='mse', optimizer='adam', metrics=['mae'])
    text2image.summary()


    return text2image

# %%
conv_model = build_text2image_model()

# %%
history_text2image_conv = conv_model.fit(
    X_text_train_onehot,
    y_img_train,
    validation_split=0.1,
    epochs=70,
    batch_size=32
)


# %%
test_loss, test_accuracy = conv_model.evaluate(X_text_test_onehot, y_img_test)

# %%
def generate_image_from_text_conv(n):
    labels = ['y_pred_img','y_test_img']

    for i, data in enumerate([pred, y_test]):
        plt.subplot(1,2,i+1)
        plt.axis('off')
        plt.title(labels[i])
        plt.imshow(np.hstack(data[n]), cmap='gray')
    plt.show()

pred = conv_model.predict(X_test)


# %%
import matplotlib.pyplot as plt

fig, axes = plt.subplots(3, 2, figsize=(14, 12)) 
fig.suptitle("Text-to-Image Model Training Curves", fontsize=16, y=0.95)

axes[0, 0].plot(history_text2image_conv.history['loss'])
axes[0, 0].plot(history_text2image_conv.history['val_loss'])
axes[0, 0].set_title('Conv Model - Loss')
axes[0, 0].set_xlabel('Epochs')
axes[0, 0].set_ylabel('Loss')
axes[0, 0].legend(['Train Loss', 'Val Loss'])

axes[0, 1].plot(history_text2image_conv.history['mae'])
axes[0, 1].plot(history_text2image_conv.history['val_mae'])
axes[0, 1].set_title('Conv Model - MAE')
axes[0, 1].set_xlabel('Epochs')
axes[0, 1].set_ylabel('MAE')
axes[0, 1].legend(['Train MAE', 'Val MAE'])

axes[1, 0].plot(history_text2image_adl.history['loss'])
axes[1, 0].plot(history_text2image_adl.history['val_loss'])
axes[1, 0].set_title('ADL Model - Loss')
axes[1, 0].set_xlabel('Epochs')
axes[1, 0].set_ylabel('Loss')
axes[1, 0].legend(['Train Loss', 'Val Loss'])

axes[1, 1].plot(history_text2image_adl.history['mae'])
axes[1, 1].plot(history_text2image_adl.history['val_mae'])
axes[1, 1].set_title('ADL Model - MAE')
axes[1, 1].set_xlabel('Epochs')
axes[1, 1].set_ylabel('MAE')
axes[1, 1].legend(['Train MAE', 'Val MAE'])

axes[2, 0].plot(history_text2image_single.history['loss'])
axes[2, 0].plot(history_text2image_single.history['val_loss'])
axes[2, 0].set_title('Single Model - Loss')
axes[2, 0].set_xlabel('Epochs')
axes[2, 0].set_ylabel('Loss')
axes[2, 0].legend(['Train Loss', 'Val Loss'])

axes[2, 1].plot(history_text2image_single.history['mae'])
axes[2, 1].plot(history_text2image_single.history['val_mae'])
axes[2, 1].set_title('Single Model - MAE')
axes[2, 1].set_xlabel('Epochs')
axes[2, 1].set_ylabel('MAE')
axes[2, 1].legend(['Train MAE', 'Val MAE'])

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()


