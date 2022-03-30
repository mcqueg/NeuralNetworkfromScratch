  As far as your question on the Back-Propagation neural network accuracy, after checking your code everything seems to be correct except for the accuracy calculation and (probably for the choice of transfer function you are using (in your case softmax)). The following is my implementation of ANN on a similar dataset to the one you are using, note that the difference in accuracy calculation and the transfer function. You might also to change the learning rate and the number of hidden layers and the number of neurons in each hidden layer. However, if it is OK you can implement the same project but using deep learning and convolutional neural networks, you will definitely get better results (please see below and also check: https://medium.com/@nutanbhogendrasharma/tensorflow-build-custom-convolutional-neural-network-with-mnist-dataset-d4c36cd52114 )

 

for itr in range(iterations):   


    # feedforward propagation

    # on hidden layer

    Z1 = np.dot(x_train, W1)

    A1 = sigmoid(Z1)


    # on output layer

    Z2 = np.dot(A1, W2)

    A2 = sigmoid(Z2)

   

    # Calculating error

    mse = mean_squared_error(A2, y_train)

    acc = accuracy(A2, y_train)

    results=results.append({"mse":mse, "accuracy":acc},ignore_index=True )


    # backpropagation

    E1 = A2 - y_train

    dW1 = E1 * A2 * (1 - A2)


    E2 = np.dot(dW1, W2.T)

    dW2 = E2 * A1 * (1 - A1)


    # weight updates

    W2_update = np.dot(A1.T, dW1) / N

    W1_update = np.dot(x_train.T, dW2) / N

 

    W2 = W2 - learning_rate * W2_update

    W1 = W1 - learning_rate * W1_update

 


def sigmoid(x):

    return 1 / (1 + np.exp(-x))

 

def mean_squared_error(y_pred, y_true):

    return ((y_pred - y_true)**2).sum() / (2*y_pred.size)

   

def accuracy(y_pred, y_true):

    acc = y_pred.argmax(axis=1) == y_true.argmax(axis=1)

    return acc.mean()

 

# feedforward

Z1 = np.dot(x_test, W1)

A1 = sigmoid(Z1)

 

Z2 = np.dot(A1, W2)

A2 = sigmoid(Z2)

 

acc = accuracy(A2, y_test)

print("Accuracy: {}".format(acc))

 

π
 

#The following is the CNN implementation:

from matplotlib import pyplot as plt

from sklearn.metrics import accuracy_score, log_loss

from tensorflow.keras import optimizers as opt

from tensorflow.keras.datasets import mnist

from tensorflow.keras import Sequential

from tensorflow.keras import layers

from tensorflow.keras import backend as K

from tensorflow import keras

from math import isclose

import tensorflow as tf

import numpy as np

import warnings

(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(f'Input data shape: {x_train.shape}')

print(f'Output data shape: {y_train.shape}')

img_rows, img_cols = 28, 28

x_train = x_train.astype('float32')

x_test = x_test.astype('float32')

x_train /= 255

x_test /= 255

if K.image_data_format() == 'channels_first':

x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)

x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)

input_shape = (1, img_rows, img_cols)

else:

x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)

x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)

input_shape = (img_rows, img_cols, 1)

model = keras.Sequential()

model.add(layers.Conv2D(32, kernel_size=(3, 3), activation='relu',

input_shape=(28, 28, 1)))

model.add(layers.MaxPooling2D(pool_size=(2, 2)))

model.add(layers.Conv2D(64, (3, 3), activation='relu'))

model.add(layers.MaxPooling2D(pool_size=(2, 2)))

model.add(layers.Dropout(0.25))

model.add(layers.Flatten())

model.add(layers.Dense(128, activation='relu'))

model.add(layers.Dense(10, activation='softmax'))

 

model.compile('adam', 'categorical_crossentropy', metrics=['accuracy'])

history = model.fit(x_train, y_train, batch_size=64, epochs=512,

validation_split=0.1, callbacks=(keras.callbacks.EarlyStopping(monitor='val_loss',

mode='min', patience=4, restore_best_weights=True)))

cnn_accuracy = accuracy_score(np.argmax(model.predict(x_test), axis=-1), y_test)

print(f'Accuracy on hold-out set: {cnn_accuracy * 100 : .2f}%')

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('Model Loss')

plt.ylabel('Loss')

plt.xlabel('Epoch')

plt.legend(['train', 'val'], loc='upper left')

plt.show()

plt.plot(history.history['accuracy'])

plt.plot(history.history['val_accuracy'])

plt.title('Model Accuracy')

plt.ylabel('Accuracy')

plt.xlabel('Epoch')

plt.legend(['train', 'val'], loc='upper left')

plt.show()

 

 