import matplotlib.pyplot as plt
from keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from tensorflow.keras.utils import to_categorical
import keras
import time
from keras.datasets import cifar10
from keras import optimizers

import tensorflow as tf

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

#print(x_train.shape)

#Normalizing data
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

#Reshaping data
# image_index = 3
# print(y_train[image_index])
# plt.imshow(x_train[image_index])
# plt.show()

model = Sequential()
model.add(Dropout(0.1))
model.add(Conv2D(filters = 96, kernel_size = (5,5),padding = 'Same',
                 activation ='relu', input_shape = (32,32,3)))
model.add(Dropout(0.25))
model.add(MaxPool2D(pool_size=(3,3), strides=(2,2)))

model.add(Conv2D(filters = 128, kernel_size = (5,5),padding = 'Same',
                 activation ='relu'))
model.add(Dropout(0.25))
model.add(MaxPool2D(pool_size=(3,3), strides=(2,2)))

model.add(Conv2D(filters = 256, kernel_size = (5,5),padding = 'Same',
                 activation ='relu'))
model.add(Dropout(0.5))
model.add(MaxPool2D(pool_size=(3,3), strides=(2,2)))

model.add(Flatten())
model.add(Dense(2048, activation = "relu"))
model.add(Dropout(0.5))
model.add(Dense(2048, activation = "relu"))
model.add(Dropout(0.5))
model.add(Dense(10, activation = "softmax"))
opt = optimizers.RMSprop(learning_rate=0.0001, decay=1e-6)

model.compile(optimizer='RMSprop', loss = "categorical_crossentropy", metrics=["accuracy"])

history = model.fit(x_train, y_train, epochs=2, batch_size=1000, validation_split=0.1, verbose=2)