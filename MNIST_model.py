from keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
import time
start_time = time.time()


(x_train, y_train), (x_test, y_test) = mnist.load_data()

#Data visualization
image_index = 0
#print(y_train[image_index])
#lt.imshow(x_train[image_index], cmap='Greys')
#plt.show()

#Normalizing data
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

#Reshaping data
x_train = x_train.reshape((-1, 784))
x_test = x_test.reshape((-1, 784))
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)




# Defining the model
model = Sequential()
model.add(Dropout(0.8))
model.add(Dense(1024, activation = "relu", input_shape = (784,)))
model.add(Dropout(0.5))

model.add(Dense(1024, activation = "relu"))
model.add(Dropout(0.5))

model.add(Dense(10, activation = "softmax"))


model.compile(optimizer = 'sgd' , loss = "categorical_crossentropy", metrics=["accuracy"])

history = model.fit(x_train, y_train, epochs=20, batch_size=32, validation_split=0.1666666667, verbose=2)

print("--- %s seconds ---" % (time.time() - start_time))

