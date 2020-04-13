from keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.utils import to_categorical
from tensorflow.keras.optimizers import SGD, RMSprop
from tensorflow.keras.constraints import max_norm



(x_train, y_train), (x_test, y_test) = mnist.load_data()

#Data visualization
image_index = 0
#print(y_train[image_index])
#lt.imshow(x_train[image_index], cmap='Greys')
#plt.show()

#Normalizing data
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
y_train = y_train.astype('float32')
y_test = y_test.astype('float32')
x_train /= 255
x_test /= 255

#Reshaping data
x_train = x_train.reshape((-1, 784))
x_test = x_test.reshape((-1, 784))
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)




# Defining the model
model = Sequential()
model.add(Dropout(0.2))
model.add(Dense(1024, activation = "relu", input_shape = (784,), kernel_constraint=max_norm(2), bias_constraint=max_norm(2)))

model.add(Dropout(0.5))

model.add(Dense(1024, activation = "relu", kernel_constraint=max_norm(2), bias_constraint=max_norm(2)))
model.add(Dropout(0.5))

model.add(Dense(10, activation = "softmax", kernel_constraint=max_norm(2), bias_constraint=max_norm(2)))


learning_rate = 0.001
decay_rate = learning_rate / 1000
momentum = 0.9
sgd = SGD(momentum=)
RMS = RMSprop(learning_rate = learning_rate, decay=decay_rate)
model.compile(optimizer = RMS , loss = "categorical_crossentropy", metrics=["accuracy"])
history = model.fit(x_train, y_train, epochs=1000, batch_size=15000, validation_split=0, verbose=2)
results = model.evaluate(x_test, y_test, batch_size=256, verbose=1)

