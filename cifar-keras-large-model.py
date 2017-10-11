# larger model for bettter cifar performance, this should be fun

from keras.datasets import cifar10
from matplotlib import pyplot as plt
from scipy.misc import toimage
import numpy as np
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.constraints import maxnorm
from keras.optimizers import SGD
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras import backend as K

#load data
(xtrain, ytrain), (xtest, ytest) = cifar10.load_data()

#functino to plot the images
def plot_images(N):
	for i in range(0,N):
		plt.subplot(330+1+i)
		plt.imshow(toimage(xtrain[i]))
	plt.show()

#plot_images(10)

# okay, let's define the classes and function we need
K.set_image_dim_ordering('th')
seed =7
np.random.seed(seed)

#loadcifar dataset
(xtrain, ytrain), (xtest, ytest) = cifar10.load_data()

# we do casts and normalisations
xtrain = xtrain.astype('float32')
xtest = xtest.astype('float32')
xtrain = xtrain/255.0
xtest = xtest/255.0 # 0 to make sure it's a floating point division

#one hot encode outputs
ytrain = np_utils.to_categorical(ytrain)
ytest = np_utils.to_categorical(ytest)
num_classes = ytest.shape[1]
#params
drop = 0.2
pool = (2,2)

#compile params
epochs = 25
lrate =0.01
decay = lrate/epochs
batch_size = 64


model = Sequential()
model.add(Conv2D(32, (3,3), input_shape=(3,32,32), activation='relu', padding='same'))
model.add(Dropout(drop))
model.add(Conv2D(32, (3,3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool))
model.add(Conv2D(64, (3,3), activation='relu', padding='same'))
model.add(Dropout(drop))
model.add(Conv2D(64, (3,3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool))
model.add(Conv2D(32, (3,3), activation='relu', padding='same'))
model.add(Dropout(drop))
model.add(Conv2D(32, (3,3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool)
#we finish the conv layer and move to the flat layer
model.add(Flatt())
model.add(Dropout(drop))
model.add(Dense(1024, activation='relu', kernel_constraint=maxnorm(3)))
model.add(Dropout(drop))
model.add(Dense(512, activation='relu', kernel_constraint=maxnorm(3)))
model.add(Dropout(drop))
model.add(Dense(num_classes, activation='softmax'))

#compile model
sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
print(model.summary()

model.fit(xtrani, ytrain, validation_data=(xtest, ytest), epochs=epochs, batch_size=batch_size)
scores = model.evaluate(xtest, ytest, verbose=0)
print("Accuracy: %.2f$$" % (scores[1]*100))

