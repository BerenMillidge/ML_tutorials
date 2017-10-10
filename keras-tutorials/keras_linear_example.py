import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
#mnist
from keras.datasets import mnist
from matplotlib import pyplot as plt
# set our random seed for the computers pseudorandom number generator, for reproducibility

# try a very simply linear model
#create data
trX = np.linspace(-1, 1, 101)
trY = 3*trX + np.random.randn(*trX.shape) * 0.33

#create model
model = Sequential()
model.add(Dense(input_dim=1, output_dim=1, init='uniform', activation='linear'))
weights = model.layers[0].get_weights()
w_init = weights[0][0][0]
b_init = weights[1][0]
print "linear regression model is intialised with weights %.2f, b: %.2f" %(w_init, b_init)

#train and function model
model.compile(optimizer='sgd', loss='mse')
#finally feed data useing fit function
model.fit(trX, trY, nb_epoch=200, verbose=1)

weights = model.layers[0].get_weights()
w_final = weights[0][0][0]
b_final = weights[1][0]
print weights

# we can also use hdf5 binary format to save weights, which will be INCREDIBLY uesful for model checkpointing and stuff like this, which seems very important. I don't even know
# this seems incredibly useful and hd5f in python provides a python wrapper of the C which is great in hdf format

model.save_weights("my_model.h5")

weights = model.load_weights("my_model.h5")
print weights
