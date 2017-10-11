# we're going to start simple with single fully connected nn as encoder and decoder

from keras.layers import *
from keras.models import Model
from keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt
from keras.callbacks import TensorBoard


# this is the size of our encoded representations
#encoding_dim = 32  # 32 floats -> compression of factor 24.5, assuming the input is 784 floats

# okay, our convolutional autoencoder for stuff
input_img = Input(shape=(28,28,1))

x = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
encoded = MaxPooling2D((2, 2), padding='same')(x)

# at this point the representation is (4, 4, 8) i.e. 128-dimensional

x = Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
x = UpSampling2D((2, 2))(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(16, (3, 3), activation='relu')(x)
x = UpSampling2D((2, 2))(x)
decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

# now we define our model andcombine
autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

# we use original mnist with shape (input, 3,28,28)  - notsure where the channels come from?
# we also must normalise

(xtrain, _), (xtest, _) = mnist.load_data()

xtrain = xtrain.astype('float32')/255.
xtest = xtest.astype('float32')/255.
xtrain= np.reshape(xtrain, (len(xtrain), 28,28,1))
xtest= np.reshape(xtest, (len(xtest), 28,28,1))

#we thentrain our model
# we can watch this occuring with tensorboard
autoencoder.fit(xtrain, xtrain, epochs=5, batch_size=128, shuffle=True, validation_data=(xtest, xtest), callbacks =[TensorBoard(log_dir='tmp/autoencoder')])



decoded_imgs = autoencoder.predict(xtest)

# now we'll plot with matplotlib
N = 10
plt.figure(figsize=(20,4))
for i in range(N):
	#display original
	ax = plt.subplot(2,N,i+1)
	plt.imshow(xtest[i].reshape(28,28))
	plt.gray()
	ax.get_xaxis().set_visible(False)
	ax.get_yaxis().set_visible(False)

	#display reconstructoin
	ax = plt.subplot(2, N, i+1+N)
	plt.imshow(decoded_imgs[1].reshape(28,28))
	plt.gray()
	ax.get_xaxis().set_visible(False)
	ax.get_yaxis().set_visible(False)
plt.show()

