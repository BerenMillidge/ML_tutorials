# we're going to start simple with single fully connected nn as encoder and decoder

from keras.layers import Input, Dense
from keras.models import Model
from keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt


# this is the size of our encoded representations
#encoding_dim = 32  # 32 floats -> compression of factor 24.5, assuming the input is 784 floats

# this is our input placeholder
input_img = Input(shape=(784,))

#deep encoder
encoded=Dense(128, activation='relu')(input_img)
encoded=Dense(64, activation='relu')(encoded)
encoded=Dense(32, activation='relu')(encoded)

#deep decoder
decoded = Dense(64, activation='relu')(encoded)
decoded = Dense(124, activation='relu')(decoded)
decoded= Dense(784, activation='relu')(decoded)

#define our models
#encoder = Model(input_img, encoded)
#decoder = Model(encoded, decoded)

autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

(x_train, _), (x_test, _) = mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
print x_train.shape
print x_test.shape

autoencoder.fit(x_train, x_train,
                epochs=5,
                batch_size=256,
                shuffle=True,
                validation_data=(x_test, x_test))

# encode and decode some digits
# note that we take them from the *test* set
#encoded_imgs = encoder.predict(x_test)
decoded_imgs = autoencoder.predict(x_test)

# now we'll plot with matplotlib
N = 10
plt.figure(figsize=(20,4))
for i in range(N):
	#display original
	ax = plt.subplot(2,N,i+1)
	plt.imshow(x_test[i].reshape(28,28))
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


