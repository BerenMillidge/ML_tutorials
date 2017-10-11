from keras.layers import *
from keras.models import Model
from keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt
from keras.callbacks import TensorBoard


# this is the size of our encoded representations
#encoding_dim = 32  # 32 floats -> compression of factor 24.5, assuming the input is 784 floats

# okay, our convolutional autoencoder for stuff denoising, so first we need to iput the model, then set up the artificial noising functinos for this test!

#first we get data and artificially noise it
(xtrain, _), (xtest, _) = mnist.load_data()
xtrain = xtrain.astype('float32')/255.
xtest = xtest.astype('float32')/255.
xtrain= np.reshape(xtrain, (len(xtrain), 28,28,1))
xtest= np.reshape(xtest, (len(xtest), 28,28,1))


#now for the noising
noise_factor = 0.5
xtrain_noisy = xtrain + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=xtrain.shape)
xtest_noisy = xtest + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=xtest.shape)

xtrain_noisy = np.clip(xtrain_noisy, 0., 1.)
xtest_noisy = np.clip(xtest_noisy, 0., 1.)

def plot_noisy_digits(N):
	plt.figure(figsize=(20,2))
	for i in xrange(N):
		ax = plt.subplot(1,n,i)
		plt.imshow(xtest_noisy[i].reshape(28,28))
		plt.gray()
		ax.get_xaxis().set_visible(False)
		ax.get_yaxis().set_visible(False)
	plt.show()

plot_noisy_digits(10)

# now for our model


input_img = Input(shape=(28,28,1))

x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
encoded = MaxPooling2D((2, 2), padding='same')(x)

# at this point the representation is (4, 4, 8) i.e. 128-dimensional

x = Conv2D(32, (3, 3), activation='relu', padding='same')(encoded)
x = UpSampling2D((2, 2))(x)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

# now we define our model andcombine
autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

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

