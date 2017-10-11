# okay, this is a standard VAE implementation. it's probaly not necessarily (!) going ot help us that much, although we should really work on our split-brain VAE ideas, but the main point is that it will be really cool, and I think vaes are awesome, so let's implement one!

#first we create our encoder network
# it's a really simple one to be homest at the moment

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

from keras.layers import Input, Dense, Lambda, Layer
from keras.models import Model
from keras import backend as K
from keras import metrics
from keras.datasets import mnist

# set our model params
batch_size = 100
original_dim = 784
latent_dim = 2
intermediate_dim = 256
epochs = 50
epsilon_std = 1.0



# we get our input data
(xtrain, ytrain), (xtest, ytest) = mnist.load_data()

xtrain = xtrain.astype('float32') / 255.
xtest = xtest.astype('float32')/255.
xtrain = xtrain.reshape((len(xtrain), np.prod(xtrain.shape[1:])))
xtest = xtest.reshape((len(xtest), np.prod(xtest.shape[1:])))


x = Input(batch_shape=(batch_size, original_dim))
h = Dense(intermediate_dim, activation='relu')(x)
z_mean = Dense(latent_dim)(h)
z_log_sigma = Dense(latent_dim)(h)

#we can use these parameters to sample new similar points from the latent space
def sampling(args):
	z_mean, z_log_sigma = args
	#epsilon is some random variation aroudn it, I think. i.e. it's from the stndard normal with the variable trick, I think, so it's actually differentiable
	epsilon = K.random_normal(shape=(batch_size, latent_dim), mean=0., stddev=epsilon_std)
	return z_mean + K.exp(z_log_sigma) * epsilon

z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_sigma])
# z is our latent space encoded points

# we can then map these back to reconstructed inputs:
decoder_h = Dense(intermediate_dim, activation='relu')
decoder_mean = Dense(original_dim, activation='sigmoid')
h_decoded = decoder_h(z)
x_decoded_mean = decoder_mean(h_decoded)

# we then instantiate models
vae = Model(x, x_decoded_mean)
encoder = Model(x, z_mean)

# thi sis a generator, from latentspace to reconstructed inputs
decoder_input = Input(shape=(latent_dim,))
_h_decoded = decoder_h(decoder_input)
_x_decoded_mean = decoder_mean(_h_decoded)
generator = Model(decoder_input, x_decoded_mean)

# we define a custom loss function for training with the end to end model, which includes the kl divergence

def vae_loss(x, x_decoded_mean):
	reconstruction_loss = objectives.binary.crossentropy(x, x_decoded_mean)
	kl_loss = -0.5 * K.mean(1+z_log_sigma - K.square(z_mean) - K.exp(z_log_sigma), axis=-1)
	return reconstruction_loss + kl_loss

#we compile the model
vae.compile(optimizer='rmsprop', loss=vae_loss)

# we fit the vae
vae.fit(xtrain, xtrain,
		shuffle=True,
		epochs = epochs,
		batch_size = batch_size,
		validation_data=(xtest, xtest))


# one cool thing we can do is to plot the latent space on the 2d plane, as our latent space is 2d (not sure why. I don't really understand the exact maths behind this, I mean it' snever really crystallised, which is unfortunate. I guess I need to sit down and work through it properly?)

def plot_latent_space():
	xtest_encoded = encoder.predict(xtest, batch_size=batch_size)
	plt.figure(figsize=(6,6))
	plt.scatter(xtest_encoded[:, 0], xtest_encoded[:,1], c=ytest)
	plt.colorbar()
	plt.show()

plot_latent_space()

# we can also use it as an encoder to generate new digits, which is pretty cool, use it's capacity as a generative model
def generate_digits(N, digit_size):
	figure = np.zeros((digit_size * N, digit_size *N))
	#we samle n points within [-15. 15] standard deviatoins
	grid_x = np.linspace(-15,15, n)
	grid_y = np.linspace(-15,15, n)

	for i, yi in enumerate(grid_x):
		for j, xi in enumerate(grid_y):
			#sample from the latent space
			z_sample = np.array([xi, yi]) * epsilon_std
			#create the xdecoder matrix - i.e. generate the predictions
			x_decoded = generator.predict(z_sample)
			#get the digit
			digit = x_decoded[0].reshape(digit_size, digitsize)
			#plot:
			figure[i * digit_size: (i+1) * digit_size,
					j * digit_size: (j+1) * digit_size] = digit
	plt.figure(figsize = (10,10))
	plt.imshow(figure)
	plt.show()

generate_digits(15,28)

