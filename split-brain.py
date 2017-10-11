# like the point of this is to hope to do a big rote conversion of the caffe model from the zhang paper into a keras model - and I have a reasonable understanding now of how to do that, which is great, so I'd better get on that!

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

model = Sequential()

# there are some image processing layers here and stuff which I don't think we'll put in the actual keras model but instead elsewhere in the file

model.add(Conv2D(96, (11,11), stride=(4,4), activation='relu', name='conv1'))
model.add(MaxPooling2D(pool_size=(3,3), stride=(2,2))
model.add(Conv2D(256, (5,5), activation='relu', name='conv2'))
model.add(MaxPooling2D(pool_size=(3,3), stride=(2,2))
model.add(Conv2D(384, (3,3), activation='relu', name='conv3'))
# btw, in the caffe model there's some stuff going on with the padding which I dno't udnerstand and so haven't converted, but which is probably vital
model.add(Conv2D(384, (3,3), activation='relu', name='conv4'))
#also group in the convolutions. I don't know what that means
model.add(Conv2D(256, (3,3), activation='relu', name='conv5'))
model.add(MaxPooling2D(pool_size=(3,3), stride=(2,2))
model.add(Conv2D(4096, (6,6), stride=(1,1), activation='relu', name='fc6')) # some stuff going on here with dilation which I also don't understand. btw this is meant to be a 1-1 convolution also, I think, which is weird? or something?
model.add(Conv2D(4096, (1,1), stride=(1,1), activation='relu', name='fc7'))


# so that's it. I can't really pretend I understand this, and it will take FOREVER! to run, so it's probably massively infeasible. I think we're realstically going to have to develop our solution from scratch...
