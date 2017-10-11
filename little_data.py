# this is following along with the tutorial from fchollet on keras with little data
# we don't actually get the data, just understand how the model works, I think!

import numpy as np
from keras.prepreocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras import applications
from keras.layers import Activation, Conv2D, MaxPooling2D
from keras import backend as K

#dimensinos of our images
img_width, img_height = 150, 150

train_data_dir = 'data/train'
validation_data_dir = 'data/validation'
nb_train_samples = 2000
nb_validation_samples = 800

epochs = 50
batch_size = 16

if K.image_data_format() == 'channels_first':
	input_shape=(3,img_width, img_height)
else: 
	input_shape = (img_width, img_height, 3)

#start the model
model = Sequential()
model.add(Conv2D(32, (3,3), input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(32, (3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64, (3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

#go to fc layers
model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

#compile model
model.compile(loss='binary_crossentropy',
				optimizer='rmsprop',
				metrics=['accuracy'])

#this is the augmentation configuratoin we will use for training
# it uses keras datagenerators for automatic data augmentatino - which is just amazing. wow is keras awesome!

train_datagen = ImageDataGenerator(
	rescale=1./ 255,
	shear_range=0.2,
	zoom_range=0.2,
	horizontal_flip=True)

#this is the augmentation wewill use for testing - only rescaling
test_datagen = ImagedataGenerator(rescale=1./255)

train_generator=train_datagen.flow_from_directory(train_data_dir, target_size=(img_wifth, img_height), batch_size= batch_size, class_mode = 'binary')

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')

model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=nb_validation_samples // batch_size)

model.save_weights('first_try.h5')
## that is pretty crazy easy to be perfectly honest. keras is an absoluetly wonderful api!


