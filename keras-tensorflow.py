# so this does mnist, but integrating keras and tensorflow which, for phd work, is going to be absolutely essential for maximum customisability and ease of the networks

import tensorflow as tf
import keras
#set our keras backend to be tf
from keras import backend as K
from keras.layers import Dense
from keras.objectives import *
from tensorflow.examples.tutorials.mnist import input_data
from keras.metrics import categorical_accuracy as accuracy

#start tf session
sess = tf.Session()
#set keras backend to session
K.set_session(sess)

#get our data
mnist_data = input_data.read_data_sets('MNIST_data', one_hot = True)



# now we begin our mnist model
img = tf.placeholder(tf.float32, shape=(None, 784))

#now we input our keras layers
x = Dense(128, activation='relu')(img)
x = Dense(128, activation='relu')(x) # not sure how the variable isn't overwritten?
preds = Dense(10, activation='softmax')(x) # output layer with 10 units and a softmax activation

# placeholder for labels and loss functoin
labels = tf.placeholder(tf.float32, shape=(None, 10))
loss = tf.reduce_mean(categorical_crossentropy(labels, preds))

# train with tensorflow optimizer

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

#init all vars
init_op = tf.global_variables_initializer()
sess.run(init_op)

# run training loop
with sess.as_default():
	for i in range(100):
		batch = mnist_data.train.next_batch(50)
		train_step.run(feed_dict={img: batch[0], labels: batch[1]})

# we now evaluate the model using keras functions
acc_value = accuracy(labels, preds)
# and back to tensorflow!
with sess.as_default():
	print acc_value.eval(feed_dict={img: mnist_data.test.images, labels: mnist_data.test.labels})

