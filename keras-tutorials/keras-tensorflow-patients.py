# first we look at our data
import pandas as pds

dfX = pds.read_csv('tutorial_patient_data.csv', usecols=[0,1,2,8,5,9,10,11,12])
dfY = pds.read_csv('tutorial_patient_data.csv', usecols=[12])

print(dfX.head())
print dfY.head()

# pandas data to numeric value functiosn

def weekdayToInt(weekday):
	return {
		'Monday': 0,
		'Tuesday': 1,
		'Wednesday': 2,
		'Thursday': 3,
		'Friday': 4,
		'Saturday': 5,
		'Sunday':6
	}[weekday]

def genderToInt(gender):
	if gender=='M':
		return 0
	else:
		return 1

def statusToInt(status):
	if status=='Show-up':
		return 1
	else:
		return 0

#dfX.DayOfTheWeek = dfX.DayOfTheWeek.apply(weekdayToInt)
dfX.Gender = dfX.Gender.apply(genderToInt)
#dfY.No_show = dfY.No_show.apply(statusToInt)

print dfX.head
print "  "
print dfY.head

# okay, now for the nn
#we set the seed for reproducibility
seed = 7
np.random.seed(seed)

import numpy as np
from keras.models import Sequential
from keras.layers import Dense

#now for the actual model
model = Sequential()
model.add(Dense(12, input_shape=(11,), init='uniform', activation='sigmoid'))
model.add(Dense(12, init='uniform', activation='sigmoid'))
model.add(Dense(12, init='uniform', activation='sigmoid'))
model.add(Dense(12, init='uniform', activation='sigmoid'))
model.add(Dense(1, init='uniform', activation='sigmoid'))
model.summary()

import keras
# we need this for our tensorboard logging!
tbCallBack = keras.callbacks.Tensorboard(log_dir='tmp/keras_logs', write_graph=True)

#now we actually make the model
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
model.fit(dfX.values, dfY.values, epochs=9, batch_size=50, verbose=1, validation_split=0.3, callbacks=[tbCallBack])

model.predict()

