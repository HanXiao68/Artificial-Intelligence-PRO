import keras
# -*- encoding:utf-8 -*-
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
import random
from keras.models import load_model

def read_data(path):
	train_x=[]
	train_y=[]
	with open(path) as f:
		lines=f.readlines()
	lines=[eval(line.strip()) for line in lines]
	random.shuffle(lines)
	d=int(0.95*len(lines))

	train_x=[s[0] for s in lines[0:d]]
	train_y=[s[1] for s in lines[0:d]]
	test_x=[s[0] for s in lines[d:]]
	test_y=[s[1] for s in lines[d:]]
	return np.array(train_x),np.array(train_y),np.array(test_x),np.array(test_y)
train_x,train_y,test_x,test_y=read_data("data")
model = Sequential()
model.add(Dense(units=5,input_dim=4,  activation='sigmoid'))
model.add(Dense(units=10,activation='sigmoid'))
model.add(Dense(units=3, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer="adam",metrics=['accuracy'])
print "Starting training "
h=model.fit(train_x, train_y, batch_size=8, epochs=500, shuffle=True)
score = model.evaluate(test_x, test_y)
print score
print('Test score:', score[0])
print('Test accuracy:', score[1])
path="model_seq.h5"
model.save(path)
model=None
model=load_model(path)
result=model.predict(test_x)
print result





