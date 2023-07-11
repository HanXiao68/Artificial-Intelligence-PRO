#!/usr/bin/python
# -*- coding: UTF-8 -*-
import keras
import numpy as np
from keras.models import Model
from keras.layers import Dense,Input,Dropout
from keras import regularizers
def read_data(path):
	train_x=[]
	train_y=[]
	with open(path) as f:
		lines=f.readlines()
	lines=[eval(line.strip()) for line in lines]
	train_x=[s[0] for s in lines]
	train_y=[s[1] for s in lines]
	#return train_x,train_y
	return np.array(train_x),np.array(train_y)
feature_input = Input(shape=(4,))

m1=Dense(units=5,input_dim=4,activation='sigmoid',name="layer1")(feature_input)
m1=Dropout(0.2)(m1)
m2=Dense(units=6, activation='sigmoid',name="layer2",kernel_regularizer=regularizers.l1(0.01))(m1)
m2=Dropout(0.2)(m2)
output=Dense(units=3, activation='softmax')(m2)
model = Model(inputs=feature_input, outputs=output)
model.compile(optimizer='adam', loss='categorical_crossentropy')
train_x,train_y=read_data("data")
model.fit(train_x, train_y, batch_size=8, epochs=100, shuffle=True)

#print model.get_layer("layer2").get_weights()[0]









