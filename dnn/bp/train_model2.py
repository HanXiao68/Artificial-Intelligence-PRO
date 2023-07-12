import keras
import numpy as np
from keras.models import Model
from keras.layers import Dense,Input
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
#first model
m11=Dense(units=5,input_dim=4,activation='relu')(feature_input)
m12=Dense(units=6, activation='relu')(m11)
#second model
m21=Dense(units=5,input_dim=4,activation='relu')(feature_input)
m22=Dense(units=6, activation='relu')(m21)

#add
m= keras.layers.Concatenate(axis= 1)([m21,m22])


output=Dense(units=3, activation='softmax')(m)
model = Model(inputs=feature_input, outputs=output)
model.compile(optimizer='adam', loss='categorical_crossentropy',metrics=["accuracy"])
train_x,train_y=read_data("data")
model.fit(train_x, train_y, batch_size=8, epochs=300, shuffle=True)









