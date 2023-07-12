import keras
import numpy as np
from keras.models import Model
from keras.layers import Dense,Input
from keras import regularizers
def read_data(path):
	train_x=[]
	train_y=[]
	with open(path) as f:
		lines=f.readlines()#[0:100]
	lines=[eval(line.strip()) for line in lines]
	train_x=[s[0] for s in lines]
	train_y=[s[1] for s in lines]
	#return train_x,train_y
	return np.array(train_x),np.array(train_y)
feature_input = Input(shape=(4,))
l=feature_input
middle_model=[]
num=3
act='relu'
for i in range(0,num):
	l=Dense(units=4,input_dim=4,activation=act,kernel_initializer=keras.initializers.Constant(value=1),name="layer{}".format(i))(l)
	#l=Dense(units=4,input_dim=4,activation=act,name="layer{}".format(i))(l)
	middle_model.append(Model(inputs=feature_input,outputs=l))

output=Dense(units=3, activation='softmax')(l)
model = Model(inputs=feature_input, outputs=output)
model.compile(optimizer='adam', loss='categorical_crossentropy')
train_x,train_y=read_data("data")
model.fit(train_x, train_y, batch_size=len(train_x), epochs=1, shuffle=True)
for j in range(0,10):
	for i in range(0,num):
		print middle_model[i].predict(train_x[j:j+1])
	print







