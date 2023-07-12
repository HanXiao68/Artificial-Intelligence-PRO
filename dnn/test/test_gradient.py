#!/usr/bin/python
# -*- coding: UTF-8 -*-
import keras
import numpy as np
from keras.models import Model
from keras.layers import Dense,Input
from keras import regularizers
def cal_distance(w1,w2):
	delta=w1-w2
	delta=delta.tolist()
	result=sum([s*s for ss in delta for s in ss])
	return result
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
act='relu'#relu tanh sigmoid
hidden_num=5
l=feature_input
layer_num=10
'''重点'''
for i in range(0,layer_num):
	l=Dense(units=hidden_num,activation=act,name="layer{}".format(i))(l)
output=Dense(units=3, activation='softmax')(l)
model = Model(inputs=feature_input, outputs=output)
model.compile( loss='categorical_crossentropy',optimizer='sgd')
train_x,train_y=read_data("data")
#训练之前各个层的权重
weights1=[ model.get_layer("layer{}".format(i)).get_weights()[0] for i in range(0,layer_num)]
model.fit(train_x, train_y, batch_size=len(train_x), epochs=1)
#训练之后各个层的权重
weights2=[ model.get_layer("layer{}".format(i)).get_weights()[0] for i in range(0,layer_num)]
delta=[ cal_distance(w1,w2) for [w1,w2] in zip(weights1,weights2)]
#有多少层，delta就是多少维度的数组，每一维计算了该层w的变化
print delta









