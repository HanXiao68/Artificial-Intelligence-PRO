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
act='relu'
hidden_models=[]
l=feature_input
#init_w
#正态分布初始化法
#my_init=keras.initializers.RandomNormal(mean=2.0, stddev=0.05)
#Glorot均匀分布初始化方法，又成Xavier均匀初始化，参数从[-limit, limit]的均匀分布产生，其中limit为sqrt(6 / (输入神经元数量 + 输出神经元))。fan_in为权值张量的输入单元数，fan_out是权重张量的输出单元数。
my_init="glorot_uniform"
layer_num=10
hidden_num=2#调整隐藏层的神经元数量
for i in range(0,layer_num):
	#l=Dense(units=hidden_num,activation=act,name="layer{}".format(i))(l)
	#调整权重的初始化
	l=Dense(units=hidden_num,activation=act,name="layer{}".format(i))(l)
	hidden_models.append(Model(inputs=feature_input, outputs=l))
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
#print "各个层的变化\n"
#print delta
print "各个层的输出\n"
print "第一个样本"
for hidden_model in hidden_models:
	print hidden_model.predict(train_x[0:1])
print "第二个样本"
for hidden_model in hidden_models:
	print hidden_model.predict(train_x[1:2])









