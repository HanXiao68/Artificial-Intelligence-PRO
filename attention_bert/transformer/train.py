#!/usr/bin/python
# -*- coding: UTF-8 -*
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
from keras.models import Model
from keras.layers import *
from keras import callbacks
import keras.backend as K
import numpy as np
import json
from keras_multi_head import MultiHeadAttention
from keras.utils.np_utils import to_categorical
import math
import keras
def read_data(path):
	with open(path) as f:
		lines =f.readlines()[0:100]
	lines=[eval(line.strip()) for line in lines]
	x,y=zip(*lines)
	x=np.array(x)
	y=np.array(y)
	x=keras.preprocessing.sequence.pad_sequences(x,padding='post',maxlen=max_length)
	return x,y

def transformer(inputs,head_num=8):
	size=inputs.shape[-1]
	feed_forward1=Dense(128,activation='relu')
	feed_forward2=Dense(size,activation=None)
	attention_layer=MultiHeadAttention(head_num=head_num)
	
	attention_outputs=MultiHeadAttention(head_num=head_num)([inputs,inputs,inputs])
	result1=Add()([inputs,attention_outputs])
	result2=TimeDistributed(feed_forward1)(result1)
	result2=TimeDistributed(feed_forward2)(result2)
	result=Add()([result1,result2])
	return result
	
	
	

if "__main__" == __name__:  
	max_length=10
	X,Y=read_data("hanzi_data/train_data_index_hanzi")
	with open("hanzi_data/hanzi_index") as f:
		hanzi_num=len(json.load(f))+1
	with open("hanzi_data/label_index_hanzi") as f:
		num_labels=len(json.load(f))+1

	inputs = Input(shape=(None,))
	text_embedding = Embedding(hanzi_num,64, input_length=max_length,mask_zero=True)(inputs)
	result=transformer(text_embedding)
	result=transformer(result)
	result= Bidirectional(LSTM(64,return_sequences=False))(result)
	output = Dense(num_labels, activation='softmax')(result)
	model = Model(inputs, output)
	model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',metrics=['acc'])
	model.fit(X, Y,batch_size=128,epochs =50,shuffle=True)
	model.save("model")


