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
		lines =f.readlines()#[0:100]
	lines=[eval(line.strip()) for line in lines]
	x,y=zip(*lines)
	x=np.array(x)
	y=np.array(y)
	x=keras.preprocessing.sequence.pad_sequences(x,padding='post',maxlen=max_length)
	return x,y

class Attention(Layer):
    def __init__(self, attention_size=128, **kwargs):
        self.attention_size = attention_size
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        # W: (EMBED_SIZE, ATTENTION_SIZE)
        # b: (ATTENTION_SIZE, 1)
        # u: (ATTENTION_SIZE, 1)
        self.W = self.add_weight(name="W_{:s}".format(self.name),
                                 shape=(input_shape[-1], self.attention_size),
                                 initializer="glorot_normal",
                                 trainable=True)
        #self.b = self.add_weight(name="b_{:s}".format(self.name),shape=(input_shape[1], 1),initializer="zeros",trainable=True)
        self.b = self.add_weight(name="b_{:s}".format(self.name),shape=(1, 1),initializer="zeros",trainable=True)
        self.u = self.add_weight(name="u_{:s}".format(self.name),
                                 shape=(self.attention_size, 1),
                                 initializer="glorot_normal",
                                 trainable=True)
        super(Attention, self).build(input_shape)

    def call(self, x, mask=None):
        # input: (BATCH_SIZE, MAX_TIMESTEPS, EMBED_SIZE)
        # et: (BATCH_SIZE, MAX_TIMESTEPS, ATTENTION_SIZE)
        et = K.tanh(K.dot(x, self.W) + self.b)
        # at: (BATCH_SIZE, MAX_TIMESTEPS)
        at = K.softmax(K.squeeze(K.dot(et, self.u), axis=-1))
        if mask is not None:
            at *= K.cast(mask, K.floatx())
        # ot: (BATCH_SIZE, MAX_TIMESTEPS, EMBED_SIZE)
        atx = K.expand_dims(at, axis=-1)
        ot = atx * x
        # output: (BATCH_SIZE, EMBED_SIZE)
        output = K.sum(ot, axis=1)
        return output

    def compute_mask(self, input, input_mask=None):
        return None

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])
def cal_att_weights(output, att_w):
	eij = np.tanh(np.dot(output, att_w[0]) + att_w[1])
	eij = np.dot(eij, att_w[2])
	eij = eij.reshape((eij.shape[0], eij.shape[1]))
	ai = np.exp(eij)
	weights = ai / np.sum(ai)
	return weights


if "__main__" == __name__:  
	max_length=10
	X,Y=read_data("hanzi_data/train_data_index_hanzi")
	with open("hanzi_data/hanzi_index") as f:
		hanzi_num=len(json.load(f))+1
	with open("hanzi_data/label_index_hanzi") as f:
		num_labels=len(json.load(f))+1

	inputs = Input(shape=(None,))
	text_embedding = Embedding(hanzi_num, 128, input_length=max_length,mask_zero=True)(inputs)
	#text_embedding= Bidirectional(LSTM(64,return_sequences=True))(text_embedding)
	attention = Attention(name="attention")(text_embedding)
	output = Dense(num_labels, activation='softmax')(attention)
	model = Model(inputs, output)
	text_embedding_model = Model(inputs, text_embedding)
	print model.summary()
	model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',metrics=['acc'])
	model.fit(X, Y,batch_size=128,epochs =5,shuffle=True)
	model.save("model")
	text_embedding_model.save("text_embedding_model")




