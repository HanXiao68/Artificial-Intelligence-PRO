# -*- encoding:utf-8 -*-
import keras
from keras.models import Model
from keras.layers import Input,Dense,Lambda
from keras import backend as K
from keras import regularizers
import numpy as np
from PIL import Image
import random
def plot(x,width,height,path):
	img = x.reshape(width,height)
	img=np.floor(255*img)
	img.dtype="int32"
	new_im = Image.fromarray(img)
	new_im.save(path)
	

def read_data(path):
	with open(path) as f :
		lines=f.readlines()
	lines=[eval(line.strip()) for line in lines]
	random.shuffle(lines)
	X,_=zip(*lines)
	t=int(0.98*len(lines))
	x_train=np.array(X[0:t])
	x_test=np.array(X[t:])
	return x_train,x_test

def get_model():
	input_img = Input(shape=(28*28,))
	encoded = Dense(784*2, activation='relu',activity_regularizer="l1")(input_img)
	decoded = Dense(784, activation='sigmoid')(encoded)
	autoencoder = Model(input=input_img, output=decoded)
	autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
	encoder = Model(input=input_img, output=encoded)
	return autoencoder,encoder

x_train,x_test=read_data("../img_data")
x_train_nosiy = x_train + 0.3 * np.random.normal(loc=0., scale=1., size=x_train.shape)
x_train_nosiy = x_train.reshape(x_train.shape[0], -1)
x_train = x_train.reshape(x_train.shape[0], -1)
x_test = x_test.astype('float32')/255.0
x_test = x_test.reshape(x_test.shape[0], -1)
#print x_train.shape
model,encoder=get_model()
model.fit(x_train_nosiy, x_train, epochs=10, batch_size=128)
encoder.save("../search_img_project_regular/img_model.h5")








