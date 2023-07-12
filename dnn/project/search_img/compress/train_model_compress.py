# -*- encoding:utf-8 -*-
import keras
from keras.models import Model
from keras.layers import Input,Dense,Lambda
from keras import backend as K
from keras import regularizers
import numpy as np
from PIL import Image
import random
import scipy.misc
def convert(x,width,height):
	img = x.reshape(width,height)
	img=np.floor(255*img)
	img.dtype="int32"
	return img
	

def read_data(path):
	with open(path) as f :
		lines=f.readlines()#[0:100]
	lines=[eval(line.strip()) for line in lines]
	random.shuffle(lines)
	X,_=zip(*lines)
	t=int(0.98*len(lines))
	x_train=np.array(X[0:t])
	x_test=np.array(X[t:])
	return x_train,x_test

def get_model():
	input_img = Input(shape=(28*28,))
	encoded = Dense(100, activation='relu')(input_img)
	decoded = Dense(784, activation='sigmoid')(encoded)
	autoencoder = Model(inputs=input_img, outputs=decoded)
	encoder = Model(inputs=input_img, outputs=encoded)
	autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
	return autoencoder,encoder

x_train,x_test=read_data("../img_data")
x_train = x_train.astype('float32')/255.0
x_train_nosiy = x_train + 0.3 * np.random.normal(loc=0., scale=1., size=x_train.shape)
x_test = x_test.astype('float32')/255.0
x_train = x_train.reshape(x_train.shape[0], -1)
x_train_nosiy = x_train.reshape(x_train.shape[0], -1)
x_test = x_test.reshape(x_test.shape[0], -1)
#print x_train.shape
model1,encoder1=get_model()
model2,encoder2=get_model()
epochs=50
# train without noise
model1.fit(x_train, x_train, epochs=epochs, batch_size=128)
# train with noise
model2.fit(x_train_nosiy, x_train, epochs=epochs, batch_size=128)
x_test1=model1.predict(x_test)
x_test2=model2.predict(x_test)
i=0
for x,x1,x2 in zip(x_test,x_test1,x_test2):
	x=convert(x,28,28)
	x1=convert(x1,28,28)
	x2=convert(x2,28,28)
	scipy.misc.imsave("result/{}-0.png".format(i),x)
	scipy.misc.imsave("result/{}-1.png".format(i),x1)
	scipy.misc.imsave("result/{}-2.png".format(i),x2)
	i+=1
encoder2.save("../search_img_project/img_model.h5")







