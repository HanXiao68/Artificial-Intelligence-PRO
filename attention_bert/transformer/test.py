# -*- coding:utf-8 -*-
import sys
import json
import numpy as np
import keras 
from keras.models import load_model  
from train import *
import numpy as np
def read_data(path):
	with open(path) as f:
		lines =f.readlines()[0:20]
	lines=[eval(line.strip()) for line in lines]
	x,y=zip(*lines)
	x=np.array(x)
	x=keras.preprocessing.sequence.pad_sequences(x,padding='post',maxlen=max_length)
	return x,y
with open("hanzi_data/hanzi_index") as f:
	index_hanzi=dict([ [index,hanzi] for [hanzi,index] in json.load(f).items()])
with open("hanzi_data/label_index_hanzi") as f:
	index_label=dict([ [index,label] for [label,index] in json.load(f).items()])
max_length=10
X,Y=read_data("hanzi_data/train_data_index_hanzi")
model=load_model("model",custom_objects={"Attention":Attention})
text_embedding_model=load_model("text_embedding_model")
att_w=model.get_layer("attention").get_weights()
#texts=text_embedding_model.predict(X)
class_results=model.predict(X)
class_results=[np.argmax(s) for s in class_results]
for x,y,class_result in zip(X.tolist(),Y,class_results):
	x=[s for s in x if s!=0]
	text=text_embedding_model.predict(x)#[0]
	text=[s[0] for s in text]
	attention_result=cal_att_weights(text, att_w)
	hanzi_list=[index_hanzi[s] for s in x]
	hanzi_score=[ "{}:{}".format(s[0],round(s[1][0],3)) for s in zip(hanzi_list,attention_result) if s[0]!="__token__"]
	print(",".join(hanzi_score)+"\t"+index_label[class_result]+"\t"+index_label[y])	
