import keras
import numpy as np
import random
from keras.models import Sequential
from keras.layers import Dense
import math
def cal_right_rate(y_true,y_pre):
	y_true=[s[0] for s in y_true]	
	y_pre=[ 1  if s[0]>0.5 else 0 for s in y_pre]	
	s=[1 for [s1,s2] in zip(y_true,y_pre)  if s1==s2]
	return 1.0*sum(s)/len(y_true)

def read_data():
	train_x=[]
	num=3
	for i in range(0,num):
		train_x.append([2*random.random()-1,2*random.random()-1])
		#train_y.append([np.random.randint(0,2)])
	return np.array(train_x),num
train_x,num=read_data()
all_status=math.pow(2,num)
drs=[]
wrs=[]
for status in range(0,int(all_status)):
	status=bin(status)[2:]
	status=(num-len(status))*"0"+status
	train_y=np.array([ [int(s)] for s in status])
	model_deep = Sequential()
	model_deep.add(Dense(units=2,input_dim=2,  activation='sigmoid'))
	model_deep.add(Dense(units=2, activation='sigmoid'))
	model_deep.add(Dense(units=1, activation='sigmoid'))
	model_deep.compile(loss='binary_crossentropy', optimizer="adam")
	model_deep.fit(train_x, train_y,epochs=100,verbose=0)

	model_width = Sequential()
	model_width.add(Dense(units=4,input_dim=2,  activation='sigmoid'))
	model_width.add(Dense(units=1, activation='sigmoid'))
	model_width.compile(loss='binary_crossentropy', optimizer="adam")
	model_width.fit(train_x, train_y,epochs=100,verbose=0)


	deep_predict=model_deep.predict(train_x)
	width_predict=model_width.predict(train_x)
	dr=cal_right_rate(train_y,deep_predict)
	wr=cal_right_rate(train_y,width_predict)
	drs.append(dr)
	wrs.append(wr)
	print(status,all_status,"\t", "deep",dr,"\t","width",wr)
print("all_dr",sum(drs))
print("all_wr",sum(wrs))

