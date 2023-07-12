import random
import numpy as np
num=100
def sigmoid_fun(x):
    return 1/(1+np.exp(-x))

def get_data(m):
	return m+2*random.random()-1
def write(path,data):
	data2=[ "{},{}".format(a1,a2) for [a1,a2] in data.tolist()]
	with open(path,"w") as f :
		f.writelines("\n".join(data2))

X1=np.array([[get_data(0),get_data(2)]  for i in range(0,num)])
W= np.random.rand(2,2)
W=2*W-1
print W
X2=np.array([np.dot(W,np.array(x)).tolist() for x in X1.tolist()])
X3=sigmoid_fun(X2)
write("data1.csv",X1)
write("data2.csv",X2)
write("data3.csv",X3)
