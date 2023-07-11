# -*- encoding:utf-8 -*-
import numpy as np
from PIL import Image
import random
import scipy.misc
def read_data(path):
	with open(path) as f :
		lines=f.readlines()#[0:10]
	lines=[eval(line.strip()) for line in lines]
	_,X=zip(*lines)
	return X
data=read_data("feature/feature")
for i in range(0,len(data)):
	scipy.misc.imsave("data/{}.png".format(i),np.array(data[i]))
	









