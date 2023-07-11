# -*- encoding:utf-8 -*-
import numpy as np
from PIL import Image
def plot(x,width,height,path):
	img=[[0 for _ in range(0,width) ] for _ in range(0,height)]
	for i in range(0,height):
		for j in range(0,width):
			img[i][j]=x[i*height+j]
	new_im = Image.fromarray(train_X[i,:,:])
	new_im.save(path)
	
    

