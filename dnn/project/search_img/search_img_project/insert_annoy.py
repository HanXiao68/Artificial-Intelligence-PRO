import numpy as np
from keras.models import load_model
from annoy import AnnoyIndex
with open("feature/feature") as f :
	lines=f.readlines()
lines=[eval(line.strip()) for line in lines]
model=load_model("img_model.h5")
f = 100
t = AnnoyIndex(f,metric="angular")
for [i,feature] in lines:
	print i
	f=np.array([feature]).reshape(1,-1)
	f2=model.predict(f)[0]
    	t.add_item(i, f2)
t.build(10) 
t.save('img.ann')
	
