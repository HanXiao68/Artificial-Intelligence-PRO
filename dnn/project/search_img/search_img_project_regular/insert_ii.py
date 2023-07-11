import numpy as np
from keras.models import load_model
import json
def extract(feature,topK=1):
	s=sorted([[i,f] for i,f in enumerate(feature)])
	s=sorted(s,key=lambda s:s[1],reverse=True)[0:topK]
	return [i for [i,f] in s]

with open("../search_img_project/feature/feature") as f :
	lines=f.readlines()#[0:100]
lines=[eval(line.strip()) for line in lines]
model=load_model("img_model.h5")
result={}
for [i,feature] in lines:
	f=np.array([feature]).reshape(1,-1)
	f2=model.predict(f)[0]
	indexes=extract(feature,topK=3)
	for index in indexes:
		if index not in result:
			result[index]=[]
		result[index].append(i)
with open("inverse_index","w") as f :
	json.dump(result,f)
	
