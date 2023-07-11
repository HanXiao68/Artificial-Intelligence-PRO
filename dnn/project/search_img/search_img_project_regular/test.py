from keras.models import load_model
import os
import json
import numpy as np
def extract(feature,topK=1):
	s=sorted([[i,f] for i,f in enumerate(feature)])
	s=sorted(s,key=lambda s:s[1],reverse=True)[0:topK]
	return [i for [i,f] in s]
def count(search_result):
	result={}
	for i in search_result:
		result[i]=result.get(i,0)+1
	result=result.items()
	result=sorted(result,key=lambda s:s[1],reverse=True)
	return [s[0] for s in result]
with open("inverse_index") as f:
	inverse_index=json.load(f)

results=["rm -rf search_result/*"]
with open("../search_img_project/feature/feature") as f :
	lines=f.readlines()[0:20]
lines=[eval(line.strip()) for line in lines]
model=load_model("img_model.h5")
l=20
for [i,feature] in lines:
	f=np.array([feature]).reshape(1,-1)
	f2=model.predict(f)[0]
	indexes=extract(feature,topK=1)
	search_results=[]
	for index in indexes:
		search_results.extend(inverse_index.get(str(index),[]))
	search_results=count(search_results)[0:l]
	results.append("mkdir search_result/{}/".format(i))
	results.append("cp ../search_img_project/data/{}.png search_result/{}/origin.png".format(i,i))
	for j in range(0,len(search_results)):
		results.append("cp ../search_img_project/data/{}.png search_result/{}/{}.png".format(search_results[j],i,j))
with open("similar_command.sh","w") as f:
	f.writelines("\n".join(results))
	
