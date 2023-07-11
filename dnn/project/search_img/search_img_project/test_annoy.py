from annoy import AnnoyIndex
import random
import os
f = 100
u = AnnoyIndex(f)
u.load('img.ann') 
results=[]
l=50
results=["rm -rf search_result/*"]
for i in range(0,200):
	index=u.get_nns_by_item(i, l)
	results.append("mkdir search_result/{}/".format(i))
	results.append("cp data/{}.png search_result/{}/origin.png".format(i,i))
	for j in range(1,l):
		results.append("cp data/{}.png search_result/{}/{}.png".format(index[j],i,j))
with open("similar_command.sh","w") as f:
	f.writelines("\n".join(results))
	
