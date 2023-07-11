from sklearn.datasets import load_iris
import random
def set_label(s):
	result=[0,0,0]
	result[s]=1
	return result

data = load_iris()
X=data.data
X=[s.tolist() for s in X]
Y=data.target
Y=[ set_label(s) for s in Y]
results=zip(X,Y)
results=[str(s) for s in results]
with open("data","w") as f:
	f.writelines("\n".join(results))
