import numpy as np

data = np.array([1,2,3,4,5,6,7,8,9],dtype='float32')

import pandas as pd
data2 = pd.DataFrame([1.2,1.3,1.4,1.5,1.6],dtype="float64")

print(data2.info(memory_usage=True))

print("==================================")
data2 = pd.DataFrame([1.2,1.3,1.4,1.5,1.6],dtype="float32")

print(data2.info(memory_usage=True))