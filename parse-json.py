import json
import re
# f = "~/PycharmProjects/pytorch"

# with open(path,"r") as f:
#     row_data = json.load(f,encoding='utf-8')

# obj = json.load(open('prediction.json','r',encoding='utf-8'))

# file = open('predict_7.json','r',encoding='utf-8')
# #
# papers = []
# #
# for line in file.readlines():
#     dic = json.loads(line,strict=False)
#     papers.append(dic)
#
# print(len(papers))

with open('predict_7.json') as f:
    data = json.load(f)

# print(json.dumps(data,sort_keys=False,indent=4))
count = 0
for d in data.values():
    count = count+1
    print("第%d轮数据  \n" %count)
    for i in d:
        print(i)
    print('\n')
# file = open('predict_7.json','r',encoding='utf-8')
# for line in file.readlines():
#     data = json.loads(line,strict=False)







# from PyQt5.QtCore import QFile
#
# file = QFile("predict_7.json")
# file.open(QFile.ReadOnly | QFile.Text)
# if file.isOpen() == 1:
#     data = file.readAll()
#     print(data)













