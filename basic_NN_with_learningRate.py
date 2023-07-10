import numpy as np

#sigmoid激活函数
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

#前向传播
def forward(X, W1, W2, b1, b2):
    Z = sigmoid(X.dot(W1) + b1) #隐藏层
    Y = sigmoid(Z.dot(W2) + b2) #输出层
    return Y

#计算损失函数
def calculate_loss(Y, T  )