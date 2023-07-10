import numpy as np

#定义一个神经网络类
class NN_HX():
    def __init__(self):
        # 设置神经网络的输入，隐藏和输出节点数
        self.input_nodes = 3
        self.hidden_nodes = 4
        self.output_nodes = 1

        # 随机初始化权重矩阵
        self.weights_input_hidden = np.random.randn(self.hidden_nodes,self.input_nodes)
        self.weights_hidden_output = np.random.randn(self.output_nodes,self.hidden_nodes)

    # 定义激活函数
    def sigmoid(self, x):
        return 1 / (1 +np.exp(-x))

    # 定义神经网络的前向传播
    def forward(self, inputs):
        #计算隐藏层的输入
        hidden_inputs = np.dot(self.weights_input_hidden, inputs)
        #计算隐藏层输出
        hidden_outputs = self.sigmoid(hidden_inputs)

        #计算输入层输入
        final_inputs = np.dot(self.weights_hidden_output, hidden_outputs)
        #计算输出层输出
        final_outputs = self.sigmoid(final_inputs)
        return final_outputs

#创建一个神经网络实例
neural_network = NN_HX()

#输入数据
inputs = np.array([0.5, 0.3, 0.2])

#输出神经网络的预测结果
print(neural_network.forward(inputs))