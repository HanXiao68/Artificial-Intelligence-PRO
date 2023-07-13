import numpy as np
import matplotlib.pyplot as plt

delta_t = 0.1 #采样时间间隔
end_t = 8  # 时间长度

time_t = end_t *10; # 采样次数
t = np.arange(0, end_t, delta_t)  # 设置时间数组

u = 2   # 外界对系统的作用 加速度
x = 1/2 * u*t**2  # 实际真实值

v_var = 1  # 测量噪声的方差

# 创建高斯噪声，精确到小数点后两位
v_noise = np.round(np.random.normal(0, v_var, time_t), 2)

#定义预测优化值的初始状态
X = np.mat([[0], [0]])

#定义测量噪声
v = np.mat(v_noise)

#定义测量值（假设测量值 = 实际状态值 + 噪声）
z = x + v

#定义状态转移矩阵
A = np.mat([[1, delta_t], [0, 1]])

#定义输入控制矩阵
B = [(1/2*(delta_t**2))],[delta_t]

#定义初始状态协方差矩阵
P = np.mat([[1, 0], [0, 1]])

#定义状态转移（预测噪声）协方差矩阵
Q = np.mat([[0.001, 0],[0,0.001]])

#定义观测矩阵
H = np.mat([1, 0])

#观测噪声协方差
R = np.mat([1])

#初始化记录系统预测优化值的列表
X_mat = np.zeros(time_t)

for i in range(time_t):
    #预测
    #估算状态变量
    X_predict = A * X + np.dot(B, u)
    #估算状态误差协方差
    P_predict = A * P*A.T +Q

    #矫正
    #更新卡尔曼增益
    K = P_predict * H.T /(H*P_predict*H.T + R)
    #更新预测优化值
    X = X_predict + K *(z[0, i]-H * X_predict)
    #更新状态误差协方差
    P = (np.eye(2) -K*H)* P_predict
    #记录系统的预测优化值
    X_mat[i] = X[0,0]

# plt.rcParams['font.sans-serif'] = ['SimHei'] #设置正常显示中文

plt.plot(x,"b",label='real_state')
plt.plot(X_mat,"g",label='predict_optimited value')
plt.plot(z.T,"r--",label='measure value')
plt.xlabel("time") #x轴名字
plt.ylabel("postion_move")                               # 设置Y轴的名字
plt.title("kalman Presentation")                     # 设置标题
plt.legend()                                    # 设置图例
plt.show()


