# import lib
import numpy as np
import matplotlib.pyplot as plt

# sigmoid
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# 代价函数
def computeCost(X, Y, theta):
    z = X * theta.T
    m = Y.size
    para1 = np.multiply(-Y, np.log(sigmoid(z)))
    para2 = np.multiply((1 - Y), np.log(1 - sigmoid(z)))
    J = 1 / m * np.sum(para1 - para2)
    return J

# 梯度下降
def gradientDecent(X, Y, theta, alpha, iters):
    cost = np.zeros(iters)
    for i in range(iters):
        theta = theta - (alpha / Y.size) * ((sigmoid(X * theta.T) - Y).T * X)
        cost[i] = computeCost(X, Y, theta)
    return theta, cost

# 对数据预处理
def init_data(data_path):
    train_data = np.genfromtxt(data_path)
    add_b = np.ones(train_data.shape[0])
    train_data = np.insert(train_data, 0, values=add_b, axis=1)
    X = train_data[:, [0, 1, 2]]
    Y = train_data[:, [3]]
    X = np.mat(X)
    Y = np.mat(Y)
    return X, Y, train_data

# 预测函数
def predict(theta_min, predictX):
    probability = sigmoid(predictX * theta_min.T)
    return [1 if x >= 0.5 else 0 for x in probability]

# 绘制原始数据
def draw_data(train_data):
    Y0 = train_data[train_data[:, 3] == 0]
    Y1 = train_data[train_data[:, 3] == 1]
    x_Y0, y_Y0 = Y0[:, 1], Y0[:, 2]
    x_Y1, y_Y1 = Y1[:, 1], Y1[:, 2]

    plt.scatter(x_Y0, y_Y0, c='b', label='Not Admitted')
    plt.scatter(x_Y1, y_Y1, c='r', marker='x', label='Admitted')
    plt.xlabel('feature1')
    plt.ylabel('feature2')
    return plt

# 决策边界theta*x=0
def boundary(theta_min, train_data):
    x1 = np.arange(-4, 4, 0.01)
    x2 = (theta_min[0, 0] + theta_min[0, 1] * x1) / (-theta_min[0, 2])
    plt = draw_data(train_data)
    plt.title('boundary')
    plt.plot(x1, x2)
    plt.show()


if __name__ == '__main__':
    X, Y, train_data = init_data('data.txt')
    theta = np.mat(np.zeros(X.shape[1]))
    iters = 5000
    alpha = 0.1
    theta_min, cost = gradientDecent(X, Y, theta, alpha, iters)
    print('theta_min:' + str(theta_min))

    predictX = X
    res = predict(theta_min, predictX)

    correct = [1 if ((a == 1 and b == 1) or (a == 0 and b == 0)) else 0 for (a, b) in zip(res, Y)]
    accuracy = (sum(map(int, correct)) % len(correct))
    print('accuracy = {0}%'.format(accuracy))

    boundary(theta_min, train_data)

