from numpy import *
import math
x = []
def loadDataSet():
    dataMat = []
    labelMat = []
    fr = open('test1.txt')
    for line in fr.readlines():
        lineArr = line.strip().split(',')
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
        labelMat.append(int(lineArr[2]))
    return dataMat, labelMat


def sigmoid(inX):
    return longfloat(1.0 / (1 + exp(-inX)))  # sigmoid函数公式


def gradAscent(dataMatIn, classLabels):
    # dataMatIn 一个2维的数组；classLabels 类别标签
    dataMatrix = mat(dataMatIn)  # 转换为矩阵
    labelMat = mat(classLabels).transpose()  # 得到矩阵的转置矩阵
    m, n = shape(dataMatrix)  # 读取矩阵的长度,二维矩阵，返回两个值
    alpha = 0.001  # 向目标移动的步长
    maxCycles = 500  # 迭代次数
    weights = ones((n, 1))  # ones()函数用以创建指定形状和类型的数组，默认情况下返回的类型是float64。但是，如果使用ones()函数时指定了数据类型，那么返回的就是该类型
    for k in range(maxCycles):
        h = sigmoid(dataMatrix * weights)  # matrix mult
        error = (labelMat - h)  # vector subtraction
        weights = weights + alpha * dataMatrix.transpose() * error  # matrix mult
    return weights


def plotBestFit(weights):
    import matplotlib as mpl
    mpl.use('Agg')  # 为了防止出现:RuntimeError: could not open display报错
    import matplotlib.pyplot as plt
    dataMat, labelMat = loadDataSet()
    dataArr = array(dataMat)
    n = shape(dataArr)[0]
    xcord1 = []
    ycord1 = []
    xcord2 = []
    ycord2 = []
    for i in range(n):
        if int(labelMat[i]) == 1:
            xcord1.append(dataArr[i, 1])
            ycord1.append(dataArr[i, 2])
        else:
            xcord2.append(dataArr[i, 1])
            ycord2.append(dataArr[i, 2])
    fig = plt.figure()  # figure: 控制dpi、边界颜色、图形大小、和子区( subplot)设置
    ax = fig.add_subplot(111)  # 参数111的意思是：将画布分割成1行1列，图像画在从左到右从上到下的第1块，
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')
    #x = arange(-3.0, 3.0, 0.1)
    #y = (-weights[0] - weights[1] * x) / weights[1]
    #ax.plot(x, y)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.savefig('plotBestFit.png')  # 因为我是腾讯云服务器,没有图形界面，所以我保存为图片。


# 随机梯度上升算法
def stocGradAscent0(dataMatrix, classLabels):
    m, n = shape(dataMatrix)
    alpha = 0.01
    weights = ones(n)  # initialize to all ones
    for i in range(m):
        h = sigmoid(sum(dataMatrix[i] * weights))
        error = classLabels[i] - h
        weights = weights + alpha * error * dataMatrix[i]  # 回归系数的更新操作
    return weights


# 改进的随机梯度上升算法
def stocGradAscent1(dataMatrix, classLabels, numIter=150):  # 较之前的增加了一个迭代次数作为第三个参数，默认值150
    m, n = shape(dataMatrix)
    weights = ones(n)
    for j in range(numIter):
        dataIndex = list(range(m))
        for i in range(m):
            alpha = 4 / (1.0 + j + i) + 0.0001
            randIndex = int(random.uniform(0, len(dataIndex)))  # 样本随机选择
            h = sigmoid(sum(dataMatrix[randIndex] * weights))
            error = classLabels[randIndex] - h
            weights = weights + alpha * error * dataMatrix[randIndex]  # 回归系数的更新操作
            del (dataIndex[randIndex])
    return weights


# 以回归系数和特征向量作为输入计算对应的sigmoid值
def classifyVector(inX, weights):
    prob = sigmoid(sum(inX * weights))
    if prob > 0.5:
        return 1.0  # 如果sigmoid值大于0.5函数返回1，否则返回0
    else:
        return 0.0


# 打开测试集和训练集，并对数据进行格式化处理的函数
def colicTest():
    global x
    frTrain = open('train.txt')
    frTest = open('test1.txt')
    trainingSet = []
    trainingLabels = []
    for line in frTrain.readlines():
        currLine = line.strip().split(',')
        lineArr = []
        for i in range(2):
            lineArr.append(float(currLine[i]))
        trainingSet.append(lineArr)
        trainingLabels.append(float(currLine[2]))

    trainWeights = stocGradAscent1(array(trainingSet), trainingLabels, 100)  # 计算回归系数向量
    x = trainWeights
    print(trainWeights)
    errorCount = 0
    numTestVec = 0.0
    for line in frTest.readlines():
        numTestVec += 1.0
        currLine = line.strip().split(',')
        lineArr = []
        for i in range(2):
            lineArr.append(float(currLine[i]))
        if int(classifyVector(array(lineArr), trainWeights)) != int(currLine[2]):
            errorCount += 1
    errorRate = (float(errorCount) / numTestVec)
    print("the error rate of this test is: %f" % errorRate)
    return errorRate


# 调用函数colicTest()10次，并求结果的平均值
def multiTest():
    numTests = 10
    errorSum = 0.0
    for k in range(numTests):
        errorSum += colicTest()
    print("after %d iterations the average error rate is: %f" % (numTests, errorSum / float(numTests)))


if __name__ == "__main__":
    multiTest()
    plotBestFit(x)