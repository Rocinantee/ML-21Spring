import matplotlib.pyplot as plt
import numpy as np
from sklearn import neural_network as net
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
import sklearn.linear_model as lm
from sklearn.ensemble import VotingClassifier as vt
from sklearn.model_selection import train_test_split
from random import sample

network = net.MLPClassifier(max_iter=50000)
svm_ = SVC(probability=True)
logisticModel = lm.LogisticRegression()


def load():
    return [np.loadtxt('data\\div_train.txt', delimiter=','), np.loadtxt('data\\div_test.txt', delimiter=',')]


def reshape():
    x_train = np.loadtxt('data\\train.txt', delimiter=',')
    reverse_list = sample(list(range(len(x_train))), len(x_train) // 20)
    for i in range(len(x_train)):
        if i in reverse_list:
            x_train[i][2] *= -1
    train_set, test_set = train_test_split(x_train, test_size=0.2, random_state=42)
    answer = test_set[..., 2]
    np.savetxt('data\\div_train.txt', train_set, fmt='%d', delimiter=',')
    np.savetxt('data\\div_test.txt', test_set, fmt='%d', delimiter=',')
    np.savetxt('data\\div_ans.txt', answer, fmt='%d', delimiter=',')


def accuray(texta):  # 计算分类器的准确度
    textb = np.loadtxt('data\\div_ans.txt', delimiter=',')
    error = 0
    for i in range(len(texta)):
        if texta[i] != textb[i]:
            error += 1
    return (len(texta) - error) / len(texta)


def boosting(trainX, trainY, test):
    estimators = [network, svm_, logisticModel]

    boostingModel = vt(estimators=[('lg', logisticModel), ('nt', network), ('svm', svm_)], voting='soft')

    for clf in (logisticModel, svm_, network, boostingModel):
        clf.fit(trainX, trainY)
        result = clf.predict(test)
        np.savetxt("data\\"+clf.__class__.__name__ + "result.txt", result, fmt='%d')
        print(clf.__class__.__name__, accuracy_score(test_ans, result))
        plt.scatter(test[:, 0], test[:, 1], c=['b' if x == -1 else 'r' for x in result])
        plt.title(clf.__class__.__name__)
        plt.show()

def paint():
    plt.scatter(trainX[:, 0], trainX[:, 1], c=['b' if x == -1 else 'r' for x in trainY])
    plt.title("trainSet")
    plt.show()
    plt.scatter(test[:, 0], test[:, 1], c=['b' if x == -1 else 'r' for x in test_ans])
    plt.title("TestSet")
    plt.show()

if __name__ == "__main__":
    train, test = load()
    trainX = train[:, 0:2]
    trainY = train[:, 2]
    test_ans = test[:, 2]
    test = test[:, 0:2]

    boosting(trainX, trainY, test)

    paint()



