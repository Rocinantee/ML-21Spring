from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
from random import sample

def draw(train,labels,str,str1):
    x = [x[0] for x in train]
    y = [x[1] for x in train]
    d = {0: 'blue', 1: 'red', 2: 'green', 3: 'brown', 4: 'grey', 5: 'yellow', 6: 'cyan', 7: 'pink',8:'navy'}
    plt.title(str)
    plt.scatter(x, y, c=[d[x] for x in labels])
    plt.savefig(str1)
    plt.show()
def reshape():
    x_train = np.loadtxt('data\\train.txt', delimiter=',')
    reverse_list = sample(list(range(len(x_train))), len(x_train) // 20)
    for i in range(len(x_train)):
        if i in reverse_list:
            x_train[i][2] *= -1
    np.savetxt('data\\reversed_train.txt', x_train, fmt='%d', delimiter=',')


def proceed():
    train = np.loadtxt('data\\reversed_train.txt', delimiter=',')
    test = np.loadtxt('data\\test.txt', delimiter=',')
    Model = KMeans(n_clusters=2)
    Model.fit(train[:, 0:2])
    label = Model.labels_
    draw([x[0:2] for x in train], label, 'K-means train dataset', 'data\\train2.jpg')
    result = Model.predict(test)
    f = open('data\\result2.txt', 'w')
    for i in result:
        f.writelines(str(i)+'\n')
    draw(test, result, 'K-means test dataset', 'data\\test2.jpg')

#reshape()
proceed()