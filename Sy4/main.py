from sklearn import neural_network as net
import matplotlib.pyplot as plt
from sklearn.externals import joblib

def load():
    f = open('train.txt')
    X_train,test = [],[]
    for line in f.readlines():
        X_train.append(eval(line))
    f = open('test.txt')
    for line in f.readlines():
        test.append(eval(line))
    f.close()
    return [X_train, test]
def save(a,file):
    p = open(file, 'w')
    for i in a:
        p.writelines(str(i)+'\n')
    p.close()
    print("Prediction is finished!")

def paintAndSave(result,test):
    save(result,'result.txt')
    c = ['yellow' if x == -1 else 'green' for x in result]
    plt.scatter([x[0] for x in test],[x[1] for x in test],c = c)
    plt.title('neural_network MLPClassifier Prediction')
    plt.savefig('result.jpg')
    plt.show()


train,test = load()
X_train = [x[0:2] for x in train]
Y_train = [x[2] for x in train]
net = net.MLPClassifier(hidden_layer_sizes=2,solver='sgd')
net.fit(X_train,Y_train)
result = net.predict(test)
paintAndSave(result,test)
#joblib.dump(net,"nerual_network.model")