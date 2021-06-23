import sklearn.linear_model as lm
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
    plt.title('Logistic Regression Prediction')
    plt.show()

train,test = load()
model = lm.LogisticRegression()
X_train = [x[0:2] for x in train]
Y_train = [x[2] for x in train]
model.fit(X_train,Y_train)
result = model.predict(test)
paintAndSave(result,test)
joblib.dump(model,"logisticModel.model")
