import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.svm import SVC
from sklearn.externals import joblib

def save(a,file):
    p = open(file, 'w')
    for i in a:
        p.writelines(str(i)+'\n')
    p.close()
    print("Prediction is finished!")

def paintAndSave(result,test,kernal):
    save(result,'result.txt')
    c = ['yellow' if x == -1 else 'green' for x in result]
    plt.scatter([x[0] for x in test],[x[1] for x in test],c = c)
    plt.title('Logistic Regression Prediction')
    plt.savefig(kernal+'result.jpg')
    plt.show()

def bc():
    data = pd.read_table(r'train.txt', header=None,sep=',')
    print(data.info())
    print(data.head())
    test = pd.read_table(r'test.txt',header= None,sep=',')
    X_train = np.array(data.loc[:][[0, 1]])
    y_train = np.array(data[2])
    y_train = np.where(y_train == 1, 1, -1)
    Xtest = np.array(test.loc[:][[0,1]])

    x_min = X_train[:, 0].min()
    x_max = X_train[:, 0].max()
    y_min = X_train[:, 1].min()
    y_max = X_train[:, 1].max()
    '''
    linear svm, poly svm, rbf svm
    '''
    plt.figure(figsize=(15, 15))
    for fig_num, kernel in enumerate(('linear', 'poly', 'rbf')):
        svm_ = SVC(kernel=kernel)
        svm_.fit(X_train, y_train)
        result = svm_.predict(Xtest)
        np.savetxt(kernel+'result.txt', result,fmt='%d')
        paintAndSave(result,Xtest,kernel)
        #joblib.dump(svm_,kernel+"_svm.model")

if __name__ == '__main__':
    bc()