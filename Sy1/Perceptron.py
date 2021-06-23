#Perceptron
#train and test
from sklearn.linear_model import Perceptron
from sklearn.externals import joblib
import numpy as np
import matplotlib.pyplot as plt

f = open("data\\train.txt")
f1 = open("data\\test.txt")
x,y = [],[]
line = f.readline()
while line:
    k = eval(line)
    x.append([k[0],k[1]])
    y.append(k[2])
    line = f.readline()
f.close()
X_train = np.array(x)
y = np.array(y)
model = Perceptron()
model.fit(X_train,y)

print("w:", model.coef_, "\nb:", model.intercept_, "\n")
testSource = []
line = f1.readline()
while line:
    k = eval(line)
    testSource.append([k[0],k[1]])
    line = f1.readline()
f1.close()
result = model.predict(testSource)
np.savetxt('data\\result1.txt', result, '%d')
joblib.dump(model, 'models\\m1.pkl')
color = ['red' if value == 1 else 'blue' for value in result]

#plt.scatter()
plt.scatter([x[0] for x in testSource], [x[1] for x in testSource], c=color)
line_x = np.arange(-100, 100)
line_y = line_x * (-model.coef_[0][0] / model.coef_[0][1]) - model.intercept_/model.coef_[0][1]
plt.plot(line_x, line_y)
plt.title('Perceptron Model Predict')
plt.show()
