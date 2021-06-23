import matplotlib.pyplot as plt

def load():
    f = open('data\\train.txt')
    X_train,test = [],[]
    for line in f.readlines():
        X_train.append(eval(line))
    f = open('data\\test.txt')
    for line in f.readlines():
        test.append(eval(line))
    f.close()

    return [X_train, test]

class Navie_Bayes:
    def __init__(self, laplace):
        self.laplace = laplace

    def fit(self, X_train):
        nagtive = [x for x in X_train if x[2] == -1]
        postive = [x for x in X_train if x[2] == 1]
        self.nagtive, self.postive = nagtive, postive
        '''先验概率 prior[-1] No prior[1] Yes'''
        prior = {-1: (len(nagtive) + self.laplace) / (len(nagtive) + len(postive) + 2*self.laplace),
                 1: (len(postive) + self.laplace) / (len(nagtive) + len(postive) + 2*self.laplace)}
        ''' 
            cp含义
            cp[0][0] x,no cp[0][1]x,yes
            cp[1][0] y,no cp[1][1]y,yes
        '''
        cp = [[{}, {}], [{}, {}]]
        for j in range(2):#先跑x 再跑y
            for i in X_train:
                if i[2] == 1:   #Yes
                    res = cp[j][1].get(i[j])
                    if res == None:
                        cp[j][1][i[j]] = 1
                    else:
                        cp[j][1][i[j]] = res+1
                else:          #No
                    res = cp[j][0].get(i[j])
                    if res == None:
                        cp[j][0][i[j]] = 1
                    else:
                        cp[j][0][i[j]] = res + 1
        self.cp = cp
        self.prior = prior
        print("Model fitting finished")
        ''' 
            cp含义
            cp[0][0] x,no cp[0][1]x,yes
            cp[1][0] y,no cp[1][1]y,yes
        '''
    def answer(self,i):
        #yes
        x, y = i[0], i[1]
        pYes = self.prior[1] * (self.cp[0][1].get(x, 0) + self.laplace) / (len(self.cp[0][1]) * self.laplace + len(self.postive)) \
               * (self.cp[1][1].get(y, 0) + self.laplace) / (len(self.cp[1][1]) * self.laplace + len(self.postive))
        pNo = self.prior[-1] * (self.cp[0][0].get(x, 0) + self.laplace) / (len(self.cp[0][0]) * self.laplace + len(self.nagtive)) \
               * (self.cp[1][0].get(y, 0) + self.laplace) / (len(self.cp[1][0]) * self.laplace + len(self.nagtive))
        return 1 if pYes > pNo else -1

    def predict(self, X_test):
        y = []
        for i in X_test:
            y.append(self.answer(i))
        return y
def save(a,file):
    p = open(file, 'w')
    for i in a:
        p.writelines(str(i)+'\n')
    p.close()
    print("Prediction is finished!")

if __name__ == '__main__':
    X_train, test = load()
    model = Navie_Bayes(1)
    model.fit(X_train)
    result = model.predict(test)
    save(result,'data\\result2.txt')
    c = ['blue' if x == -1 else 'red' for x in result]
    plt.scatter([x[0] for x in test],[x[1] for x in test],c = c)
    plt.title('Bayes Model Predict')
    plt.show()


