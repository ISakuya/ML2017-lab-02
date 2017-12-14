from math import exp
import numpy as np
import matplotlib.pyplot as plt
import sklearn.datasets as sd
import random
import math
from sklearn.model_selection import train_test_split

def predict(xi, w):
    n = xi.shape[0]
    row = xi.reshape((1,n))
    yhat = row.dot(w)[0][0]
    return 1.0 / (1.0 + exp(-yhat))

def getLoss(w,x,y):
    m = x.shape[0]
    n = x.shape[1]
    loss = 0
    for i in range(m):
        xi = x[i].reshape((n,1))
        if y[i] == 1:
            loss += -math.log(predict(xi, w))
        else:
            loss += -math.log(1 - predict(xi, w))
    return loss

def getError(w,x,y):
    error = 0
    m = x.shape[0]
    for i in range(m):
        if predict(x[i],w)>-0.85:
            hx = 1
        else: hx = 0
        if hx != y[i]: error += 1
    return error/m

def getGradient(w,x,y):
    m = X_train.shape[0]
    n = X_train.shape[1]
    for k in range(batch):
        i = int(random.uniform(0,m-1))
        xi = x[i].reshape((n,1))
        yhat = predict(x[i], w)
        error = y_train[i] - yhat
        gradient = -error * yhat * (1.0 - yhat) * xi
    return gradient/batch
    
def updateWeight(w,x,y,i):
    gradient =  getGradient(w,x,y,i)
    w = w - alpha * gradient
    return w

def nag(w,x,y,vt):
    m = x.shape[0]
    n = x.shape[1]
    w = w - vt*gamma
    gradient = getGradient(w,x,y)
    vt = gamma*vt + alpha*gradient
    return w-vt

def RMSProp(w,x,y,s):
    g = getGradient(w,x,y)
    n = g.shape[0]
    tempg = g*g
    s = gamma*s +(1-gamma)*tempg
    for i in range(n):
        g[i][0] = alpha/math.sqrt(s[i][0]+epsl)*g[i][0]
    return w-g

def adaDelta(w,x,y,s,dx):
    g = getGradient(w,x,y)
    n = g.shape[0]
    tempg = g*g
    s = gamma*s +(1-gamma)*tempg
    for i in range(n):
        g[i][0] = math.sqrt(dx[i][0]+epsl)/math.sqrt(s[i][0]+epsl)*g[i][0]
    dx = gamma*dx +(1-gamma)*tempg
    return w-g

def adam(w,x,y,t,v,s):
    t = t+1
    g = getGradient(w,x,y)
    n = g.shape[0]
    v = gamma1*v+(1-gamma1)*g
    tempg = g*g
    s = gamma2*s+(1-gamma2)*tempg
    if t<200:
        tempv = v/(1-pow(gamma1,t))
        temps = s/(1-pow(gamma2,t)) + epsl
    else:
        tempv = v
        temps = s + epsl
    for i in range(n):
        temps[i][0] = math.sqrt(temps[i][0])
    g = alpha * tempv / temps
    return w-g


data_train = sd.load_svmlight_file('a9a')
data_test = sd.load_svmlight_file('a9a.t')
X_train,y_train = data_train[0],data_train[1]
X_test,y_test = data_test[0],data_test[1]
appendTrain = np.ones((X_train.shape[0],1))
appendTest = np.ones((X_test.shape[0],1))
X_train = np.column_stack((appendTrain,X_train.toarray()))
X_test = np.column_stack((appendTest,X_test.toarray()))

n = X_train.shape[1]
w_nag = np.zeros((n,1))
w_sgd = np.zeros((n,1))
w_rms = np.zeros((n,1))
w_adad = np.zeros((n,1))
w_adam = np.zeros((n,1))
vt = np.zeros((n,1))
gamma = 0.9
gamma1 = 0.9
gamma2 = 0.999
epsl = 0.00000001

s_rms = np.zeros((n,1))

s_adad = np.zeros((n,1))
dx = np.zeros((n,1))

v = np.zeros((n,1))
s_adam = np.zeros((n,1))

for i in range(y_train.shape[0]):
    if y_train[i] == -1: y_train[i] = 0
for i in range(y_test.shape[0]):
    if y_test[i] == -1: y_test[i] = 0


w = np.zeros((X_train.shape[1],1))

sgdLossList = []
nagLossList = []
rmsLossList = []
adadLossList = []
adamLossList = []
alpha = 0.01
times = 500
batch = 1000
for i in range(times):
    if i%10==0: print(i)
    w_nag = nag(w_nag, X_train, y_train, vt)
    w_rms = RMSProp(w_rms, X_train, y_train, s_rms)
    w_adad = adaDelta(w_adad, X_train, y_train, s_adad, dx)
    w_adam = adam(w_adam, X_train, y_train, i, v, s_adam)
    #sgdLossList.append(getLoss(w_sgd,X_test,y_test))
    nagLossList.append(getLoss(w_nag,X_test,y_test))
    rmsLossList.append(getLoss(w_rms,X_test,y_test))
    adadLossList.append(getLoss(w_adad,X_test,y_test))
    adamLossList.append(getLoss(w_adam,X_test,y_test))
time = np.arange(times)
plt.plot(time, nagLossList)
plt.plot(time, adadLossList)
plt.plot(time, rmsLossList)
plt.plot(time, adamLossList)
plt.show()
print(getError(w_nag,X_test,y_test))
print(getError(w_rms,X_test,y_test))
print(getError(w_adad,X_test,y_test))
print(getError(w_adam,X_test,y_test))


'''
from math import exp
import numpy as np
import matplotlib.pyplot as plt
import sklearn.datasets as sd
import random
import math
from sklearn.model_selection import train_test_split

def predict(xi, w):
    n = xi.shape[0]
    row = xi.reshape((1,n))
    yhat = row.dot(w)[0][0]
    return 1.0 / (1.0 + exp(-yhat))

def getLoss(w,x,y):
    m = x.shape[0]
    n = x.shape[1]
    loss = 0
    for i in range(m):
        xi = x[i].reshape((n,1))
        if y[i] == 1:
            loss += -math.log(predict(xi, w))
        else:
            loss += -math.log(1 - predict(xi, w))
    return loss

def getGradient(w,x,y,i):
    m = X_train.shape[0]
    n = X_train.shape[1]
    xi = x[i].reshape((n,1))
    yhat = predict(x[i], w)
    error = y_train[i] - yhat
    gradient = -error * yhat * (1.0 - yhat) * xi / m
    return gradient
    
def updateWeight(w,x,y,i):
    gradient =  getGradient(w,x,y,i)
    w = w - alpha * gradient
    return w

def nagUpdateWeight(w,x,y,i,vt):
    m = x.shape[0]
    n = x.shape[1]
    w = w - vt*gamma
    gradient = getGradient(w,x,y,i)
    vt = gamma*vt + alpha*gradient
    w = w - vt
    return w

def RMSProp(w,x,y,i,s):
    g = getGradient(w,x,y,i)
    n = g.shape[0]
    tempg = g*g
    s = gamma*s +(1-gamma)*tempg
    for i in range(n):
        g[i][0] = alpha/math.sqrt(s[i][0]+epsl)*g[i][0]
    w = w - g
    return w

def adaDelta(w,x,y,i,s,dx):
    g = getGradient(w,x,y,i)
    n = g.shape[0]
    tempg = g*g
    s = gamma*s +(1-gamma)*tempg
    for i in range(n):
        g[i][0] = math.sqrt(dx[i][0]+epsl)/math.sqrt(s[i][0]+epsl)*g[i][0]
    dx = gamma*dx +(1-gamma)*tempg
    w = w - g
    return w

def adam(w,x,y,i,t,v,s):
    t = t+1
    g = getGradient(w,x,y,i)
    n = g.shape[0]
    v = gamma1*v+(1-gamma1)*g
    tempg = g*g
    s = gamma2*s+(1-gamma2)*tempg
    if t<200:
        tempv = v/(1-pow(gamma1,t))
        temps = s/(1-pow(gamma2,t)) + epsl
    else:
        tempv = v
        temps = s + epsl
    #for i in range(n):
    #    g[i][0] = alpha * tempv[i][0]/sqrt(temps[i][0]+epsl)
    for i in range(n):
        temps[i][0] = math.sqrt(temps[i][0])
    g = alpha * tempv / temps
    w = w - g
    return w

data_train = sd.load_svmlight_file('a9a')
data_test = sd.load_svmlight_file('a9a.t')
X_train,y_train = data_train[0],data_train[1]
X_test,y_test = data_test[0],data_test[1]
appendTrain = np.ones((X_train.shape[0],1))
appendTest = np.ones((X_test.shape[0],1))
X_train = np.column_stack((appendTrain,X_train.toarray()))
X_test = np.column_stack((appendTest,X_test.toarray()))

n = X_train.shape[1]
w_nag = np.zeros((n,1))
w_sgd = np.zeros((n,1))
w_rms = np.zeros((n,1))
w_adad = np.zeros((n,1))
w_adam = np.zeros((n,1))
vt = np.zeros((n,1))
gamma = 0.9
gamma1 = 0.9
gamma2 = 0.999
epsl = 0.00000001

s_rms = np.zeros((n,1))

s_adad = np.zeros((n,1))
dx = np.zeros((n,1))

v = np.zeros((n,1))
s_adam = np.zeros((n,1))

for i in range(y_train.shape[0]):
    if y_train[i] == -1: y_train[i] = 0
for i in range(y_test.shape[0]):
    if y_test[i] == -1: y_test[i] = 0

alpha = 0.08
times = 10
w = np.zeros((X_train.shape[1],1))

sgdLossList = []
nagLossList = []
rmsLossList = []
adadLossList = []
adamLossList = []
for i in range(times):
    print(i)
    for j in range(X_train.shape[0]):
        w_nag = nagUpdateWeight(w_nag, X_train, y_train, j, vt)
        w_rms = RMSProp(w_rms, X_train, y_train, j, s_rms)
        w_adad = adaDelta(w_adad, X_train, y_train, j, s_adad, dx)
        w_adam = adam(w_adam, X_train, y_train, j, i, v, s_adam)
    #sgdLossList.append(getLoss(w_sgd,X_test,y_test))
    nagLossList.append(getLoss(w_nag,X_test,y_test))
    rmsLossList.append(getLoss(w_rms,X_test,y_test))
    adadLossList.append(getLoss(w_adad,X_test,y_test))
    adamLossList.append(getLoss(w_adam,X_test,y_test))
time = np.arange(times)
plt.plot(time, nagLossList)
plt.plot(time, adadLossList)
plt.show()
plt.plot(time, rmsLossList)
plt.plot(time, adamLossList)
plt.show()
'''