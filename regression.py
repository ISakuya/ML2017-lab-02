import numpy as np
import matplotlib.pyplot as plt
import sklearn.datasets as sd
from sklearn.model_selection import train_test_split
import random
import math

def getGradient(w,x,y):
    m = x.shape[0]
    n = x.shape[1]
    gradient = np.zeros((n,1))
    for k in range(batch):
        i = int(random.uniform(0,m-1))
        gradient += w
        if (1 - (y[i]*np.dot(w.T,x[i].T))[0])>0:
            gradient += 0-C*y[i]*(x[i].reshape((n,1)))
    return gradient/batch

def nag(w,x,y,vt):
    m = x.shape[0]
    n = x.shape[1]
    w = w - vt*gamma
    gradient = getGradient(w,x,y)
    vt = gamma*vt + alpha*gradient
    return vt

def RMSProp(w,x,y,s):
    g = getGradient(w,x,y)
    n = g.shape[0]
    tempg = g*g
    s = gamma*s +(1-gamma)*tempg
    for i in range(n):
        g[i][0] = alpha/math.sqrt(s[i][0]+epsl)*g[i][0]
    return g

def adaDelta(w,x,y,s,dx):
    g = getGradient(w,x,y)
    n = g.shape[0]
    tempg = g*g
    s = gamma*s +(1-gamma)*tempg
    for i in range(n):
        g[i][0] = math.sqrt(dx[i][0]+epsl)/math.sqrt(s[i][0]+epsl)*g[i][0]
    dx = gamma*dx +(1-gamma)*tempg
    return g

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
    return g

def getLoss(w,x,y):
    loss = 0
    for i in range(x.shape[0]):
        loss += max(0,1 - (y[i]*np.dot(w.T,x[i].T)[0]))
    return 0.5*(np.dot(w.T,w)[0][0])+C*loss

def getError(w,x,y):
    error = 0
    m = x.shape[0]
    for i in range(m):
        if x[i].dot(w)[0]>-0.85:
            predict = 1
        else: predict = -1
        if predict != y[i]: error += 1
    return error

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

algos = [getGradient,nag,RMSProp,adaDelta,adam]
sgdLossList = []
nagLossList = []
rmsLossList = []
adadLossList = []
adamLossList = []
alpha = 0.001
C = 1
times = 100
batch = 1000
for i in range(times):
    print(i)
    vt = nag(w_nag, X_train, y_train, vt)
    w_nag = w_nag - vt
    
    g_rms = RMSProp(w_rms, X_train, y_train, s_rms)
    w_rms = w_rms - g_rms
    
    g_adad = adaDelta(w_adad, X_train, y_train, s_adad, dx)
    w_adad = w_adad - g_adad
    
    g_adam = adam(w_adam, X_train, y_train, i, v, s_adam)
    w_adam = w_adam - g_adam

    nagLossList.append(getLoss(w_nag,X_test,y_test))
    rmsLossList.append(getLoss(w_rms,X_test,y_test))
    adadLossList.append(getLoss(w_adad,X_test,y_test))
    adamLossList.append(getLoss(w_adam,X_test,y_test))
time = np.arange(times)*batch

plt.plot(time, nagLossList)
plt.plot(time, rmsLossList)
plt.plot(time, adadLossList)
plt.plot(time, adamLossList)
plt.show()

print(getError(w_nag,X_test,y_test))
print(getError(w_rms,X_test,y_test))
print(getError(w_adad,X_test,y_test))
print(getError(w_adam,X_test,y_test))


'''
import numpy as np
import matplotlib.pyplot as plt
import sklearn.datasets as sd
from sklearn.model_selection import train_test_split
import random
import math

def getGradient(w,x,y,i):
    m = x.shape[0]
    n = x.shape[1]
    gradient = np.zeros((n,1))
    #for t in range(batch):
    for k in range(batch):
        i = int(random.uniform(0,m-1))
        gradient += w
        if (1 - (y[i]*np.dot(w.T,x[i].T))[0])>0:
            gradient += 0-C*y[i]*(x[i].reshape((n,1)))
    return gradient/batch

def nag(w,x,y,i,vt):
    m = x.shape[0]
    n = x.shape[1]
    w = w - vt*gamma
    #i = int(random.uniform(0,m-1))
    gradient = getGradient(w,x,y,i)
    vt = gamma*vt + alpha*gradient
    return vt

def RMSProp(w,x,y,i,s):
    g = getGradient(w,x,y,i)
    n = g.shape[0]
    tempg = g*g
    s = gamma*s +(1-gamma)*tempg
    for i in range(n):
        g[i][0] = alpha/math.sqrt(s[i][0]+epsl)*g[i][0]
    return g

def adaDelta(w,x,y,i,s,dx):
    g = getGradient(w,x,y,i)
    n = g.shape[0]
    tempg = g*g
    s = gamma*s +(1-gamma)*tempg
    for i in range(n):
        g[i][0] = math.sqrt(dx[i][0]+epsl)/math.sqrt(s[i][0]+epsl)*g[i][0]
    dx = gamma*dx +(1-gamma)*tempg
    return g

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
    return g

def getLoss(w,x,y):
    loss = 0
    for i in range(x.shape[0]):
        loss += max(0,1 - (y[i]*np.dot(w.T,x[i].T)[0]))
    return 0.5*(np.dot(w.T,w)[0][0])+C*loss

def getError(w,x,y):
    error = 0
    m = x.shape[0]
    for i in range(m):
        if x[i].dot(w)[0]>-0.85:
            predict = 1
        else: predict = -1
        if predict != y[i]: error += 1
    return error
v=2
alpha = 0.001
C = 1
times = 500
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

algos = [getGradient,nag,RMSProp,adaDelta,adam]
sgdLossList = []
nagLossList = []
rmsLossList = []
adadLossList = []
adamLossList = []
batch = 1000
for i in range(times):
    #print(i)
    gradient = 0
    j = int(random.uniform(0,X_train.shape[0]))
    gradient = getGradient(w_sgd, X_train, y_train, j)
    w_sgd = w_sgd - alpha*gradient
    
    vt = nag(w_nag, X_train, y_train, j, vt)
    w_nag = w_nag - vt
    
    g_rms = RMSProp(w_rms, X_train, y_train, j, s_rms)
    w_rms = w_rms - g_rms
    
    g_adad = adaDelta(w_adad, X_train, y_train, j, s_adad, dx)
    w_adad = w_adad - g_adad
    
    g_adam = adam(w_adam, X_train, y_train, j, i, v, s_adam)
    w_adam = w_adam - g_adam
    sgdLossList.append(getLoss(w_sgd,X_test,y_test))
    nagLossList.append(getLoss(w_nag,X_test,y_test))
    rmsLossList.append(getLoss(w_rms,X_test,y_test))
    adadLossList.append(getLoss(w_adad,X_test,y_test))
    adamLossList.append(getLoss(w_adam,X_test,y_test))
time = np.arange(times)
#time = time*X_train.shape[0]

#plt.plot(time, sgdLossList)
plt.plot(time, nagLossList)
plt.plot(time, rmsLossList)
plt.plot(time, adadLossList)
plt.plot(time, adamLossList)
plt.show()

#print(getError(w_sgd,X_test,y_test))
print(getError(w_nag,X_test,y_test))
print(getError(w_rms,X_test,y_test))
print(getError(w_adad,X_test,y_test))
print(getError(w_adam,X_test,y_test))

'''