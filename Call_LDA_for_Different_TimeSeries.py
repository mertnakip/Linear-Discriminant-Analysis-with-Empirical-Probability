import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from random import gauss
from LinearDiscriminantAnalysis import LinearDiscriminantAnalysis as LDA_mine


# ============= Time Series 1 ================

time_series = []
for k in range(1, 1001):
    time_series.append(1/2*np.sign(np.cos(np.pi/2*k)*np.cos(np.pi/2*(2**(1/2))*k))+1/2)

time_series = np.array(time_series)

plt.plot(time_series[:100])
plt.xlabel('Samples')
plt.ylabel('x_k')

nPast = 10
nFuture = 1
nSamples = len(time_series)-nPast-nFuture

x = []
y = []

x0 = []
x1 = []

for s in range(nSamples):
    x.append(time_series[s:s+nPast])
    y.append(time_series[s+nPast:s+nPast+nFuture])

    if y[s] == 0:
        x0.append(x[s])
    else:
        x1.append(x[s])

x = np.array(x)
y = np.array(y)


x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=True, test_size=0.3)

model_mine = LDA_mine()
model_mine.fit(x_train, y_train)
y_test_pred_mine = model_mine.predict(x_test)

test_acc = accuracy_score(y_test[:, 0], y_test_pred_mine)

# ========= FROM THE LIBRARY =====

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

model = LDA()
model.fit(x_train, y_train[:, 0])
y_test_pred = model.predict(x_test)

test_acc_lib = accuracy_score(y_test[:, 0], y_test_pred)



# ============= Time Series 2 ================

time_series = []
for k in range(1, 1001):
    time_series.append(1/2*np.sign(gauss(0.0, 1.0))+1/2)

time_series = np.array(time_series)

plt.plot(time_series[:100])
plt.xlabel('Samples')
plt.ylabel('x_k')

nPast = 10
nFuture = 1
nSamples = len(time_series)-nPast-nFuture

x = []
y = []

x0 = []
x1 = []

for s in range(nSamples):
    x.append(time_series[s:s+nPast])
    y.append(time_series[s+nPast:s+nPast+nFuture])

    if y[s] == 0:
        x0.append(x[s])
    else:
        x1.append(x[s])

x = np.array(x)
y = np.array(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=True, test_size=0.3)

model_mine = LDA_mine()
model_mine.fit(x_train, y_train)
y_test_pred_mine = model_mine.predict(x_test)

test_acc = accuracy_score(y_test[:, 0], y_test_pred_mine)

# ========= FROM THE LIBRARY =====

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

model = LDA()
model.fit(x_train, y_train[:, 0])
y_test_pred = model.predict(x_test)

test_acc_lib = accuracy_score(y_test[:, 0], y_test_pred)

# ============= Logistic Map Function ================

time_series = []
x_k = [0.1]
threshold = 0.02
for k in range(1, 1001):
    time_series.append(1/2*np.sign(x_k[-1]-threshold)+1/2)
    x_k.append(1*x_k[-1]*(1-x_k[-1]))

time_series = np.array(time_series)

plt.plot(x_k[:100])
plt.plot(time_series[:100])
plt.legend(['x^k', 'x^k_q'])
plt.xlabel('Samples')

nPast = 10
nFuture = 1
nSamples = len(time_series)-nPast-nFuture

x = []
y = []

x0 = []
x1 = []

for s in range(nSamples):
    x.append(time_series[s:s+nPast])
    y.append(time_series[s+nPast:s+nPast+nFuture])

x = np.array(x)
y = np.array(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=True, test_size=0.3)

model_mine = LDA_mine()
model_mine.fit(x_train, y_train)
y_test_pred_mine = model_mine.predict(x_test)

test_acc = accuracy_score(y_test[:, 0], y_test_pred_mine)

# ========= FROM THE LIBRARY =====

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

model = LDA()
model.fit(x_train, y_train[:, 0])
y_test_pred = model.predict(x_test)

test_acc_lib = accuracy_score(y_test[:, 0], y_test_pred)
