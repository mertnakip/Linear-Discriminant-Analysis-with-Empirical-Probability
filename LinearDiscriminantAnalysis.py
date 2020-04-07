import numpy as np
import math
from sklearn.metrics import accuracy_score


class LinearDiscriminantAnalysis:
    def __init__(self):
        self.Q = []
        self.w = []
        self.b = []
        self.threshold = 0

    def CalculateCovarianceMatrix(self, x, mu):
        S = np.zeros((x.shape[1], x.shape[1]))
        for x_s in x:
            S += np.matmul(np.transpose(x_s - mu[:, 0])[:, np.newaxis], (x_s - mu[:, 0])[np.newaxis, :])
        S = S / x.shape[0]
        return S

    def SeparateClasses(self, x, y):
        nSamples = x.shape[0]
        x0 = []
        x1 = []

        for s in range(nSamples):
            if y[s] == 0:
                x0.append(x[s])
            else:
                x1.append(x[s])

        x0 = np.array(x0)
        x1 = np.array(x1)

        mu0 = np.mean(x0, axis=0)
        mu1 = np.mean(x1, axis=0)

        return x0, x1, mu0[:, np.newaxis], mu1[:, np.newaxis]

    def SearchThreshold(self, x, y):
        acc = []
        threshold_list = []
        for i in range(20):
            self.threshold = i/10-1
            y_pred = self.predict(x)
            acc.append(accuracy_score(y, y_pred))

        self.threshold = np.argmax(acc)/10-1


    def fit(self, x, y):
        x0, x1, mu0, mu1 = self.SeparateClasses(x, y)
        S0 = self.CalculateCovarianceMatrix(x0, mu0)
        S1 = self.CalculateCovarianceMatrix(x1, mu1)

        self.Q = 1/2*(np.linalg.inv(S1)-np.linalg.inv(S0))
        self.w = np.matmul(np.transpose(mu0),np.linalg.inv(S0)) - np.matmul(np.transpose(mu1), np.linalg.inv(S1))

        P0 = x0.shape[1] / x.shape[1]
        P1 = x1.shape[1] / x.shape[1]

        self.b = (-1/2)*np.matmul(np.matmul(np.transpose(mu0), np.linalg.inv(S0)), mu0)\
                 + (1/2)*np.matmul(np.matmul(np.transpose(mu1), np.linalg.inv(S1)), mu1)\
                 - math.log((P0/P1)*math.sqrt(np.linalg.det(S0)/np.linalg.det(S1)))

        self.SearchThreshold(x, y)

    def predict(self, x):
        y_pred = np.zeros((x.shape[0],))
        for s in range(x.shape[0]):
            predicted_reg = np.matmul(np.matmul(x[s:s+1, :], self.Q), np.transpose(x[s:s+1, :])) + np.matmul(self.w, np.transpose(x[s:s+1, :])) + self.b
            if predicted_reg < self.threshold:
                y_pred[s] = 0
            else:
                y_pred[s] = 1

        return  y_pred
