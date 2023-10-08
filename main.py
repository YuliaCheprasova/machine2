import random

import numpy as np
import math as m
from random import randint, uniform
import matplotlib.pyplot as plt

def Y(x):
    y=np.zeros(len(x))
    for i in range(len(x)):
        y[i]=m.sin(x[i]) + x[i]/10
        noise = np.random.normal(0, 0.5, 1)[0]
        noise = np.clip(noise, -y[i]*0.05, y[i] * 0.05)
        #print(y[i], " ", noise)
        y[i] += noise
    return y

def Y_true(x):
    y=np.zeros(len(x))
    for i in range(len(x)):
        y[i]=m.sin(x[i]) + x[i]/10
    return y

def K(u):
    if abs(u)<=1:
        k = 0.75*(1-u**2)
    else:
        k=0
    return k

def Ksum(z0, z, c):
    sum=0
    for i in range (len(z)):
        sum+=K((z0-z[i])/c)
    return sum



def M(x0, x, y, c):
    sum = 0
    ksum = Ksum(x0, x, c)
    if ksum==0:
        sum=0
    else:
        for i in range(len(x)):
            sum += (K((x0-x[i])/c))/ksum*y[i]
    return sum

def delete(j, x, n):
    xwithoutj = np.zeros(n-1)
    for i in range (j):
       xwithoutj[i] = x[i]
    for i in range (j, len(x)-1):
        xwithoutj[i] = x[i+1]
    return xwithoutj

def derivative(y,ypredict):
    gradient = np.zeros(len(y))
    for i in range(len(y)):
        gradient[i] = ypredict[i]-y[i]
    return 2*sum(gradient)/len(y)


if __name__ == '__main__':

    s = 100
    num_iteration = 50
    x = np.zeros(s)
    for i in range(s):
        x[i] = randint(1, 100)
    y_true = Y_true(x)
    y = Y(x)
    plt.scatter(x, y_true, label = "истинные значения")
    plt.scatter(x, y, label = "значения с шумом")
    #plt.show()
    ypredict = np.zeros(s)
    E1 = np.zeros(s)
    E2 = np.zeros(s)
    sorted_x = sorted(x)
    max_dist = sorted_x[1] - sorted_x[0]
    for j in range(len(sorted_x) - 1):
        if abs(sorted_x[j] - sorted_x[j + 1]) > max_dist:
            max_dist = abs(x[j] - x[j + 1])
    c = np.zeros(num_iteration+1)
    MSE = np.zeros(num_iteration+1)
    c[0] = uniform(max_dist, s / 3)
    for i in range(num_iteration):

        for j in range(s):
            x0 = x[j]
            xwithoutj = delete(j, x, s)
            y0 = y[j]
            ywithoutj = delete(j, y, s)
            ypredict[j] = M(x0, xwithoutj, ywithoutj, c[i])
            E1[j] = (ypredict[j] - y0) ** 2
            E2[j] = abs(ypredict[j] - y0)
        MSE[i] = sum(E1)/s
        NMSE = sum(E2) / (s*(max(y)-min(y)))
        gradient = derivative(y, ypredict)
        print(c[i], " ", MSE[i])
        c[i+1] = c[i] - 0.1 * gradient

    plt.scatter(x, ypredict, label = "предсказанные значения")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.show()
    plt.scatter(c, MSE)
    plt.show()
    #print(f"Среднеквадратическая ошибка = {MSE[num_iteration-1]}")
    #print(f"Нормализованная среднеквадратическая ошибка = {NMSE}")






