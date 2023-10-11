
import numpy as np
import math as m
from random import randint
import matplotlib.pyplot as plt


def Y(x1, x2):
    y=np.zeros(len(x1))
    for i in range(len(x1)):
        y[i]=m.sin(x1[i]) + x2[i]/10
        percent = randint(0, 5)
        noise = np.random.normal(0, 0.5, 1)[0]
        noise = np.clip(noise, (-y[i]*percent)/100, (y[i] * percent)/100)
        #print(y[i], " ", noise)
        y[i] += noise
    return y


def Y_true(x1, x2):
    y=np.zeros(len(x1))
    for i in range(len(x1)):
        y[i]=m.sin(x1[i]) + x2[i]/10
    return y


def K(u):
    if abs(u)<=1:
        k = 0.75*(1-u**2)
    else:
        k=0
    return k


def Ksum(z01, z02, z1, z2, c1, c2):
    sum=0
    for i in range(len(z1)):
        sum+=K((z01-z1[i])/c1)*K((z02-z2[i])/c2)
    return sum


def M(x01, x02, x1, x2, y, c1, c2):
    sum = 0
    ksum = Ksum(x01, x02, x1, x2, c1, c2)
    if ksum==0:
        sum=0
    else:
        for i in range(len(x1)):
            sum += K((x01-x1[i])/c1)*K((x02-x2[i])/c2)*y[i]
        sum = sum/ksum
    return sum

def delete(j, x, n):
    xwithoutj = np.zeros(n-1)
    for i in range (j):
       xwithoutj[i] = x[i]
    for i in range (j, len(x)-1):
        xwithoutj[i] = x[i+1]
    return xwithoutj

def max_distance(x):
    sorted_x = sorted(x)
    max_dist = 0
    prev = sorted_x[0]
    for j in range(len(sorted_x)):
        if abs(sorted_x[j] - prev) > max_dist:
            max_dist = abs(sorted_x[j] - prev)
        prev = sorted_x[j]
    return max_dist


if __name__ == '__main__':

    s = 100
    len_test = int(s * 0.2)
    x_start1 = np.zeros(s + len_test)
    x_start2 = np.zeros(s + len_test)
    for i in range(s + len_test):
        x_start1[i] = randint(1, 100)
        x_start2[i] = randint(1, 100)
    y_start = Y(x_start1, x_start2)
    y_true_start = Y_true(x_start1, x_start2)
    x1 = x_start1[:s]
    x2 = x_start2[:s]
    x_test1 = x_start1[s:]
    x_test2 = x_start2[s:]
    y = y_start[:s]
    y_true = y_true_start[:s]
    y_true_test = y_true_start[s:]
    y_test = y_start[s:]
    ypredict = np.zeros(s)
    ypredict_test = np.zeros(len_test)
    E1 = np.zeros(s)
    E2 = np.zeros(s)
    num_iteration = s
    max_dist1 = max_distance(x_start1)
    max_dist2 = max_distance(x_start2)
    #c = np.zeros(num_iteration)
    MSE = np.zeros(num_iteration**2)
    c1 = np.linspace(max_dist1, s/3, num_iteration)
    c2 = np.linspace(max_dist2, s/3, num_iteration)
    NMSE = np.zeros(num_iteration**2)
    count = 0
    for i in range(num_iteration):
        for k in range(num_iteration):
            for j in range(s):
                xi1 = x1[j]
                xwithoutj1 = delete(j, x1, s)
                xi2 = x2[j]
                xwithoutj2 = delete(j, x2, s)
                y0 = y[j]
                ywithoutj = delete(j, y, s)
                ypredict[j] = M(xi1, xi2, xwithoutj1, xwithoutj2, ywithoutj, c1[i], c2[k])
                E1[j] = (ypredict[j] - y0) ** 2
                E2[j] = abs(ypredict[j] - y0)
            MSE[count] = sum(E1)/s
            NMSE[count] = sum(E2) / (s*(max(y)-min(y)))
            if (count == 0):
                mse_min = MSE[count]
            elif(MSE[count] < mse_min - 0.01):
                mse_min = MSE[count]
                imin = count
                imin1 = i
                imin2 = k
                #print(MSE[count])
            count+=1
    #print(MSE[count-1])
    """for i in range(num_iteration**2):
        if (MSE[i] == min(MSE)):
            imin = i
            if (imin < 100):
                imin1 = 0
                imin2 = imin
            else:
                imin1 = imin//100
                imin2 = imin%100"""
    for j in range(s):
        xi1 = x1[j]
        xwithoutj1 = delete(j, x1, s)
        xi2 = x2[j]
        xwithoutj2 = delete(j, x2, s)
        y0 = y[j]
        ywithoutj = delete(j, y, s)
        ypredict[j] = M(xi1, xi2, xwithoutj1, xwithoutj2, ywithoutj, c1[imin1], c2[imin2])
        E1[j] = (ypredict[j] - y0) ** 2
        E2[j] = abs(ypredict[j] - y0)
    #MSE_t = sum(E1) / s
    #NMSE_t = sum(E2) / (s * (max(y) - min(y)))
    points = np.zeros(s)
    for i in range(s):
        points[i] = i+1
    points_test = np.zeros(len_test)
    for i in range(len_test):
        points_test[i] = i+1
    plt.plot(points, y_true, label="истинные значения")
    plt.scatter(points, y, label="значения с шумом")
    plt.plot(points, ypredict, label = "предсказанные значения")
    plt.xlabel("i")
    plt.ylabel("Yi")
    plt.legend()
    plt.show()
    #plt.plot(points, MSE)
    #plt.xlabel("i")
    #plt.ylabel("MSE")
    #plt.show()
    print(f"Параметр1, при котором среднеквадратическая ошибка минимальна = {c1[imin1]}")
    print(f"Параметр2, при котором среднеквадратическая ошибка минимальна = {c2[imin2]}")
    print(f"Среднеквадратическая ошибка = {MSE[imin]}")
    print(f"Нормализованная среднеквадратическая ошибка = {NMSE[imin]}")
    print("--------------------------------Тестовая выборка-----------------------------------")
    for j in range(len_test):
        xi1 = x_test1[j]
        xwithoutj1 = delete(j, x_test1, len_test)
        xi2 = x_test2[j]
        xwithoutj2 = delete(j, x_test2, len_test)
        y0 = y_test[j]
        ywithoutj = delete(j, y_test, len_test)
        ypredict_test[j] = M(xi1, xi2, xwithoutj1, xwithoutj2, ywithoutj, c1[imin1], c2[imin2])
        E1[j] = (ypredict_test[j] - y0) ** 2
        E2[j] = abs(ypredict_test[j] - y0)
    MSE_test = sum(E1) / len_test
    NMSE_test = sum(E2) / (len_test * (max(y_test) - min(y_test)))
    print(f"Среднеквадратическая ошибка при тесте = {MSE_test}")
    print(f"Нормализованная среднеквадратическая ошибка при тесте = {NMSE_test}")
    plt.plot(points_test, y_true_test, label="истинные значения")
    plt.scatter(points_test, y_test, label="значения с шумом")
    plt.plot(points_test, ypredict_test, label="предсказанные значения")
    plt.xlabel("i_test")
    plt.ylabel("Yi_test")
    plt.legend()
    plt.show()



