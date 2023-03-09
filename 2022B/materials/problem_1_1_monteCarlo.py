# -*- coding = utf-8 -*-
# @Time : 2022/9/16 14:13
# @Author : ty3
# @File : problem_1_1_monteCarlo.py
# @Software : PyCharm
import math
import random
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

from math  import *
from numpy import *
from sympy import *

theta_k = np.arange(0,2*math.pi, 2*math.pi/9)     # FY01-09无偏差坐标极角度
# 给定半径
R = 100
rho_k = np.zeros(9)
rho_k.fill(R)            # FY01-FY09极径
index_k = np.arange(0,9) # 设置FY0K下标


def product_random_difference():
    # 参照表中所给数据
    real_location_theta = [0, 40.10, 80.21, 119.75, 159.86, 199.96, 240.07, 280.17, 320.28]
    real_location_rho   = [100, 98, 112, 105, 98, 112, 105, 98, 112]
    # 理想极坐标                         
    ideal_location_theta = arange(0,360,360/9)
    ideal_location_rho = np.zeros(9)
    ideal_location_rho.fill(100)

    mean_dif_theta = np.mean(real_location_theta-ideal_location_theta)
    std_dif_theta  = np.std(real_location_theta-ideal_location_theta)

    mean_dif_rho   = np.mean(real_location_rho-ideal_location_rho)
    std_dif_rho    = np.std(real_location_rho-ideal_location_rho)

    dif_theta = np.random.normal(loc=mean_dif_theta,scale=std_dif_theta)
    dif_rho   = np.random.normal(loc=mean_dif_rho,  scale=std_dif_rho)
    print("随机偏差值：",dif_rho,dif_theta)
    return [dif_rho,dif_theta]

def monte_carlo(dif_rho,dif_theta):
# 使FY0X略有偏差
# 依据题目给定数据参考偏差范围（最好给定正态分布随机数）
# 实现正态分布随机数
    index_x = random.choice(index_k)  # 随机选择FY0X编号
    theta_x = theta_k[index_x] + dif_theta # 产生误差
    rho_x   = rho_k  [index_x] + dif_rho

    print("出现位置偏差的飞机：FY0",index_x+1,":",theta_x,rho_x)

    while True:
        index_a = random.choice(index_k)
        if index_a != index_x:
            break                           # 随机选择FY0A

    while True:
        index_b = random.choice(index_k)
        if index_b != index_x:
            if index_b != index_a:
                break                       # 随机选择FY0B
    print("除FY00外发送信号的飞机:FY0",index_a+1,"FY0",index_b+1)

    # 上帝视角求真实tan_alpha1 & tan_alpha2
    x_a = R*cos(theta_k[index_a])
    y_a = R*sin(theta_k[index_a])
    x_b = R*cos(theta_k[index_b])
    y_b = R*sin(theta_k[index_b])
    x_c = rho_x*cos(theta_x)
    y_c = rho_x*sin(theta_x)

    true_ac = (y_c-y_a) / (x_c-x_a)
    true_bc = (y_c-y_b) / (x_c-x_b)
    true_oc =  y_c/x_c
    talpha1 = atan((true_oc-true_ac)/(1+true_oc*true_ac))
    talpha2 = atan((true_bc-true_oc)/(1+true_bc*true_oc)) # FY0X的已知信息：alpha1+alpha2+(x_a,y_a)+(x_b+y_b)

    # print("传送给偏差飞机的角度值alpha1:",math.degrees(talpha1),",alpha2:",math.degrees(talpha2))
    print("传送给偏差飞机的角度值alpha1:", talpha1,",alpha2:", talpha2)


    x,y               = symbols('x y')
    k_ac, k_oc, k_bc  = symbols('k_ac k_oc k_bc') 

    # 还原FY0X定位过程
    # 直角坐标系解
    f1 = k_ac - (y-y_a)/(x-x_a)
    f2 = k_bc - (y-y_b)/(x-x_b)
    f3 = k_oc - y/x
    f4 = tan(talpha1) - (k_oc-k_ac)/(1+k_oc*k_ac)
    f5 = tan(talpha2) - (k_bc-k_oc)/(1+k_bc*k_oc)

    print("偏差飞机实际位置:",x_c,y_c)
    ans=sp.solve([f1,f2,f3,f4,f5],[x,y,k_ac,k_bc,k_oc])
    print("偏差飞机定位位置:",ans[0][0],ans[0][1])
    error_x = (ans[0][0]-x_c)/x_c
    error_y = (ans[0][1]-y_c)/y_c
    return [error_x,error_y]

# 按照正态分布产生误差
error_list = []
size=1001 #随机数数量
for i in range(size):
    dif = product_random_difference()
    # 使用蒙特卡洛模拟算法计算误差均值
    error_list.append(monte_carlo(dif[0],dif[1]))

# 绘制阵列图
plt.show()

# 计算
error_x = []
error_y = []
for element in error_list:
    error_x.append(element[0])
    error_y.append(element[1])

# plt.scatter(np.arange[ 1,len(error_x),1], error_x, color='red',s=5)
# plt.axis("equal")
ax1 = plt.subplot(211)
ax1.scatter(np.arange(0,size,1), error_x, color='red',s=1)
ax1.set_ylim(bottom=-1e-13,top=1e-13)
ax2 = plt.subplot(212)
ax2.scatter(np.arange(0,size,1), error_y, color='blue',s=1)
ax2.set_ylim(bottom=-1e-13,top=1e-13)
plt.show()
mean_err_x = np.mean(error_x)
mean_err_y = np.mean(error_y)
print("Xc误差率均值:",mean_err_x,"Yc误差率均值",mean_err_y)
