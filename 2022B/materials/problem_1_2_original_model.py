import math
from re import X
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

from math  import *
from numpy import *
from sympy import *

# 令除FY00之外的飞机为A,B, 出现位置偏差的飞机为C。
# ∠ACB范围[-140,-120,-100,-80,-60,-40,-20,20,40,60,80,100,120,140]
# 计算∠ACO, ∠BCO的对应关系(alpha_1,alpha_2)
# 需要注意的是，当C点为∠的顶点（方向向左），且CO在∠BCA中间，BC在上，AC在下，此时alpha_1,alpha_2为正值

# 得到标准位置处的(alpha1,alpha2)序对
def set_alpha(num_a, num_b, num_c):
    index_a = num_a-1
    index_b = num_b-1
    index_c = num_c-1

    theta_k = np.arange(0,2*math.pi, 2*math.pi/9)     # FY01-09无偏差坐标极角度
    R = 100                  # 给定半径,假设为100
    rho_k = np.zeros(9)
    rho_k.fill(R)            # FY01-FY09极径

    # 上帝视角求真实tan_alpha1 & tan_alpha2
    x_a = R*cos(theta_k[index_a])
    y_a = R*sin(theta_k[index_a])
    x_b = R*cos(theta_k[index_b])
    y_b = R*sin(theta_k[index_b])
    x_c = R*cos(theta_k[index_c])
    y_c = R*sin(theta_k[index_c])

    true_ac = (y_c-y_a) / (x_c-x_a)
    true_bc = (y_c-y_b) / (x_c-x_b)
    true_oc =  y_c/x_c

    talpha1 = atan((true_oc-true_ac)/(1+true_oc*true_ac))
    talpha2 = atan((true_bc-true_oc)/(1+true_bc*true_oc)) # FY0X的已知信息：alpha1+alpha2+(x_a,y_a)+(x_b+y_b)

    return [talpha1,talpha2]

# 遍历得到(alpha1,alpha2)的二位数据表,且确定其中一架飞机为FY01
def get_alpha_diagram():
    diagram=[[0 for i in range(8)] for i in range(8)]
    for i in range(2,10):
        for j in range(2,10):
            if i == j :
                diagram[i-2][j-2]=[0,0]
                continue
            diagram[i-2][j-2]=set_alpha(1,i,j)
    return diagram

# 计算实际alpha与标准alpha差异值
def get_dif(idea,true):
    if idea>true :
        return idea-true
    else:
        return true-idea

# 当给定编号的偏差飞机得到(alpha1,alpha2)弧度制信息时，进行信息比对，取误差最小的编号序列即可
def find_b(alpha1,alpha2,num_c):
    # print(math.degrees(alpha1),math.degrees(alpha2))
    num_b=0
    min_dif=100.0 # 最小误差，初始值设为最大值
    alpha_diagram=get_alpha_diagram()
    for i in range(0,8):
        if i==num_c-2: 
            continue
        ideal_alpha1 = alpha_diagram[i][num_c-2][0]
        ideal_alpha2 = alpha_diagram[i][num_c-2][1]
        cur_dif = get_dif(ideal_alpha1,alpha1)+get_dif(ideal_alpha2,alpha2)
        print("FY0",i+2,":",math.degrees(cur_dif))
        if cur_dif < min_dif:
            min_dif=cur_dif
            num_b=i+2
    print("预测标号及标准alpha角度对")
    print("FY0",num_b,np.round(math.degrees(alpha_diagram[num_b-2][num_c-2][0])),np.round(math.degrees(alpha_diagram[num_b-2][num_c-2][1])))
    return num_b

# Monte Carlo验证
# 按照正态分布产生偏差值
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
    print("随机偏差值：",dif_rho,math.degrees(dif_theta))
    # return [0,0]
    return [dif_rho,dif_theta]

# 使用Monte_carlo产生弧度信息与发送信号的飞机编号
# 与问题1(1)基本一致
def monte_carlo(dif_rho,dif_theta):
# 使FY0X略有偏差
# 依据题目给定数据参考偏差范围（最好给定正态分布随机数）
# 实现正态分布随机数
    theta_k = np.arange(0,2*math.pi, 2*math.pi/9)     # FY01-09无偏差坐标极角度
    # 给定半径
    R = 100
    rho_k = np.zeros(9)
    rho_k.fill(R)            # FY01-FY09极径
    index_k = np.arange(0,9) # 设置FY0K下标
    index_x = random.choice(np.arange(1,9))  # 随机选择FY0X编号
    theta_x = theta_k[index_x] + dif_theta # 产生误差
    rho_x   = rho_k  [index_x] + dif_rho

    print("出现位置偏差的飞机：FY0",index_x+1,"当前位置:",math.degrees(theta_x),rho_x)
    error_c=index_x+1

    index_a=0                              #固定A飞机为FY01
    while True:
        index_b = random.choice(index_k)
        if index_b != index_x:
            if index_b != index_a:
                break                       # 随机选择FY0B
    launch_b=index_b+1
    # print("除FY00,FY01外发送信号的飞机:FY0",launch_b)

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

    print("传送给偏差飞机的角度值alpha1:",math.degrees(talpha1),",alpha2:",math.degrees(talpha2))
    # print("传送给偏差飞机的弧度值alpha1:", talpha1,",alpha2:", talpha2)
    return [talpha1, talpha2, error_c, launch_b]

size=1000
correct=0 #
for i in range(size):
    dif = product_random_difference()
    # 使用蒙特卡洛模拟算法计算误差率
    info=(monte_carlo(dif[0],dif[1])) # 获取弧度信息与既定编号:info[alpha1,alpha2,num_c,num_b]
    num_b=find_b(info[0],info[1],info[2])
    print("预测编号:",num_b,"实际编号",info[3])
    if num_b==info[3]:
        correct+=1
    else:
        print(num_b,"!!!!!!!!!")
    plt.scatter(i,correct,color='red',s=0.01)
x=np.arange(0,size,1)
y=x
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.legend('预测')
plt.xlabel("测试次数")
plt.ylabel("正确次数")
plt.plot(x,y+1,label='y = x')
plt.legend(loc='upper left')
percent=correct/size
print("预测编号正确率：", percent)
plt.show()
