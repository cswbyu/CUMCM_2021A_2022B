import math
import random
import numpy as np
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

# 计算差值
def get_dif(idea,true):
    if idea>true :
        return idea-true
    else:
        return true-idea

# 产生服从正态分布的位置随机偏差值
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

# 选择产生位置偏差的接收无人机，使其位置产生偏差
# 返回编号与发生偏差后的位置信息
def choose_c(dif_rho,dif_theta):
    index_x = random.choice(np.arange(1,9))  # 随机选择FY0X编号
    theta_x = theta_k[index_x] + dif_theta  # 产生误差
    rho_x   = rho_k  [index_x] + dif_rho
    print("C:",index_x+1)
    return [index_x+1,theta_x,rho_x]

# 计算实际alpha值对
# 参数：接收无人机与发送无人机编号
# 返回：发送给被动接受无人机的alpha对
def cal_true_alpha(num_c, theta_c, rho_c, num_b):
    index_a = 0 # 已知FY01发送信息
    index_b = num_b-1
    index_c = num_c-1

    x_a = R*cos(theta_k[index_a])
    y_a = R*sin(theta_k[index_a])
    x_b = R*cos(theta_k[index_b])
    y_b = R*sin(theta_k[index_b])
    x_c = rho_c*cos(theta_c)
    y_c = rho_c*sin(theta_c)

    true_ac = (y_c-y_a) / (x_c-x_a)
    true_bc = (y_c-y_b) / (x_c-x_b)
    true_oc =  y_c/x_c
    talpha1 = atan((true_oc-true_ac)/(1+true_oc*true_ac))
    talpha2 = atan((true_bc-true_oc)/(1+true_bc*true_oc)) # FY0X的已知信息：alpha1+alpha2

    return [talpha1,talpha2]

def get_subdif(alpha1,alpha2,num_c):
    # print(math.degrees(alpha1),math.degrees(alpha2))
    print("alpha1:",math.degrees(alpha1),"alpha2:",math.degrees(alpha2))
    num_b=0
    min_dif=100.0 # 最小误差，初始值设为最大值
    second_dif=100.0 #记录第二小的误差
    # dif_limit=math.radians(8) # 误差限
    alpha_diagram=get_alpha_diagram()
    for i in range(0,8):
        if i==num_c-2: 
            continue
        ideal_alpha1 = alpha_diagram[i][num_c-2][0]
        ideal_alpha2 = alpha_diagram[i][num_c-2][1]
        cur_dif = get_dif(ideal_alpha1,alpha1)+get_dif(ideal_alpha2,alpha2)
        print("FY0",i+2,":",math.degrees(cur_dif))
        if cur_dif < min_dif:
            second_dif=min_dif
            min_dif=cur_dif
            num_b=i+2
        else :
            if cur_dif < second_dif:
                second_dif=cur_dif

    print("最小差值：",math.degrees(min_dif),"第二小差值：",math.degrees(second_dif))
    print("差值之差",math.degrees(get_dif(min_dif,second_dif)))
        
    print("预测标号及标准alpha角度对")
    print("FY0",num_b,np.round(math.degrees(alpha_diagram[num_b-2][num_c-2][0])),np.round(math.degrees(alpha_diagram[num_b-2][num_c-2][1])))
    return [num_b,get_dif(min_dif,second_dif)]

def choose_b(flag):
    print(flag)
    while True:
        num_b = random.choice(np.arange(2,10))
        if(flag[num_b-1]==0):         # 随机选择未被标记的FY0B
            return num_b
        if(sum(flag)==9):
            return 0

def find_process(times):
    flys_list=[] # 记录每次实际所需无人机的数组
    error=0
    correct=0 #
    dif_limit=math.radians(8)
    for i in range(times):
        flag=np.zeros(9) # 发射无人机选择标志
        flag[0]=1        # 标记FY01
        flys=1 #至少需要一架无人机
        dif = product_random_difference()
        info_c=choose_c(dif[0],dif[1])
        flag[info_c[0]-1]=1 # 标记接收机
        num_b=choose_b(flag)
        print("ACTUAL B1:",num_b)
        if num_b == 0:
            wrong+=1
            continue
        alpha_pair=cal_true_alpha(info_c[0],info_c[1],info_c[2],num_b)
        
        predict_b=get_subdif(alpha_pair[0],alpha_pair[1],info_c[0])
        print("GUESS B1:",predict_b[0])
        if predict_b[1] < dif_limit:
            flys+=1
            flag[num_b-1]=1 # 标记被抛弃的无人机编号
            num_b_copy=choose_b(flag)
            print("ACTUAL B2:",num_b_copy)
            alpha_pair_copy=cal_true_alpha(info_c[0],info_c[1],info_c[2],num_b_copy)
            predict_copy=get_subdif(alpha_pair_copy[0],alpha_pair_copy[1],info_c[0])
            print("GUESS B2:",predict_copy[0])
            if predict_b[1] < predict_copy[1] :
                predict_b=predict_copy
                num_b=num_b_copy
        if predict_b[0]!=num_b :
            print("ACTUAL:",num_b,"GUESS",predict_b[0])
            error+=1
            # break
        else :
            correct+=1
            plt.scatter(i,correct,color='red',s=0.01)
        flys_list.append(flys)

    print("平均所需无人机数量:",np.mean(flys_list))
    print("总错误率：",error/times)
    print("正确率：",1-error/times)
    return

find_process(1000)
