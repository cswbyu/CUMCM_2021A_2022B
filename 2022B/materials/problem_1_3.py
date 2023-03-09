import math
from random import sample

import numpy as np
import pandas
from matplotlib import pyplot as plt
from sympy import symbols, solve

# 记录历次迭代的实际误差
radius_variance_history = []
angle_variance_history = []

# 初始化推断半径及推断角度
radius_inferred = 100
theta_1_inferred = 0

# 初始化实际位置与理想位置
real_locations_polar = [(0, 0), (100, 0), (98, 40.10), (112, 80.21), (105, 119.75), (98, 159.86), (112, 199.96),
                        (105, 240.07),
                        (98, 280.17), (112, 320.28)]
correct_locations_polar = [(0, 0), (100, 0), (100, 40), (100, 80), (100, 120), (100, 160), (100, 200),
                           (100, 240),
                           (100, 280), (100, 320)]
real_locations_cartesian = []
correct_locations_cartesian = []
for location in real_locations_polar:
    x = location[0] * math.cos(location[1] / 180 * math.pi)
    y = location[0] * math.sin(location[1] / 180 * math.pi)
    real_locations_cartesian.append([x, y])
for location in correct_locations_polar:
    x = location[0] * math.cos(location[1] / 180 * math.pi)
    y = location[0] * math.sin(location[1] / 180 * math.pi)
    correct_locations_cartesian.append([x, y])

# 实际位置
real_locations_df = pandas.DataFrame(real_locations_cartesian, columns=['x', 'y'])
# 理想位置
correct_locations_df = pandas.DataFrame(correct_locations_cartesian, columns=['x', 'y'])
# 推断位置
inferred_locations_df = pandas.DataFrame(correct_locations_cartesian, columns=['x', 'y'])


# 根据最新的推断半径和推断角度更新理想位置
def update():
    for i in range(1, 10):
        correct_locations_df.iloc[i, 0] = radius_inferred * np.cos((i - 1) / 9 * 2 * math.pi + theta_1_inferred)
        correct_locations_df.iloc[i, 1] = radius_inferred * np.sin((i - 1) / 9 * 2 * math.pi + theta_1_inferred)


# 求解基本模型
# id_a, id_b, id_c分别为A, B, C三点的编号
def solve_basic_model(id_a, id_b, id_c):
    # 根据实际位置求解tan_alpha1与tan_alpha2，作为观测到的已知输入
    k_oc_real = real_locations_df.iloc[id_c, 1] / real_locations_df.iloc[id_c, 0]
    k_ac_real = (real_locations_df.iloc[id_c, 1] - real_locations_df.iloc[id_a, 1]) / (
            real_locations_df.iloc[id_c, 0] - real_locations_df.iloc[id_a, 0])
    k_bc_real = (real_locations_df.iloc[id_c, 1] - real_locations_df.iloc[id_b, 1]) / (
            real_locations_df.iloc[id_c, 0] - real_locations_df.iloc[id_b, 0])
    tan_alpha1 = (k_oc_real - k_ac_real) / (1 + k_oc_real * k_ac_real)
    tan_alpha2 = (k_bc_real - k_oc_real) / (1 + k_bc_real * k_oc_real)

    x_c, y_c = symbols("x_c y_c")
    k_ac, k_oc, k_bc = symbols("k_ac k_oc k_bc")
    f1 = k_ac - (y_c - correct_locations_df.iloc[id_a, 1]) / (x_c - correct_locations_df.iloc[id_a, 0])
    f2 = k_oc - y_c / x_c
    f3 = k_bc - (y_c - correct_locations_df.iloc[id_b, 1]) / (x_c - correct_locations_df.iloc[id_b, 0])
    f4 = tan_alpha1 - (k_oc - k_ac) / (1 + k_oc * k_ac)
    f5 = tan_alpha2 - (k_bc - k_oc) / (1 + k_bc * k_oc)
    ans = solve([f1, f2, f3, f4, f5], [x_c, y_c, k_ac, k_oc, k_bc])
    return ans[0][0], ans[0][1]


# 求角度的偏差
# 仅用于误差估算，不可用于调整方案
def calculate_real_angle_variance():
    theta_list = []
    for i in range(1, 10):
        if i == 9:
            j = 1
        else:
            j = i + 1
        x_i = real_locations_df.iloc[i, 0]
        y_i = real_locations_df.iloc[i, 1]
        x_j = real_locations_df.iloc[j, 0]
        y_j = real_locations_df.iloc[j, 1]
        tan_i = y_i / x_i
        tan_j = y_j / x_j
        tan_theta = (tan_j - tan_i) / (1 + tan_j * tan_i)
        theta_list.append(math.atan(tan_theta))
    print("theta list:", theta_list)
    return np.var(np.array(theta_list))


# 求与圆心真实距离的均值及方差
# 仅用于误差估算，不可用于调整方案
def calculate_real_average_variance():
    x = np.array(real_locations_df.iloc[1:, 0], dtype=float)
    y = np.array(real_locations_df.iloc[1:, 1], dtype=float)
    distance = np.sqrt(x ** 2 + y ** 2)
    return np.average(distance), np.std(distance)


# 用于计算推断半径及angle
def calculate_inferred_radius_angle():
    x = np.array(inferred_locations_df.iloc[1:, 0], dtype=float)
    y = np.array(inferred_locations_df.iloc[1:, 1], dtype=float)
    distance = np.sqrt(x ** 2 + y ** 2)
    return np.average(distance), math.atan(inferred_locations_df.iloc[1, 1] / inferred_locations_df.iloc[1, 0])


iteration_times = 50
for iteration in range(iteration_times):

    total_list = [1, 2, 3, 4, 5, 6, 7, 8, 9]

    # 发射信号的无人机编号集合
    launch_list = sample(total_list, 3)

    # 接收信号的无人机编号集合
    for element in launch_list:
        total_list.remove(element)
    receive_list = total_list

    # 对接收信号的无人机进行位置调整
    for receiver in receive_list:
        x_inferred_1, y_inferred_1 = solve_basic_model(launch_list[0], launch_list[1], receiver)
        x_inferred_2, y_inferred_2 = solve_basic_model(launch_list[0], launch_list[2], receiver)
        x_inferred_3, y_inferred_3 = solve_basic_model(launch_list[1], launch_list[2], receiver)
        x_inferred = (x_inferred_1 + x_inferred_2 + x_inferred_3) / 3
        y_inferred = (y_inferred_1 + y_inferred_2 + y_inferred_3) / 3
        x_adjusted = real_locations_df.iloc[receiver, 0] + correct_locations_df.iloc[receiver, 0] - x_inferred
        y_adjusted = real_locations_df.iloc[receiver, 1] + correct_locations_df.iloc[receiver, 1] - y_inferred

        real_locations_df.iloc[receiver, 0] = x_adjusted
        real_locations_df.iloc[receiver, 1] = y_adjusted
        inferred_locations_df.iloc[receiver, 0] = x_inferred
        inferred_locations_df.iloc[receiver, 1] = y_inferred

    plt.scatter(real_locations_df.iloc[:, 0], real_locations_df.iloc[:, 1], color="red", s=17)

    # 计算与原点距离的平均值（即实际半径）和方差
    radius_real, radius_variance_real = calculate_real_average_variance()
    # 计算两两飞行器夹角的方差
    angle_variance = calculate_real_angle_variance()
    # 将上述两种方差添加到历史记录表中，便于之后作图
    radius_variance_history.append(radius_variance_real)
    angle_variance_history.append(angle_variance)

    theta = np.linspace(0, 2 * np.pi, 200)
    x_real = radius_real * np.cos(theta)
    y_real = radius_real * np.sin(theta)
    plt.plot(x_real, y_real, color="black", linewidth=1)
    #     x_inferred = radius_inferred * np.cos(theta)
    #     y_inferred = radius_inferred * np.sin(theta)
    #     plt.plot(x_inferred, y_inferred, color= "yellow")
    plt.axis("equal")

    # 更新推断半径和角度
    radius_inferred, theta_1_inferred = calculate_inferred_radius_angle()

    print("radius_real, radius_variance_real: ", radius_real, radius_variance_real)
    print("radius_inferred: ", radius_inferred)
    print("theta_1_inferred: ", theta_1_inferred)
    print("angle_variance: ", angle_variance)
    plt.title("iteration %d" % (iteration + 1))
    plt.show()
    update()
    print(correct_locations_df)
    print("iteration %d finished\n" % (iteration + 1))

plt.plot(range(1, (iteration_times + 1)), radius_variance_history)
plt.plot(range(1, (iteration_times + 1)), angle_variance_history)
plt.show()

plt.plot(range(1, (35 + 1)), radius_variance_history[0:35], color="red")
plt.xlabel("number of iterations")
plt.ylabel("variance of the distance to FY00")
plt.figure(figsize=(24, 32))
plt.show()

plt.plot(range(1, (35 + 1)), angle_variance_history[0:35], color="purple")
plt.xlabel("number of iterations")
plt.ylabel("variance of angle")
plt.figure(figsize=(24, 32))
plt.show()
