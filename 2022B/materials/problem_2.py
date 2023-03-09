import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

d = 50
dif_theta = 10
dif_rho = 0.2


def calculate_side_length(a, b):
    delta_x = real_locations_df.iloc[a - 1, 0] - real_locations_df.iloc[b - 1, 0]
    delta_y = real_locations_df.iloc[a - 1, 1] - real_locations_df.iloc[b - 1, 1]
    return math.sqrt(delta_x ** 2 + delta_y ** 2)


# 计算相邻两点之间距离的方差（所有点）
def calculate_variance():
    side_length_list = [calculate_side_length(1, 2), calculate_side_length(1, 3), calculate_side_length(2, 3),
                        calculate_side_length(2, 4), calculate_side_length(2, 5), calculate_side_length(3, 5),
                        calculate_side_length(3, 6), calculate_side_length(4, 5), calculate_side_length(5, 6),
                        calculate_side_length(4, 7), calculate_side_length(4, 8), calculate_side_length(5, 8),
                        calculate_side_length(5, 9), calculate_side_length(6, 9), calculate_side_length(6, 10),
                        calculate_side_length(7, 8), calculate_side_length(8, 9), calculate_side_length(9, 10),
                        calculate_side_length(7, 11), calculate_side_length(7, 12), calculate_side_length(8, 12),
                        calculate_side_length(8, 13), calculate_side_length(9, 13), calculate_side_length(9, 14),
                        calculate_side_length(10, 14), calculate_side_length(10, 15), calculate_side_length(11, 12),
                        calculate_side_length(12, 13), calculate_side_length(13, 14), calculate_side_length(14, 15)]
    return np.var(side_length_list)


# 计算相邻两点之间距离的方差（中间9个点）
def calculate_variance_1():
    side_length_list = [calculate_side_length(5, 8), calculate_side_length(8, 9), calculate_side_length(9, 5),
                        calculate_side_length(8, 4), calculate_side_length(4, 5), calculate_side_length(13, 8),
                        calculate_side_length(13, 9), calculate_side_length(6, 9), calculate_side_length(6, 5)]
    return np.var(side_length_list)


# 计算相邻两点之间距离的方差（中间3个点）
def calculate_variance_2():
    side_length_list = [calculate_side_length(5, 8), calculate_side_length(8, 9), calculate_side_length(9, 5)]
    return np.var(side_length_list)


# 求解从vector1旋转到vector2的夹角(逆时针为正)
# vector1: (x1, y1)
# vector2: (x2, y2)
def calculate_vector_angle(vector1, vector2):
    v1 = np.array(vector1, dtype=np.float64)
    v2 = np.array(vector2, dtype=np.float64)
    norm_product = np.linalg.norm(v1) * np.linalg.norm(v2)
    # 叉乘求解rho
    rho = np.arcsin(np.cross(v1, v2) / norm_product)
    # 点乘求解theta
    theta = np.arccos(np.dot(v1, v2) / norm_product)
    if rho < 0:
        return - theta
    else:
        return theta


correct_locations = [[2 * math.sqrt(3) * d, 0],
                     [3 / 2 * math.sqrt(3) * d, d / 2],
                     [3 / 2 * math.sqrt(3) * d, -d / 2],
                     [math.sqrt(3) * d, d],
                     [math.sqrt(3) * d, 0],
                     [math.sqrt(3) * d, -d],
                     [1 / 2 * math.sqrt(3) * d, 3 / 2 * d],
                     [1 / 2 * math.sqrt(3) * d, d / 2],
                     [1 / 2 * math.sqrt(3) * d, - d / 2],
                     [1 / 2 * math.sqrt(3) * d, - 3 / 2 * d],
                     [0, 2 * d],
                     [0, d],
                     [0, 0],
                     [0, -d],
                     [0, -2 * d]
                     ]

correct_locations[4] = [100, 0]
correct_locations[7] = [-80, 20]
correct_locations[8] = [0, 0]

# 初始化理想位置
correct_locations_df = pd.DataFrame(correct_locations)

# 初始化实际位置
real_locations = []
for location in correct_locations:
    real_locations.append([location[0] + dif_rho * math.cos(dif_theta), location[1] + dif_rho * math.sin(dif_theta)])

real_locations_df = pd.DataFrame(real_locations)


# id_a, id_b, id_c
# id_a为待调整的无人机
# 调整夹角为某一特定值（默认60°）
def adjust_angle_to_certain_degree(id_a, id_b, id_c, degree=60):
    id_a = id_a - 1
    id_b = id_b - 1
    id_c = id_c - 1

    vector_ab_real = np.array([real_locations_df.iloc[id_b, 0] - real_locations_df.iloc[id_a, 0],
                               real_locations_df.iloc[id_b, 1] - real_locations_df.iloc[id_a, 1]])
    vector_ac_real = np.array([real_locations_df.iloc[id_c, 0] - real_locations_df.iloc[id_a, 0],
                               real_locations_df.iloc[id_c, 1] - real_locations_df.iloc[id_a, 1]])
    vector_bisector_real = (vector_ab_real + vector_ac_real) / 2
    vector_bisector_real = vector_bisector_real / np.linalg.norm(vector_bisector_real)

    real_angle = calculate_vector_angle(vector_ab_real, vector_ac_real) / math.pi * 180
    if real_angle > degree:
        while real_angle > degree:
            real_locations_df.iloc[id_a, 0] -= vector_bisector_real[0] / 100
            real_locations_df.iloc[id_a, 1] -= vector_bisector_real[1] / 100
            vector_ab_real = np.array([real_locations_df.iloc[id_b, 0] - real_locations_df.iloc[id_a, 0],
                                       real_locations_df.iloc[id_b, 1] - real_locations_df.iloc[id_a, 1]])
            vector_ac_real = np.array([real_locations_df.iloc[id_c, 0] - real_locations_df.iloc[id_a, 0],
                                       real_locations_df.iloc[id_c, 1] - real_locations_df.iloc[id_a, 1]])
            vector_bisector_real = (vector_ab_real + vector_ac_real) / 2
            vector_bisector_real = vector_bisector_real / np.linalg.norm(vector_bisector_real)
            real_angle = calculate_vector_angle(vector_ab_real, vector_ac_real) / math.pi * 180
    elif real_angle < degree:
        while real_angle < degree:
            real_locations_df.iloc[id_a, 0] += vector_bisector_real[0] / 100
            real_locations_df.iloc[id_a, 1] += vector_bisector_real[1] / 100
            vector_ab_real = np.array([real_locations_df.iloc[id_b, 0] - real_locations_df.iloc[id_a, 0],
                                       real_locations_df.iloc[id_b, 1] - real_locations_df.iloc[id_a, 1]])
            vector_ac_real = np.array([real_locations_df.iloc[id_c, 0] - real_locations_df.iloc[id_a, 0],
                                       real_locations_df.iloc[id_c, 1] - real_locations_df.iloc[id_a, 1]])
            vector_bisector_real = (vector_ab_real + vector_ac_real) / 2
            vector_bisector_real = vector_bisector_real / np.linalg.norm(vector_bisector_real)
            real_angle = calculate_vector_angle(vector_ab_real, vector_ac_real) / math.pi * 180


plt.plot(real_locations_df.iloc[[4, 7, 8, 4], 0], real_locations_df.iloc[[4, 7, 8, 4], 1], color="red")
plt.axis("equal")
plt.show()
for i in range(100):
    adjust_angle_to_certain_degree(5, 8, 9)

    plt.plot(real_locations_df.iloc[[4, 7, 8, 4], 0], real_locations_df.iloc[[4, 7, 8, 4], 1], color="red")
    plt.axis("equal")
    plt.show()

    adjust_angle_to_certain_degree(8, 9, 5)

    plt.plot(real_locations_df.iloc[[4, 7, 8, 4], 0], real_locations_df.iloc[[4, 7, 8, 4], 1], color="red")
    plt.axis("equal")
    plt.show()

    adjust_angle_to_certain_degree(9, 5, 8)

    plt.plot(real_locations_df.iloc[[4, 7, 8, 4], 0], real_locations_df.iloc[[4, 7, 8, 4], 1], color="red")
    plt.axis("equal")
    plt.show()
