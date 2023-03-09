import openpyxl
import code_1
import code_4
import numpy
import sympy

# 打开excel工作表'附件3.xlsx'
workbook = openpyxl.load_workbook(filename='附件3.xlsx')
sheet = workbook['附件3']

# 计算初始顶点, alpha, beta, p的数据
variation = -0.60
alpha0 = 36.795
beta0 = 78.169
alpha = alpha0 / 180 * numpy.pi
beta = beta0 / 180 * numpy.pi
cp_length = (1 - 0.466) * 300
A = -numpy.cos(beta) * numpy.cos(alpha)
B = -numpy.cos(beta) * numpy.sin(alpha)
C = -numpy.sin(beta)
px = cp_length * A
py = cp_length * B
pz = cp_length * C
p = [px, py, pz]


# 定义一个类：反射器
class Reflector(object):
    def __init__(self, first_node, second_node, third_node):
        self.first_node = first_node
        self.second_node = second_node
        self.third_node = third_node


reflectors_for_sphere = []
reflectors_for_paraboloid = []

# 读入附件3的数据
for i in range(4300):
    first = sheet['A%d' % (i + 2)].value
    second = sheet['B%d' % (i + 2)].value
    third = sheet['C%d' % (i + 2)].value
    reflectors_for_sphere.append(Reflector(first, second, third))

# 创建抛物面上的反射板构成的集合
for reflector in reflectors_for_sphere:
    r1 = reflector.first_node
    r2 = reflector.second_node
    r3 = reflector.third_node
    move_node_list = code_4.move_node_name_list
    if r1 in move_node_list and r2 in move_node_list and r3 in move_node_list:
        reflectors_for_paraboloid.append(reflector)


# 定义函数：已知三个点坐标，求出平面法向量
def get_normal_vector(x1, y1, z1, x2, y2, z2, x3, y3, z3):
    a = (y1 - y2) * (z2 - z3) - (y2 - y3) * (z1 - z2)
    b = (z1 - z2) * (x2 - x3) - (x1 - x2) * (z2 - z3)
    c = (x1 - x2) * (y2 - y3) - (y1 - y2) * (x2 - x3)
    d = sympy.sqrt(a ** 2 + b ** 2 + c ** 2)
    a = a / d
    b = b / d
    c = c / d
    if c < 0:
        a = -a
        b = -b
        c = -c
    return [a, b, c]


# 定义函数：求两向量的点乘
def get_dot_product(n1, n2):
    return n1[0] * n2[0] + n1[1] * n2[1] + n1[2] * n2[2]


# 定义函数：求行列式的值
def get_determinant(n1, n2, n3):
    sum_1 = n1[0] * n2[1] * n3[2] + n1[1] * n2[2] * n3[0] + n1[2] * n2[0] * n3[1]
    sum_2 = n1[2] * n2[1] * n3[0] + n1[1] * n2[0] * n3[2] + n1[0] * n2[2] * n3[1]
    return sum_1 - sum_2


# 定义函数：求两点间距离
def get_distance(l1, l2):
    return sympy.sqrt((l1[0] - l2[0]) ** 2 + (l1[1] - l2[1]) ** 2 + (l1[2] - l2[2]) ** 2)


# 定义函数：求向量模长
def get_modulus(n):
    return sympy.sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2])


# 定义函数：求直线与平面交点
def get_intersection(start, direction):
    g = sympy.Symbol('g')
    x = start[0] + direction[0] * g
    y = start[1] + direction[1] * g
    z = start[2] + direction[2] * g
    eq_1 = [A * (x - px) + B * (y - py) + C * (z - pz)]
    result = sympy.solve(eq_1, g)
    x = start[0] + direction[0] * result[g]
    y = start[1] + direction[1] * result[g]
    z = start[2] + direction[2] * result[g]
    return [x, y, z]


# 定义函数：求垂足
def get_drop_feet(point1, point2, point3):
    t0 = point2[0] - point1[0]
    t1 = point2[1] - point1[1]
    t2 = point2[2] - point1[2]
    s = sympy.Symbol('s')
    x = point1[0] + t0 * s
    y = point1[1] + t1 * s
    z = point1[2] + t2 * s
    eq = [(point3[0] - x) * t0 + (point3[1] - y) * t1 + (point3[2] - z) * t2]
    ans = sympy.solve(eq, s)
    x = point1[0] + t0 * ans[s]
    y = point1[1] + t1 * ans[s]
    z = point1[2] + t2 * ans[s]
    return [x, y, z]


# 求基准反射球面的接收比
def get_rate_for_sphere():
    print("sphere:")
    total = len(reflectors_for_sphere)
    print("total = %d" % total)
    count = 0
    for r in reflectors_for_sphere:
        vertex_1 = code_1.main_cable_node[r.first_node]
        vertex_2 = code_1.main_cable_node[r.second_node]
        vertex_3 = code_1.main_cable_node[r.third_node]
        v1 = [vertex_1.x, vertex_1.y, vertex_1.z]
        v2 = [vertex_2.x, vertex_2.y, vertex_2.z]
        v3 = [vertex_3.x, vertex_3.y, vertex_3.z]
        n1 = get_normal_vector(vertex_1.x, vertex_1.y, vertex_1.z, vertex_2.x, vertex_2.y, vertex_2.z, vertex_3.x,
                               vertex_3.y, vertex_3.z)
        n2 = [A, B, C]
        x0 = sympy.Symbol('x0')
        y0 = sympy.Symbol('y0')
        z0 = sympy.Symbol('z0')
        n = [x0, y0, z0]
        m1 = get_dot_product(n1, n2) / (get_modulus(n1) * get_modulus(n2))
        m2 = get_dot_product(n1, n) / (get_modulus(n1) * get_modulus(n))
        eq = [get_determinant(n1, n2, n), m1 + m2, x0 ** 2 + y0 ** 2 + z0 ** 2 - 1]
        ans = sympy.solve(eq, [x0, y0, z0])
        if get_distance(n2, (-ans[0][0], -ans[0][1], -ans[0][2])) < 0.001:
            direction = ans[1]
        else:
            direction = ans[0]

        intersection_1 = get_intersection(v1, direction)
        intersection_2 = get_intersection(v2, direction)
        intersection_3 = get_intersection(v3, direction)
        drop_feet_1 = get_drop_feet(intersection_1, intersection_2, p)
        drop_feet_2 = get_drop_feet(intersection_1, intersection_3, p)
        drop_feet_3 = get_drop_feet(intersection_2, intersection_3, p)
        distance_list = [get_distance(intersection_1, p), get_distance(intersection_2, p),
                         (get_distance(intersection_3, p))]
        if (intersection_1[0] - drop_feet_1[0]) * (drop_feet_1[0] - intersection_2[0]) > 0:
            distance_list.append(get_distance(drop_feet_1, p))
        if (intersection_1[0] - drop_feet_2[0]) * (drop_feet_2[0] - intersection_3[0]) > 0:
            distance_list.append(get_distance(drop_feet_2, p))
        if (intersection_2[0] - drop_feet_3[0]) * (drop_feet_3[0] - intersection_3[0]) > 0:
            distance_list.append(get_distance(drop_feet_3, p))
        min_distance = min(distance_list)

        if min_distance < 0.5:
            count = count + 1
    print(count / total)


# 求馈源舱有效区域接收到的反射信号与300米口径内反射面的反射信号之比
def get_rate_for_paraboloid():
    print("paraboloid:")
    total = len(reflectors_for_paraboloid)
    print("total = %d" % total)
    count = 0
    for r in reflectors_for_paraboloid:
        vertex_1 = code_1.main_cable_node[r.first_node]
        vertex_2 = code_1.main_cable_node[r.second_node]
        vertex_3 = code_1.main_cable_node[r.third_node]
        v1 = [vertex_1.x, vertex_1.y, vertex_1.z]
        v2 = [vertex_2.x, vertex_2.y, vertex_2.z]
        v3 = [vertex_3.x, vertex_3.y, vertex_3.z]
        n1 = get_normal_vector(vertex_1.x, vertex_1.y, vertex_1.z, vertex_2.x, vertex_2.y, vertex_2.z, vertex_3.x,
                               vertex_3.y, vertex_3.z)
        n2 = [A, B, C]
        x0 = sympy.Symbol('x0')
        y0 = sympy.Symbol('y0')
        z0 = sympy.Symbol('z0')
        n = [x0, y0, z0]
        m1 = get_dot_product(n1, n2) / (get_modulus(n1) * get_modulus(n2))
        m2 = get_dot_product(n1, n) / (get_modulus(n1) * get_modulus(n))
        eq = [get_determinant(n1, n2, n), m1 + m2, x0 ** 2 + y0 ** 2 + z0 ** 2 - 1]
        ans = sympy.solve(eq, [x0, y0, z0])
        if get_distance(n2, (-ans[0][0], -ans[0][1], -ans[0][2])) < 0.001:
            direction = ans[1]
        else:
            direction = ans[0]

        intersection_1 = get_intersection(v1, direction)
        intersection_2 = get_intersection(v2, direction)
        intersection_3 = get_intersection(v3, direction)
        drop_feet_1 = get_drop_feet(intersection_1, intersection_2, p)
        drop_feet_2 = get_drop_feet(intersection_1, intersection_3, p)
        drop_feet_3 = get_drop_feet(intersection_2, intersection_3, p)
        distance_list = [get_distance(intersection_1, p), get_distance(intersection_2, p),
                         (get_distance(intersection_3, p))]
        if (intersection_1[0] - drop_feet_1[0]) * (drop_feet_1[0] - intersection_2[0]) > 0:
            distance_list.append(get_distance(drop_feet_1, p))
        if (intersection_1[0] - drop_feet_2[0]) * (drop_feet_2[0] - intersection_3[0]) > 0:
            distance_list.append(get_distance(drop_feet_2, p))
        if (intersection_2[0] - drop_feet_3[0]) * (drop_feet_3[0] - intersection_3[0]) > 0:
            distance_list.append(get_distance(drop_feet_3, p))
        min_distance = min(distance_list)

        if min_distance < 0.5:
            count = count + 1
    print(count / total)


get_rate_for_sphere()
get_rate_for_paraboloid()
