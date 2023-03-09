import sympy
import code_1
import numpy

best_variation = -0.60
variance = float('inf')

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
Dp = -(A * px + B * py + C * pz)

for i in range(244):
    variation = -0.60 + i / 200
    flag = 1
    temp_sum = 0
    ox = (300 - variation) * A
    oy = (300 - variation) * B
    oz = (300 - variation) * C
    Do = -(A * ox + B * oy + C * oz)
    D = 2 * Do - Dp

    offset_list = []
    temp_variance = 0

    for j in range(2226):
        name = code_1.node_namelist[j]
        node = code_1.main_cable_node[name]
        x = node.x
        y = node.y
        z = node.z
        direction = code_1.node_information[name].direction_1
        delta_x = px - x
        delta_y = py - y
        delta_z = pz - z
        distance = numpy.linalg.norm(numpy.cross(numpy.array([delta_x, delta_y, delta_z]), numpy.array([A, B, C])))
        if distance < 150:
            t = sympy.Symbol('t')
            X = x + t * direction[0]
            Y = y + t * direction[1]
            Z = z + t * direction[2]
            eq = [(X - px) ** 2 + (Y - py) ** 2 + (Z - pz) ** 2 - (A * X + B * Y + C * Z + D) ** 2]
            ans = sympy.solve(eq, t)
            if len(ans) == 2:
                offset = min(abs(ans[0][0]), abs(ans[1][0]))
            else:
                offset = ans[t]

            if offset > 0.6:
                flag = 0
            else:
                offset_list.append(offset)

    temp_variance = numpy.var(offset_list)
    temp_sum = sum(offset_list)
    temp_average = numpy.mean(offset_list)
    print("variation = %f, temp_variance = %f, temp_sum = %f, temp_average = %f, count = %d" % (
        variation, temp_variance, temp_sum, temp_average, len(offset_list)))

    if flag and temp_variance < variance:
        best_variation = variation
        variance = temp_variance

print("best_variation = %f, variance = %f" % (best_variation, variance))
print("程序结束！")
