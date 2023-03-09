import openpyxl
import math

workbook_1 = openpyxl.load_workbook(filename="附件1.xlsx")
workbook_2 = openpyxl.load_workbook(filename="附件2.xlsx")

sheet_1 = workbook_1['附件1']
sheet_2 = workbook_2['附件2']


class Node(object):
    def __init__(self, out_x, out_y, out_z):
        self.x = out_x
        self.y = out_y
        self.z = out_z


class NodeInformation(object):
    def __init__(self, out_direction_1, out_direction_2, out_pull_rope_length, out_length):
        self.direction_1 = out_direction_1
        self.direction_2 = out_direction_2
        self.pull_rope_length = out_pull_rope_length
        self.length = out_length


def calculate_distance(a1, a2):
    return math.sqrt((a1.x - a2.x) ** 2 + (a1.y - a2.y) ** 2 + (a1.z - a2.z) ** 2)


def get_direction(a1, a2):
    k1 = a2.x - a1.x
    k2 = a2.y - a1.y
    k3 = a2.z - a1.z
    k = math.sqrt(k1 ** 2 + k2 ** 2 + k3 ** 2)
    k1 = k1 / k
    k2 = k2 / k
    k3 = k3 / k
    if k3 < 0:
        k1 = -k1
        k2 = -k2
        k3 = -k3
    return k1, k2, k3


main_cable_node = {}
pull_rope_low_node = {}
pull_rope_high_node = {}
node_information = {}
node_namelist = []

for i in range(2226):
    name = sheet_1['A%d' % (i + 2)].value
    x = sheet_1['B%d' % (i + 2)].value
    y = sheet_1['C%d' % (i + 2)].value
    z = sheet_1['D%d' % (i + 2)].value
    main_cable_node[name] = Node(x, y, z)
    node_namelist.append(name)

for i in range(2226):
    name = sheet_2['A%d' % (i + 2)].value
    low_x = sheet_2['B%d' % (i + 2)].value
    low_y = sheet_2['C%d' % (i + 2)].value
    low_z = sheet_2['D%d' % (i + 2)].value
    high_x = sheet_2['E%d' % (i + 2)].value
    high_y = sheet_2['F%d' % (i + 2)].value
    high_z = sheet_2['G%d' % (i + 2)].value
    pull_rope_low_node[name] = Node(low_x, low_y, low_z)
    pull_rope_high_node[name] = Node(high_x, high_y, high_z)

    pull_rope_length = calculate_distance(pull_rope_low_node[name], pull_rope_high_node[name])
    length = calculate_distance(pull_rope_high_node[name], main_cable_node[name])
    direction_1 = get_direction(pull_rope_low_node[name], pull_rope_high_node[name])
    direction_2 = get_direction(main_cable_node[name], pull_rope_high_node[name])
    node_information[name] = NodeInformation(direction_1, direction_2, pull_rope_length, length)
