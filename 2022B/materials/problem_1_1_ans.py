import sympy as sp
import numpy as np
from sympy import *
from numpy import *
init_printing(use_unicode=True)

x,y = symbols('x y')
tan_talpha1, tan_talpha2 = symbols('tan_talpha1 tan_talpha2')
k_ac, k_oc, k_bc = symbols('k_ac k_oc k_bc')
x_a, y_a = symbols('x_a y_a')
x_b, y_b = symbols('x_b y_b')
R = Symbol('R')

f1 = k_ac - (y-y_a)/(x-x_a)
f2 = k_bc - (y-y_b)/(x-x_b)
f3 = k_oc - y/x
f4 = tan_talpha1 - (k_oc-k_ac)/(1+k_oc*k_ac)
f5 = tan_talpha2 - (k_bc-k_oc)/(1+k_bc*k_oc)

ans = sp.solve([f1,f2,f3,f4,f5],[x,y,k_ac,k_oc,k_bc])
print(ans)
print(sp.solve([f1,f2],[x,y]))