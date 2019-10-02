import numpy as np
import math
r=np.random
'''
a=1
b=2
c=a*b
d=np.ones([c])*3
print(d)
'''
def zuhe(a,b):
    """计算组合数"""
    return math.factorial(b)/(math.factorial(a)*math.factorial(b-a))

print((zuhe(3,50)))