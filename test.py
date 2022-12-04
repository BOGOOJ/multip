# line.py
 
import matplotlib.pyplot as plt
import numpy as np
 
from mpl_toolkits.mplot3d import Axes3D

import numpy as np
def cos_sim(a, b):
    a_norm = np.linalg.norm(a,axis=1)
    b_norm = np.linalg.norm(b)
    res = np.dot(a,b)/(a_norm * b_norm)
    return res

x=np.array([[1,2,4],[-1,3,-30]])

# x=np.random.randint(1,10,5)
# v=np.argsort(x)

# for i in range(5):
#     for j in range(v[i]+1,5):
#         print(x[v.tolist().index(j)],end=" ")
#     print()
# print(v)
# print(x)
b=np.zeros(3)
b[0]=1
# print(np.cos(x[0].T,b))
print(cos_sim(x,b))
print(cos_sim(x,b)*np.array([[1,-1],[-1,1]]))

# a_norm = np.linalg.norm(x,axis=1)
# print(a_norm)
# print(cos_sim(x,b)*a_norm)
