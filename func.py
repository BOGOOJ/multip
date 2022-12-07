import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import ticker
# #例子1
# def f(X):
#     return np.power((X[0]-10),3)+np.power((X[1]-20),3)

# def g1(X):
#     return -np.power(X[0]-5,2)-np.power(X[1]-5,2)+100

# def g2(X):
#     return np.power(X[0]-6,2)+np.power(X[1]-5,2)-82.81


# def G(gi):
#     if gi>0:
#         return gi
#     return 0

# def H(hi,o):
#     if np.abs(hi)-0>0:
#         return np.abs(hi)
#     return 0

# #约束违规值
# def V(X):
#     return (G(g1(X))+G(g2(X)))/2

# x1u=100
# x1l=13
# x2u=100
# x2l=0

# dim=2
# globalbest=-6961.81388

# #例子8-----------------------------------------------------
# def Parameters():
#     dim=20
#     xl=0
#     xu=10
#     return [xl,xu,dim]
 
# def f(X):
#     a=np.sum( np.power( np.cos(X),4) )
#     b=2*np.cumprod( np.power( np.cos(X),2))[-1]
#     ctmp=np.arange(1,X.shape[0]+1,1)
#     c=np.sqrt( np.sum( ctmp * np.power(X,2) ))
#     print(a,b,c)
#     if c==0:
#         print(X)
#     return -np.abs( (a-b)/c)

# def g1(X):
#     return 0.75-np.cumprod(X)[-1]

# def g2(X):
#     n=20 #例子给的
#     return np.sum(X)-7.5*n



# def show():
#     fg = plt.figure()
#     ax = Axes3D(fg)
#     X=np.arange(0,10,0.1)
#     Y=np.arange(0,10,0.1)
#     X,Y=np.meshgrid(X,Y)

#     C=np.stack((X,Y),axis=2)
#     #方法过于拙劣 使用map将每一行分别映射f 有没有更好的
#     Z=[]
#     for i in range(C.shape[0]):
#         Z.append(list(map(f,C[0])))
#     Z=np.array(Z)
#     print("end")
#     min=0
#     for i in range(40):
#         for j in range(40):
#             if Z[i][j]<min:
#                 min=Z[i][j] 
#     print(min)
#     # # X[0].apend
#     # Z=[]
#     # for i in range(X.shape[0]):
#     #     z=[]
#     #     for j in range(X.shape[1]):
#     #         z.append(f(C[0]))
#     #     Z.append(Z)
        
#     ax.plot_surface(X,Y,Z,rstride=1,cstride=1,cmap='rainbow')

#     plt.show()
    
# # show()


#例子9---------------------------------------
def Parameters():
    dim=10
    xl=0
    xu=1
    return [xl,xu,dim]

def f(X):
    n=2
    res=-np.power( np.sqrt(n),n) * np.cumprod(X)[-1]
    return res 

def h(X):
    res = np.sum( np.power(X,2)) - 1
    return res
# X=np.arange(0.1,1,0.1)
# f(X)
def H(X,o=0.1): #o不知道设多少猜的
    if np.abs(h(X))-o>0:
        return np.abs(h(X))
    #百分之20违反约束违反度的---------------
    return 0

#约束违规值
def V(X):
    return H(X)

def show():
    fg = plt.figure()
    ax = Axes3D(fg)
    X=np.arange(0,1,0.01)
    Y=np.arange(0,1,0.01)
    X,Y=np.meshgrid(X,Y)
    n=2
    Z=-np.power( np.sqrt(n),n) *X*Y
    # Z2=np.power(X,2)+np.power(Y,2)-1
    # C=np.stack((X,Y),axis=2)
    # # #方法过于拙劣 使用map将每一行分别映射f 有没有更好的
    # Z=[]
    # for i in range(C.shape[0]):
    #     Z.append(list(map(f,C[0])))
    # Z=np.array(Z)
    # print("end")
    min=0
    # for i in range(40):
    #     for j in range(40):
    #         if Z[i][j]<min:
    #             min=Z[i][j] 
    # print(min)
    # # X[0].apend
    # Z=[]
    # for i in range(X.shape[0]):
    #     z=[]
    #     for j in range(X.shape[1]):
    #         z.append(f(C[0]))
    #     Z.append(Z)
        
    ax.plot_surface(X,Y,Z,rstride=1,cstride=1,cmap='rainbow')

    plt.show()
    
# show()