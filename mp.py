import numpy as np
class funtion():
    def __init__(self):
        print("starting SSA")
def Parameters(F):
    if F=='F1':
        # ParaValue=[-100,100,30] [-100,100]代表初始范围，30代表dim维度
        ParaValue = [-100,100,30]

    elif F=='F2':
        ParaValue = [-10, 10, 30]

    elif F=='F3':
        ParaValue = [-100, 100, 30]

    elif F=='F4':
        ParaValue = [-100, 100, 30]

    elif F=='F5':
        ParaValue = [-30, 30, 30]

    elif F=='F6':
        ParaValue = [-100,100,30]
    return ParaValue
# 标准测试函数采用单峰测试函数（Dim = 30），计算适应度
def fun(F,X):  # F代表函数名，X代表数据列表
    if F == 'F1':
        O = np.sum(X*X)

    elif F == 'F2':
        O = np.sum(np.abs(X))+np.prod(np.abs(X))

    elif F == 'F3':
        O = 0
        for i in range(len(X)):
            O = O+np.square(np.sum(X[0:i+1]))


    elif F == 'F4':
        O = np.max(np.abs(X))

    elif F=='F5':
        X_len = len(X)
        O = np.sum(100 * np.square(X[1:X_len] - np.square(X[0:X_len - 1]))) + np.sum(np.square(X[0:X_len - 1] - 1))

    elif F == 'F6':
        O = np.sum(np.square(np.abs(X+0.5)))
    return O


# 对超过边界的个体随机分配到当前最优个体附近
def Bounds(s,Lb,Ub,bestX): #Lb是下限 ub是上限
    temp = s
    for i in range(len(s)):
        lowb=(bestX[i]*3)//4 #最好个体位置3/4处
        highb=(Ub[0,i]-bestX[i])//4+bestX[i]  #最好个体位置往上1/4
        if temp[i]<Lb[0,i] or temp[i]>Ub[0,i]:
            temp[i]=np.random.randint(lowb,highb)*np.random.random()+0.05  #保证位置在最优个体附近
    return temp

def cos_sim(a,dim):
    b=np.zeros(dim)
    b[0]=1
    a_norm = np.linalg.norm(a,axis=1)
    b_norm = np.linalg.norm(b)
    res = np.dot(a,b)/(a_norm * b_norm)
    return res

#不等式约束 有多个
def Gs(i,x):
    return x

#等式约束 有多个
def Hs(i,x):
    return x

#违规值
def Vs(x,Hs,Gs):
    Hn = 0 #等式约束个数
    Gn = 0 #不等式约束个数
    n=Hn+Gn #约束总个数
    res=0
    for i in range(n):
        res+=Gs(i,x)+Hs(i+Gn,x)
    return res/n

#变异算子  待优化------
def Ms(X,i,pNum,sortIndex):
    Fi = np.random.rand(1)#缩放因子
    choice_list = np.setdiff1d(np.arange(0,pNum,1),i)
    r1 = np.random.choice(choice_list)
    r2 = np.random.choice(np.setdiff1d(choice_list,r1))
    #(3) v = x+f*(x1-x2)  #x≠x1≠x2
    v=X[sortIndex[0,i],:] + Fi*(X[sortIndex[0,r1],:] - X[sortIndex[0,r2],:])
    return v
        
    #交叉算子
def C(x,i,j,v,cr,dim): #cr为交叉概率
    jr = np.random.randint(0,dim-1) #0到dim-1的一个随机整数
    if(np.random.random()<cr or j==jr):
        return v[j]
    return x


# pop是种群，M是迭代次数，f是用来计算适应度的函数
# pNum是生产者
def SSA(pop,M,c,d,dim,f):
    #global fit
    P_percent=0.2
    pNum = round(pop*P_percent)  # 生产者的人口规模占总人口规模的20%
    lb = c*np.ones((1,dim))  # 生成1*dim的全1矩阵，并全乘以c；lb是下限
    ub = d*np.ones((1,dim))  # ub是上限
    Xz = np.zeros((pop,dim))  # 生成pop*dim的全0矩阵，代表麻雀位置
    Xf = np.zeros((pop,dim)) #麻雀反向位置
    X  = np.zeros((pop,dim)) #麻雀最终位置
    # allfit = np.zeros((pop*2,1)) #双向适应度初始化
    fit = np.zeros((pop,1))   # 适应度值初始化
    
    
    #双向初始化 ---------------------------------------------------------------------------------------------
    for i in range(pop):
        Xz[i,:] = lb+(ub-lb)*np.random.rand(1,dim)  # 麻雀属性随机初始化初始值  np.random.rand(1,dim)返回一行 dim列个不同的0-1的随机数
        Xf[i,:] = ub+lb-Xz[i,:]
        # fit[i,0] = fun(f,X[i,:])  # 初始化最佳适应度值
        fz = fun(f,Xz[i,:])  #正向适应度值
        ff = fun(f,Xf[i,:])  #反向适应度值
        if ff<fz:     #求极小值所以适应度越小越好 保留小的
            fit[i,0] = ff
            X[i,:] = Xf[i,:]
        else:
            fit[i,0] = fz
            X[i,:] = Xz[i,:]
           
            
    pFit = fit.copy()  #最佳适应度矩阵
    pX = X.copy()  # 最佳种群位置
    fMin = np.min(fit[:,0]) # fMin表示全局最优适应值，生产者能量储备水平取决于对个人适应度值的评估
    bestI = np.argmin(fit[:,0])#函数用于返回一维列表最小值索引或多维列表展平之后的最小值索引
    bestX = X[bestI,:] # bestX表示fMin对应的全局最优位置的变量信息
    cr=0.2  #暂定 交叉概率为0.2-------------------------------------------
    Convergence_curve = np.zeros((1,M))  # 初始化收敛曲线
    
    # 迭代更新------------------------------------------------------------------------------------------------------------------------------------------------------------------
    for t in range(M): 
    #2、根据可行性法则将种群分为可行解和不可行解后排序划分   ------------------------  没做    
        
        sortIndex = np.argsort(pFit.T)  # 对麻雀的适应度值进行排序，按适应度从小到大进行排列下标  返回一维数组排序后的下标
        fmax = np.max(pFit[:,0])  # 取出最大的适应度值
        B = np.argmax(pFit[:,0])  # 取出最大的适应度值得下标
        worse = X[B,:]  # 最差适应度坐标
        r2 = np.random.rand(1) # 预警值
        
        # 这一部位为发现者（探索者）的位置更新 --------------------------------------------------------------------------------待检查
        if r2 < 0.8: # 预警值较小，说明没有捕食者出现
            for i in range(pNum):
                r1 = np.random.rand(1)
                X[sortIndex[0,i],:] = pX[sortIndex[0,i],:]*np.exp(-(i)/(r1*M))  # 对自变量做一个随机变换
                #处理边界
                X[sortIndex[0,i],:] = Bounds(X[sortIndex[0,i],:],lb,ub,bestX)  # 对超过边界的变量进行去除
                fit[sortIndex[0,i],0] = fun(f,X[sortIndex[0,i],:])   # 算新的适应度值   为什么要算两遍值-----------
        elif r2 >= 0.8: # 预警值较大，说明有捕食者出现威胁到了种群的安全，需要去其它地方觅食
            for i in range(pNum):
                # v=M(pX[sortIndex[0,i],:],pX[sortIndex[0,pop-i//2],:],pX[sortIndex[0,pop-i],:])  #存在x,x1,x2相同可能 待修改----
                v=Ms(pX,i,pNum,sortIndex)  #是pX------------------ pNum 是什么呢？  v为[0,pop,dim]的ndarray 有三维！
                for j in range(dim):
                    r = np.random.rand(1)
                    jr = np.random.randint(0,dim-1) 
                    if(r<=cr or j==jr):
                        X[sortIndex[0,i],j]=C(pX[sortIndex[0,i],j],i,j,v,cr,dim)
                    else:   
                        Q = np.random.rand(1)  # 也可以替换成  np.random.normal(loc=0, scale=1.0, size=1)
                        X[sortIndex[0,i],j] = pX[sortIndex[0,i],j]+Q  # Q是服从正态分布的随机数。
                X[sortIndex[0,i],:] = Bounds(X[sortIndex[0,i],:],lb,ub,bestX)
                fit[sortIndex[0,i],0] = fun(f,X[sortIndex[0,i],:])  #为什么要算两遍值-----------
        bestII = np.argmin(fit[:,0])
        bestXX = X[bestII,:]  #bestXX是当前发现者所占最优位置  bestX是全局最优位置



        #  这一部位为加入者（追随者）的位置更新------------------------------------------------------------------------------待检查
        for ii in range(pop-pNum):
            i = ii+pNum
            
            A = np.floor(np.random.rand(1,dim)*2)*2-1
            if i > pop/2:  #  这个代表这部分麻雀处于十分饥饿的状态（因为它们的能量很低，也就是适应度值很差），需要到其它地方觅食
                v=Ms(pX,i,pop,sortIndex)
                for j in range(dim):
                    r = np.random.rand(1)
                    jr = np.random.randint(0,dim-1) 
                    if(r<=cr or j==jr):
                        X[sortIndex[0,i],j]=C(pX[sortIndex[0,i],j],i,j,v,cr,dim)
                    else:   
                        Q = np.random.rand(1)  # 也可以替换成  np.random.normal(loc=0, scale=1.0, size=1)
                        X[sortIndex[0,i],j] = Q*np.exp((worse[j]-pX[sortIndex[0,i],j])/np.square(i))
            else:  # 这一部分追随者是围绕最好的发现者周围进行觅食，其间也有可能发生食物的争夺，使其自己变成生产者
                X[sortIndex[0,i],:] = bestXX+np.dot(np.abs(pX[sortIndex[0,i],:]-bestXX),1/(A.T*np.dot(A,A.T)))*np.ones((1,dim))
            X[sortIndex[0,i],:] = Bounds(X[sortIndex[0,i],:],lb,ub,bestX)
            fit[sortIndex[0,i],0] = fun(f,X[sortIndex[0,i],:])
            

        #侦察者位置更新-------------------------------------------------------------------------------------------------------------未完成 
        # 这一部位为意识到危险（注意这里只是意识到了危险，不代表出现了真正的捕食者）的麻雀的位置更新
        # np.arange()函数返回一个有终点和起点的固定步长的排列，如[1,2,3,4,5]，起点是1，终点是5，步长为1。
        # 一个参数时，参数值为终点，起点取默认值0，步长取默认值1
        arrc = np.arange(len(sortIndex[0,:]))
        #c=np.random.shuffle(arrc)
        # 这个的作用是在种群中随机产生其位置（也就是这部分的麻雀位置一开始是随机的，意识到危险了要进行位置移动，
        #  处于种群外围的麻雀向安全区域靠拢，处在种群中心的麻雀则随机行走以靠近别的麻雀）
        c = np.random.permutation(arrc)  # 随机排列序列
        b = sortIndex[0,c[0:20]]   #取20个随机麻雀作为侦察者
        for j in range(len(b)):
            if pFit[sortIndex[0,b[j]],0] > fMin:  #bestX要不要换成bestXX-------------
                X[sortIndex[0,b[j]],:] = bestX+np.random.rand(1,dim)*np.abs(pX[sortIndex[0,b[j]],:]-bestX)  #np.random.rand(1,dim)为一行dim列的符合正态分布的随机数
            else:  
                v=Ms(pX,b[j],pop,sortIndex)
                for k in range(dim):  #检查sortIndex[0,b[j],:]  -----没问题-----------
                    r = np.random.rand(1)
                    jr = np.random.randint(0,dim-1) 
                    if(r<=cr or k==jr):
                        X[sortIndex[0,b[j]],k]=C(pX[sortIndex[0,b[j]],k],j,k,v,cr,dim)
                    else:   
                        X[sortIndex[0,b[j]],k] = pX[sortIndex[0,b[j]],k]+(2*np.random.rand(1)-1)*np.abs(pX[sortIndex[0,b[j]],k]-worse[k])/(pFit[sortIndex[0,b[j]]]-fmax+10**(-50))
            X[sortIndex[0,b[j]],:] = Bounds(X[sortIndex[0,b[j]],:],lb,ub,bestX)
            fit[sortIndex[0,b[j]],0] = fun(f,X[sortIndex[0,b[j]]])
        

    #社区学习--------------------------------------------------------------------------------------------------------------------
    # 有的O里是没有的  原本效率低
        #各个个体Xi为圆心，指定半径Ri，划分各个个体Xi的社区区域，Xi在Oi随机选择个体来学习，生产新的个体Hi，然后根据适应度好坏进行选择 
        # 将X与原点距离从小排序找出排序后下标 提高查找效率
        #为什么这样社区就少了很多人呢？？？？？？？？？？？？？？？？？xia
        Xdist=np.linalg.norm(X,axis=1)*cos_sim(X,dim)  #加上正负方向
        XsortIndex =np.argsort(Xdist)  #一维
        for i in range(pop):
            O=[]#该个体社区内个体下标
            # -------Ri为0 
            XsortIndexI=XsortIndex[i]
            Ri=np.linalg.norm((pX[XsortIndexI,:]-X[XsortIndexI,:])) #每个个体的社区的半径
            # for j in range(i+1,pop):  #从Xi往前找
            for j in range(i+1,pop):  #从Xi往前找    
                if np.linalg.norm((X[XsortIndexI,:] - X[XsortIndex[j],:])) <= Ri:              #若两点之间距离小于等于Ri 则加入Oi中
                    O.append(XsortIndex[j])
                break
            #只找一边可以吗？----------------
            for j in range(i-1,-1,-1):  #从Xi往后找    
                if np.linalg.norm((X[XsortIndexI,:] - X[XsortIndex[j],:])) <= Ri:              #若两点之间距离小于等于Ri 则加入Oi中
                    O.append(XsortIndex[j])    
                break
                 #找到的是否有问题？-------------------
                 
            if(len(O)!=0):  #如果社区内找到个体则进行社区学习
                rand_o=np.random.randint(len(O))
                rand_l=np.random.randint(pop-1)
                rand_r=np.random.random()
                #(11)
                H = X[XsortIndexI,:] + rand_r*(pX[XsortIndex[rand_l],:] - X[XsortIndex[O[rand_o]],:])
                #(12)
                if fit[XsortIndexI,0]>=fun(f,H):
                    X[XsortIndexI,:]=H
                    fit[XsortIndexI,0]=fun(f,H)
            #(13)更新
            if fit[XsortIndexI,0]<=pFit[XsortIndexI,0]:  #感觉有些问题 下标是否要用sortIndex[0,i]？------------
                pX[XsortIndexI,:]=X[XsortIndexI,:]
                pFit[XsortIndexI,:]=fit[XsortIndexI,:]
            if pFit[XsortIndexI,:]< fMin:
                fMin=pFit[XsortIndexI,:]
                bestX=pX[XsortIndexI,:]
            print(O)

        #更新------------------------------------------------------------------------------------
        # for i in range(pop):
        #     if fit[i,0] < pFit[i,0]:
        #         pFit[i,0] = fit[i,0]
        #         pX[i,:] = X[i,:]
        #     if pFit[i,0] < fMin:
        #         fMin = pFit[i,0]
        #         bestX = pX[i,:]
        Convergence_curve[0,t] = fMin
        # print("Fmin:",fMin)
        # print("bestX:",bestX)
    return fMin,bestX,Convergence_curve






