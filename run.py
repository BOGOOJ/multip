import numpy as np
import mp as fun
import sys
import matplotlib.pyplot as plt
def main(argv):
    SearchAgents_no=50 # 麻雀数量初始化
    Function_name='F4' # 标准测试函数
    Max_iteration=1000  # 最大迭代次数
    [lb,ub,dim]=fun.Parameters(Function_name)  # 选择单峰测试函数为Function_name
    [fMin,bestX,SSA_curve]=fun.SSA(SearchAgents_no,Max_iteration,lb,ub,dim,Function_name)
    print(['最优值为：',fMin])
    print(['最优变量为：',bestX])
    
    plt.figure(figsize = [10,10])
    thr1=np.arange(len(SSA_curve[0,:]))
    plt.subplot(221)
    plt.plot(thr1[:], SSA_curve[0,:])
    plt.xlabel('generation')
    plt.ylabel('F4 value')
    plt.title('MutiStrategySSA-1000g')
    plt.subplot(222)
    plt.plot(thr1[500:], SSA_curve[0,500:])
    plt.xlabel('generation')
    plt.ylabel('F4 value')
    plt.title('MutiStrategySSA-500g')
    plt.subplot(223)
    plt.plot(thr1[100:], SSA_curve[0,100:])
    plt.xlabel('generation')
    plt.ylabel('F4 value')
    plt.title('MutiStrategySSA-100g')
  
    plt.subplot(224)
    plt.plot(thr1[985:], SSA_curve[0,985:])
    # ax.set_ylim(0,0.001)
    plt.xlabel('generation')
    plt.ylabel('F4 value')
    plt.title('MutiStrategySSA-15g')
    plt.show()
if __name__=='__main__':
	main(sys.argv)
