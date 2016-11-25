import random
import numpy as np
from collections import defaultdict
from time import time
import pp


def sigmoid(z):
    return 1.0 / (1 + np.exp(-z))

def dsigmoid(z):
    return np.exp(-z)/(1+np.exp(-z))**2
def rmse(U,V,R):
    error = 0.0
    nums = 0
    for u in R:
        for i in R[u]:
            error += 25*(sigmoid(U[u].dot(V[i]))- R[u][i])**2
            nums += 1
    return np.sqrt(error/nums)

def reverseR(R):
    Rr=defaultdict(dict)
    for u in R:
        for i in R[u]:
            Rr[i][u] = R[u][i]
    return Rr

def PMF(R,N,M,K, lambdaU,lambdaV):
    def costL(U,V,*args):
        R,Rr=args
        cost=0.0
        for u in R:
            for i in R[u]:
                cost += 0.5 * (R[u][i] - sigmoid(U[u].dot(V[i])))**2
        cost += lambdaU/2 * np.linalg.norm(U)**2+lambdaV/2 * np.linalg.norm(V)**2
        return cost
    def gradient(U,V, *args):
        R,Rr=args
        dU = np.zeros(U.shape)
        dV = np.zeros(V.shape)
        for u in R:
            for i in R[u]:
                tmp = sigmoid(U[u].dot(V[i]))
                dU[u] += V[i] * tmp*(1-tmp) * (tmp-R[u][i]) 
            dU[u] += lambdaU * U[u]
        for i in Rr:
            for u in Rr[i]:
                tmp = sigmoid(U[u].dot(V[i]))
                dV[i] += U[u] * tmp*(1-tmp) * (tmp-R[u][i])
            dV[i] += lambdaV * V[i]
        if np.linalg.norm(dU)>1:
            dU = dU / np.linalg.norm(dU)
        if np.linalg.norm(dV)>1:
            dV = dV / np.linalg.norm(dV)
        return dU,dV
    def train(U,V):
        args=R,Rr
        res=[]
        steps=10**3
        rate = 0.1
        pregradU = 0
        pregradV = 0
        tol=1e-3
        momentum = 0.8
        stage = max(steps/100 , 1)
        for step in xrange(steps):
            dU,dV = gradient(U,V,*args)
            dU = dU + momentum*pregradU
            dV = dV + momentum*pregradV
            pregradU = dU
            pregradV = dV
            if not step%stage and rate>0.001:
                rate = 0.95*rate
            U -= rate * dU
            V -= rate * dV
            e = costL(U,V,*args)
            res.append(e)
            # if not step%stage:
            print step,e
            if step>100 and abs(sum(res[-10:])-sum(res[-20:-10]))<tol:
                print "====================" 
                print "stop in %d step"%(step)
                print "error is ",e
                print "====================" 
                break
        return U, V
    U = np.random.normal(0,0.01,size=(N,K))
    V = np.random.normal(0,0.01,size=(M,K))
    Rr = reverseR(R)
    return train(U,V)

def func(R,N,M,K,lambdaU,lambdaV,R_val):
    U,V = PMF(R,N,M,K,lambdaU,lambdaV)
    return U,V,rmse(U,V,R_val)


def t_movielens(ratio):
    #data from: http://files.grouplens.org/datasets/movielens/ml-100k.zip
    def gen_data(fname):
        N,M = 0,0
        data = []
        for line in open(fname,'r'):
            u,i = int(line.split()[0]),int(line.split()[1])
            data.append(line.split()[0:3])
            N=max(N,u)
            M=max(M,i)
        N+=1
        M+=1
        return data,N,M
    def get_R(data,l):
        R = defaultdict(dict)
        for li in l:
            u,i,r = [int(x) for x in data[li][0:3]]
            R[u][i] = r/max_r
        return R
    def test(K,lambdaU,lambdaV):
        print 'N:%d, M:%d, K:%d, lambdaU:%s, lambdaV:%s'%(N,M,K,lambdaU,lambdaV)
        ppservers = ()
        job_server = pp.Server(5,ppservers=ppservers)
        jobs = []
        repeatN = 5
        for t in range(repeatN):
            idlist = np.random.permutation(trainNum)
            RList = idlist[:int(0.8*trainNum)]
            valList = idlist[int(0.8*trainNum):]
            R = get_R(data,RList)
            R_val = get_R(data,valList)
            print "job begin"
            #func(R,N,M,K,lambdaU,lambdaV,R_val)
            jobs.append(job_server.submit(func,(R,N,M,K,lambdaU,lambdaV,R_val),(rmse,PMF,sigmoid,dsigmoid,reverseR),("numpy as np","from collections import defaultdict","random")))
        job_server.wait()
        print "jobs finish"
        sumrmse = 0.0
        for job in jobs:
            U,V,rmse1 = job()
            sumrmse += rmse1
        return sumrmse/repeatN,U,V
    
    max_r = 5.0
    data,N,M = gen_data("./u.data")
    rateNum = len(data)
    fout = "pp09pmf-"+str(ratio)+".ans"
#+++++++++++++++prepare train test data+++++++++++++
    trainNum = int(ratio*rateNum)
    R_test = get_R(data,range(trainNum,rateNum))
#======================================= lambda
    lambdaList = [0.1,1,10,100]
    K = 2 
    resUV = defaultdict(dict)
    timeUV = defaultdict(dict)
    lambdaU_ = 0.1
    lambdaV_ = 0.1
    minrmse = 1000000
    
    for lambdaU in lambdaList:
        for lambdaV in lambdaList:
            startTime = time()
            sumrmse,U,V = test(K,lambdaU,lambdaV)
            timeUV[lambdaU][lambdaV] = time() - startTime
            if sumrmse < minrmse:
                minrmse = sumrmse
                lambdaU_ = lambdaU
                lambdaV_ = lambdaV
            resUV[lambdaU][lambdaV] = sumrmse
            print "==================result=lambda======================"
            print "u,v,rmse,time",lambdaU,lambdaV,sumrmse,timeUV[lambdaU][lambdaV]
    with open(fout,'w') as f:
        print >>f,"=========train lambda========="
        print >>f,"lambda:u,v",lambdaU_,lambdaV_
        print >>f,"result of UV:",resUV
        print >>f,"training time of UV:",timeUV
#======================================= k
    Klist = [1,2,3,4,5]
    minrmse = 1000000
    resK = defaultdict(float)
    timeK = defaultdict(float)
    K_ = 2
    for K in Klist:
        startTime = time()
        sumrmse,U,V = test(K,lambdaU_,lambdaV_)
        timeK[K] = time() - startTime
        if sumrmse < minrmse:
            minrmse = sumrmse
            K_ = K
            U_ = U
            V_ = V
        resK[K] = sumrmse
        print "==================result=K============================="
        print "k,rmse,time",K,sumrmse,timeK[K]
    with open(fout,'a') as f:
        print >>f,"=========train K========="
        print >>f,"K:",K_
        print >>f,"result of K:",resK
        print >>f,"training time of K:",timeK
#=================test=======================
    restest = rmse(U_,V_,R_test)
    print "==================result=TEST============================="
    print "test:",restest
    with open(fout,'a') as f:
        print >>f,"=========test ==========="
        print >>f,"result of test:",restest


if __name__ == "__main__":
   t_movielens(0.8)
   t_movielens(0.2)
