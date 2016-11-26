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
def meanap(U,V,R):
    ap = 0.0
    nums = 0
    for u in R:
        for i in R[u]:
            ub = U[u]
            vb = V[i]
            ap += abs(sigmoid(U[u].dot(V[i]))-R[u][i])/R[u][i]
            nums += 1
    return ap/nums
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
            if not step%stage:
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

def test(R,N,M,K,lambdaU,lambdaV,R_test):
    print 'N:%d, M:%d, K:%d,lambdaU:%s, lambdaV:%s' \
        %(N,M,K,lambdaU,lambdaV)
    start = time()
    U,V = PMF(R,N,M,K,lambdaU,lambdaV)
    print "=================RESULT======================="
    print 'K:%d,lambdaU:%s, lambdaV:%s' \
            %(K,lambdaU,lambdaV)
    print "rmse",rmse(U,V,R_test)
    print "map",meanap(U,V,R_test)
    print "time",time()-start
    return U,V,rmse(U,V,R_test),meanap(U,V,R_test)



def t_yelp(limitu,limiti):
    #data from: http://www.trustlet.org/wiki/Epinions_datasets
    N,M = 0,0
    max_r = 5.0
    cNum = 8
    R=defaultdict(dict)
    T=defaultdict(dict)
    R_test=defaultdict(dict)
    print 'get T'
    for line in open('./yelp_data/users.txt','r'):
        u = int(line.split(':')[0])
        uf = line.split(':')[1][1:-1].split(',')
        if len(uf)>1:
            for x in line.split(':')[1][1:-1].split(',')[:-1]:
                v = int(x)
                if u<limitu and v<limitu:
                    T[u][v] = 1.0

    print 'get R'
    for line in open('./yelp_data/ratings-train.txt','r'):
        u,i,r = [int(x) for x in line.split('::')[:3]]
        if u<limitu and i<limiti:
            R[u][i] = r/max_r
    #         N=max(N,u)
    #         M=max(M,i)
    # N+=1
    # M+=1
    print 'get R_test'
    for line in open('./yelp_data/ratings-test.txt','r'):
        u,i,r = [int(x) for x in line.split('::')[:3]]
        if u<limitu and i<limiti:
            R_test[u][i] = r/max_r

    lambdaU_,lambdaV_,K_=0.2, 0.2, 2
    U_,V_,rmse_,mae_ = test(R,N,M,K_,lambdaU_,lambdaV_,R_test)
    job_server = pp.Server(5)
    jobs = []
    for lambdaU in [0.1,0.5,1]:
        for lambdaV in [0.1,0.5,1]:
            jobs.append(job_server.submit(test,(R,N,M,K_,lambdaU,lambdaV,R),(meanap,rmse,PMF,sigmoid,dsigmoid,reverseR),("numpy as np","from collections import defaultdict","random","from time import time")))
    job_server.wait()
    for job in jobs:
            U,V,rmse1,mae1 = job()
            if mae1+rmse1<mae_+rmse_:
                U_ = U
                V_ = V
                lambdaU_ = lambdaU
                lambdaV_ = lambdaV
    print "jobs finish"
    jobs = []
    for K in [1,2,3,4,5]:
        jobs.append(job_server.submit(test,(R,N,M,K,lambdaU_,lambdaV_,R),(meanap,rmse,PMF,sigmoid,dsigmoid,reverseR),("numpy as np","from collections import defaultdict","random","from time import time")))
    job_server.wait()
    for job in jobs:
            U,V,rmse1,mae1 = job()
            if mae1+rmse1<mae_+rmse_:
                K_ = K
                U_ = U
                V_ = V
                lambdaU_ = lambdaU
                lambdaV_ = lambdaV
    print "=========all finish=============="
    print "rmse-test",rmse(U_,V_,R_test)
    print "map-test",meanap(U_,V_,R_test)
    
if __name__ == "__main__":
   t_yelp(100,2000)
