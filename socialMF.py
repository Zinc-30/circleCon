import random
import numpy as np
from collections import defaultdict
from time import time
import pp

def sigmoid(z):
    return 1.0 / (1 + np.exp(-z))

def dsigmoid(z):
    return np.exp(-z)/(1+np.exp(-z))**2
def reverseR(R):
    Rr=defaultdict(dict)
    for u in R:
        for i in R[u]:
            Rr[i][u] = R[u][i]
    return Rr
def normalize(T):
    for u in T:
        All = sum(T[u].values())
        if All>0:
            for v in T[u]:
                T[u][v] /= All
    return T 
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

def socialMF(R,N,M, T, K, lambdaU,lambdaV,lambdaT):
    def costL(U,V,*args):
        R,T,Rr=args
        cost=0.0
        for u in R:
            for i in R[u]:
                cost += 0.5 * (R[u][i] - sigmoid(U[u].dot(V[i])))**2
        cost += lambdaU/2 * np.linalg.norm(U)+lambdaV/2 * np.linalg.norm(V)
        for u in T:
            e = np.copy(U[u]) #1xK
            for v in T[u]:
                e -= T[u][v]*U[v]
            cost += lambdaT/2 * e.dot(e.T)
        return cost

    def gradient(U,V, *args):
        R,T,Rr=args
        dU = np.zeros(U.shape)
        dV = np.zeros(V.shape)
        for u in R:
            for i in R[u]:
                tmp = U[u].dot(V[i])
                dU[u] += V[i] * dsigmoid(tmp) * (sigmoid(tmp)-R[u][i]) 
            dU[u] += lambdaU * U[u]
            e = np.copy(U[u]) #1xK
            for v in T[u]:
                e -= T[u][v] * U[v]
            dU[u] += lambdaT * e
            e2 = np.zeros_like(U[u])
            for v in T[u]:
                if v in T and u in T[v]:
                    e = np.copy(U[v])
                    for w in T[v]:
                        e-= T[v][w] * U[w]
                    e2 += T[v][u] * e 
            dU[u] -= lambdaT * e2

        for i in Rr:
            for u in Rr[i]:
                tmp = U[u].dot(V[i])
                dV[i] += U[u] * dsigmoid(tmp) * (sigmoid(tmp)-R[u][i])
            dV[i] += lambdaV * V[i]
        return dU,dV

    def f(x, *args):
        x=x.reshape(N+M,K)
        u, v = x[:N,], x[N:,]
        return costL(u, v, *args)

    def gradf(x, *args):
        x=x.reshape(N+M,K)
        u, v = x[:N,], x[N:,]
        gu,gv=gradient(u,v,*args)
        x_ = np.vstack((gu,gv)).ravel()
        return x_

    def optim(x0):
        from scipy import optimize
        x0=np.vstack(x0).ravel()
        args=R,T,Rr
        x = optimize.fmin_cg(f, x0, fprime=gradf, args=args)
        x=x.reshape(N+M,K)
        u, v = x[:N,], x[N:,]
        return  u,v

    U = np.random.normal(size=(N,K))
    V = np.random.normal(size=(M,K))
    Rr = reverseR(R)
    T=normalize(T)
    x0=U,V  
    return optim(x0)

def test(R,T,N,M,K,max_r,lambdaU,lambdaV,lambdaT,R_test):
    print 'N:%d, M:%d, K:%d,lambdaU:%s, lambdaV:%s,lambdaT:%s' \
            %(N,M,K,lambdaU,lambdaV,lambdaT)
    #raw_input('Press any key to start...')
    U,V = socialMF(R,N,M,T,K,lambdaU,lambdaV,lambdaT)  
    # vfn = np.vectorize(sigmoid)
    # R_hat = defaultdict(dict)
    # for u in R:
    #     for i in R[u]:
    #         R_hat[u][i] = sigmoid(U[u].dot(V[i])) *max_r
    # print 'R_hat:\n', R_hat
    start = time()
    print "rmse",rmse(U,V,R_test)
    print "map",meanap(U,V,R_test)
    print "time",time()-start
    return U,V,rmse(U,V,R_test),meanap(U,V,R_test)


def t_toy():
    R0 = [
         [5,3,0,1],
         [4,0,0,1],
         [1,1,0,5],
         [1,0,0,4],
         [0,1,5,4],
        ]
    max_r = 5.0
    T0 = [[3,2],[1,3,4],[2],[1,5],[3]]
    N,M,K=5,4,4
    lambdaU,lambdaV,lambdaT=0.02, 0.02, 0.0

    R=defaultdict(dict)
    T=defaultdict(dict)
    for i in xrange(len(R0)):
        for j in xrange(len(R0[0])):
            if R0[i][j]>0:
                R[i][j]=1.0 * R0[i][j] / max_r
    print R
    for i in xrange(len(T0)):
        for j in T0[i]:
            T[i][j-1]=1.0
    test(R,N,M,T,K,max_r,lambdaU,lambdaV,lambdaT)

def t_yelp():
    #data from: http://www.trustlet.org/wiki/Epinions_datasets
    N,M = 0,0
    max_r = 5.0
    cNum = 8
    R=defaultdict(dict)
    T=defaultdict(dict)
    R_test=defaultdict(dict)
    limitu = 1000
    limiti = 20000
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
        N=max(N,u)
        M=max(M,i)
        if u<limitu and i<limiti:
            R[u][i] = r/max_r

    N+=1
    M+=1
    print 'get R_test'
    for line in open('./yelp_data/ratings-test.txt','r'):
        u,i,r = [int(x) for x in line.split('::')[:3]]
        if u<limitu and i<limiti:
            R_test[u][i] = r/max_r

    lambdaU,lambdaV,lambdaT,K=0.2, 0.2, 0.1, 4
    job_server = pp.Server()
    jobs = []
    for lambdaU in [0.2,0.5,1]:
        for lambdaV in [0.2,0.5,1]:
            for lambdaT in [0.1,0.5,1]:
                # test(R,T,N,M,K,max_r,lambdaU,lambdaV,lambdaT,R_test)
                jobs.append(job_server.submit(test,(R,T,N,M,K,max_r,lambdaU,lambdaV,lambdaT,R),(meanap,rmse,socialMF,sigmoid,dsigmoid,reverseR,normalize),("numpy as np","from collections import defaultdict","random","from time import time")))
    job_server.wait()
    rmse_,mae_ = 100000,1000000
    for job in jobs:
            U,V,rmse1,mae1 = job()
            if mae1+rmse1<mae_+rmse_:
                U_ = U
                V_ = V
                lambdaU_ = lambdaU
                lambdaV_ = lambdaV
                lambdaT_ = lambdaT
                mae_,rmse_ = mae1,rmse1
    print "jobs finish"
    jobs = []
    for K in [1,2,3,4,5]:
        jobs.append(job_server.submit(test,(R,T,N,M,K,max_r,lambdaU_,lambdaV_,lambdaT,R),(meanap,rmse,socialMF,sigmoid,dsigmoid,reverseR,normalize),("numpy as np","from collections import defaultdict","random","from time import time")))
    job_server.wait()
    for job in jobs:
            U,V,rmse1,mae1 = job()
            if mae1+rmse1<mae_+rmse_:
                K_ = K
                U_ = U
                V_ = V
                lambdaU_ = lambdaU
                lambdaV_ = lambdaV
                mae_,rmse_ = mae1,rmse1
    print "jobs finish"
    print "rmse-test",rmse(U_,V_,R_test)
    print "map-test",meanap(U_,V_,R_test)


if __name__ == "__main__":
  t_yelp()
   # t_toy()
