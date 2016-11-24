import random
import numpy as np
from collections import defaultdict
from matplotlib import pyplot as plt

# def normalize(T):
#     for u in T:
#         All = sum(T[u].values())
#         if All>0:
#             for v in T[u]:
#                 T[u][v] /= All
#     return T
def gen_data(fname):
    for line in open(fname):
        try:
            yield [int(i)-1 for i in line.split()] #index start from 1
        except:
            print line

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
def average(Rr,c):
    All = 0.0;
    sumn = 0;
    for i in c:
        All += sum(Rr[i].values())
        sumn += len(Rr[i].values())
    return All/sumn
def rmse(U,V,R):
    error = 0.0
    nums = 0
    for u in R:
        for i in R[u]:
            error += (sigmoid(U[u].dot(V[i])) *5 - R[u][i])**2
            nums += 1
    return np.sqrt(error/nums)
def main(R,T,clists, N,M,K, lambdaU,lambdaV,lambdaT):
    def CircleCon3(T,clists):
        nc = defaultdict(dict)
        res = defaultdict(dict)
        for i in range(len(clists)):
            res[i] = defaultdict(dict)
        for u in T:
            for v in T[u]:
                All = 0
                for ci,c in enumerate(clists):
                    nc[ci]=(len(set(R[v].keys())&set(c)))
                    All += nc[ci]
                if All>0:
                    for ci,c in enumerate(clists):
                        if nc[ci]>0:
                            res[ci][u][v] = 1.0 * nc[ci]/All
        return res

    def costL(U,V,*args):
        vid,R,T,Rr=args
        cost=0.0
        bias = average(Rr,vid)
        for u in R:
            for i in (set(vid)&set(R[u])):
                cost += 0.5 * (R[u][i] - bias - U[u].dot(V[i]))**2
        cost += lambdaU/2 * np.linalg.norm(U)
        cost += lambdaV/2 * np.linalg.norm(V)
        for u in T:
            e = np.copy(U[u]) #1xK
            for v in T[u]:
                e -= T[u][v]*U[v]
            cost += lambdaT/2 * e.dot(e.T)
        return cost

    def gradient(U,V, *args):
        vid,R,T,Rr=args
        dU = np.zeros(U.shape)
        dV = np.zeros(V.shape)
        bias = average(Rr,vid)
        for u in R:
            for i in set(vid)&set(R[u]):
                dU[u] += V[i]*(bias+U[u].dot(V[i])-R[u][i]) 
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

        for i in vid:
            for u in Rr[i]:
                dV[i] += U[u] * (bias+U[u].dot(V[i])-R[u][i])
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

    def optim(x0,vid,cid):
        from scipy import optimize
        x0=np.vstack(x0).ravel()
        Sc = normalize(Se[cid])
        args=vid,R,Sc,Rr
        print "Sc",Sc
        x = optimize.fmin_cg(f, x0, fprime=gradf, args=args)
        x=x.reshape(N+M,K)
        u, v = x[:N,], x[N:,]
        return  u,v


    Rr = reverseR(R)
    Se = CircleCon3(T,clists)
    Uembd = defaultdict(dict)
    Vembd = np.random.normal(size=(M,K))
    print Vembd
    for cid,c in enumerate(clists):
        Uc = np.random.normal(size=(N,K))
        x0=Uc,Vembd
        Uc,Vembd = optim(x0,c,cid)
        Uembd[cid] = Uc
    return Uembd,Vembd

def test(R,T,C,N,M,K,max_r,lambdaU,lambdaV,lambdaT):
    print 'N:%d, M:%d, K:%d,lambdaU:%s, lambdaV:%s,lambdaT:%s' \
            %(N,M,K,lambdaU,lambdaV,lambdaT)
    #raw_input('Press any key to start...')
    Rr = reverseR(R)
    bias = []
    for c in C:
        bias.append(average(Rr,c))

    v2c = defaultdict(int)
    for cid,c in enumerate(C):
        for v in c:
            v2c[v] = cid

    Uembd,Vembd = main(R,T,C,N,M,K,lambdaU,lambdaV,lambdaT)
    print "u",Uembd
    print "v",Vembd  
    R_=np.zeros((N,M))
    for u in R:
        for v in R[u]:
            cid = v2c[v]
            if cid<len(Uembd):
                ub = Uembd[cid][u]
                vb = Vembd[v]
                R_[u][v] = max_r*max(0,(ub.dot(vb)+bias[cid]))
            else:
                R_[u][v] = 0;
    
    print R 
    print 'R_hat:\n', R_


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
    C=list()
    for i in xrange(len(R0)):
        for j in xrange(len(R0[0])):
            if R0[i][j]>0:
                R[i][j]=1.0 * R0[i][j] / max_r
    print R
    for i in xrange(len(T0)):
        for j in T0[i]:
            T[i][j-1]=1.0
    for i in xrange(3):
        if i%2:
            C.append([i,i+1])
    C = [[0,1,2,3]]
    test(R,T,C,N,M,K,max_r,lambdaU,lambdaV,lambdaT)

def t_yelp():
    #data from: http://www.trustlet.org/wiki/Epinions_datasets
    N,M = 0,0
    max_r = 5.0
    R=defaultdict(dict)
    T=defaultdict(dict)
    R_test=defaultdict(dict)
    print 'get T'
    # limit = 10**2
    for line in open('./data/users.txt'):
        u = int(line.split(':')[0])
        uf = line.split(':')[1][1:-1].split(',')
        if len(uf)>1:
            for x in line.split(':')[1][1:-1].split(',')[:-1]:
                v = int(x)
                T[u][v] = 1.0

    print 'get R'
    for line in open('./data/ratings-train.txt'):
        u,i,r = [int(x) for x in line.split('::')[:3]]
        R[u][i] = r/max_r
        N=max(N,u)
        M=max(M,i)
    N+=1
    M+=1
    for line in open('./data/ratings-test.txt'):
        u,i,r = [int(x) for x in line.split('::')[:3]]
        R_test[u][i] = r/max_r
    K=5
    # for line in open('./data/items.txt'):

    lambdaU,lambdaV,lambdaT=0.1, 0.1, 1.0
    # test(R,N,M,T,K,max_r,lambdaU,lambdaV,lambdaT)

if __name__ == "__main__":
#   t_epinion()
   t_yelp()
