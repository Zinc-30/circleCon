import random
import numpy as np
from collections import defaultdict
from scipy.sparse import csr_matrix as cm

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
    if sumn>0:
        return All/sumn
    else:
        return 0
    
def rmse(U,V,R,bias,v2c):
    error = 0.0
    nums = 0
    for u in R:
        for i in R[u]:
            cid = v2c[i]
            if cid<len(U):
                ub = U[cid][u]
                vb = V[i]
                error += (max(0,(ub.dot(vb)+bias[cid])) - R[u][i])**2
                nums += 1
    return np.sqrt(error/nums)
def mae(U,V,R,bias,v2c):
    ap = 0.0
    nums = 0
    for u in R:
        for i in R[u]:
            cid = v2c[i]
            if cid<len(U):
                ub = U[cid][u]
                vb = V[i]
                ap += abs(max(0,(ub.dot(vb)+bias[cid]))-R[u][i])/R[u][i]
                nums += 1
    return ap/nums
def circleRec(R,T,clists, N,M,K, lambdaU,lambdaV,lambdaT,Rmat):
    def CircleCon3(T,clists):
        nc = defaultdict(dict)
        res = defaultdict(dict)
        for i in range(len(clists)):
            res[i] = defaultdict(dict)
        for u in T:
            for v in T[u]:
                All = 0
                for ci,c in enumerate(clists):
                    if v in R:
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
    def get_csrmat(mat1,u,v):
        

    def gradient(U,V, *args):
        vid,R,T,Rr=args
        dU = np.zeros(U.shape)
        dV = np.zeros(V.shape)
        bias = average(Rr,vid)
        dU = (U.dot(V)+bias-Rmat).dot(V.T)
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
        x = optimize.fmin_cg(f, x0, fprime=gradf, args=args)
        x=x.reshape(N+M,K)
        u, v = x[:N,], x[N:,]
        return  u,v

    def train(U,V,vid,cid):
        Sc = normalize(Se[cid])
        args=vid,R,Sc,Rr
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

    Rr = reverseR(R)
    Se = CircleCon3(T,clists)
    Uembd = defaultdict(dict)
    Vembd = np.random.normal(0,0.01,size=(M,K))
    for cid,c in enumerate(clists):
        Uc = np.random.normal(0,0.01,size=(N,K))
        x0=Uc,Vembd
        # Uc,Vembd = optim(x0,c,cid)
        Uc,Vembd = train(Uc,Vembd,c,cid)
        Uembd[cid] = Uc
    return Uembd,Vembd

def test(R,T,C,N,M,K,max_r,lambdaU,lambdaV,lambdaT,R_test,Rmat):
    def lookR_hat():
        R_hat=np.zeros((N,M))
        for u in R:
            for v in R[u]:
                cid = v2c[v]
                if cid<len(Uembd):
                    ub = Uembd[cid][u]
                    vb = Vembd[v]
                    R_hat[u][v] = max_r*max(0,(ub.dot(vb)+bias[cid]))
                else:
                    R_hat[u][v] = 0
        print "R_hat",R_hat 
        return R_hat
    print 'N:%d, M:%d, K:%d,lambdaU:%s, lambdaV:%s,lambdaT:%s' \
            %(N,M,K,lambdaU,lambdaV,lambdaT)
    #raw_input('Press any key to start...')
    Uembd,Vembd = circleRec(R,T,C,N,M,K,lambdaU,lambdaV,lambdaT,Rmat)
    # print "u",Uembd
    # print "v",Vembd  
    Rr = reverseR(R)
    bias = []
    for c in C:
        bias.append(average(Rr,c))

    v2c = defaultdict(int)
    for cid,c in enumerate(C):
        for v in c:
            v2c[v] = cid

    # print 'R_hat:\n', R_hat
    print rmse(Uembd,Vembd,R_test,bias,v2c)
    print mae(Uembd,Vembd,R_test,bias,v2c)

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
    N,M,K=5,4,4
    lambdaU,lambdaV,lambdaT=0.02, 0.02, 0.01
    test(R,T,C,N,M,K,max_r,lambdaU,lambdaV,lambdaT,R)

def t_yelp():
    #data from: http://www.trustlet.org/wiki/Epinions_datasets
    N,M = 0,0
    max_r = 5.0
    cNum = 8
    R=defaultdict(dict)
    T=defaultdict(dict)
    R_test=defaultdict(dict)
    limit = 100
    print 'get T'
    for line in open('./yelp_data/users.txt','r'):
        u = int(line.split(':')[0])
        uf = line.split(':')[1][1:-1].split(',')
        if len(uf)>1:
            for x in line.split(':')[1][1:-1].split(',')[:-1]:
                v = int(x)
                if u<limit and v<limit:
                    T[u][v] = 1.0
    print 'get R'
    k = 0
    ul,il,rl = [],[],[]
    for line in open('./yelp_data/ratings-train.txt','r'):
        u,i,r = [int(x) for x in line.split('::')[:3]]
        if u<limit and i<limit:
            N=max(N,u)
            M=max(M,i)
            ul.append(u)
            il.append(i)
            rl.append(r)
            R[u][i] = r/max_r
    # print ul
    Rcsr = cm((rl,(ul,il)))
    N+=1
    M+=1
    print 'get R_test'
    for line in open('./yelp_data/ratings-test.txt','r'):
        u,i,r = [int(x) for x in line.split('::')[:3]]
        if u<limit and i<limit:
            R_test[u][i] = r/max_r
    print "get Circle"
    C = [[] for i in range(cNum)]
    for line in open('./yelp_data/items-class.txt','r'):
        i,ci = [int(x) for x in line.split(' ')]
        if i<limit:
            C[ci].append(i)

    lambdaU,lambdaV,lambdaT,K=0.2, 0.2, 0.1, 4
    test(R,T,C,N,M,K,max_r,lambdaU,lambdaV,lambdaT,R_test,Rcsr)

if __name__ == "__main__":
#   t_epinion()
   t_yelp()
   # t_toy()
