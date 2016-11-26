import numpy as np
import scipy.sparse as sp
import pp
import time

def RSTE(R,S,N,M,K,lambdaU,lambdaV,lambdaT,R_test,ul,il):
    def sigmoid(z):
        return 1.0 / (1+np.exp(-z))
    def dsigmoid(z):
        return np.exp(-z)/np.power((1+np.exp(-z)),2)
    def rmse(U,V,R):
        keylist = np.array(R.todok().keys()).T
        utl = keylist[0]
        itl = keylist[1]
        error = (get_csrmat(sigmoid(U.dot(V.T)),utl,itl)-R).power(2).sum()/R.nnz
        return 5*np.sqrt(error)
    def mae(U,V,R):
        keylist = np.array(R.todok().keys()).T
        utl = keylist[0]
        itl = keylist[1]
        error = abs(get_csrmat(sigmoid(U.dot(V.T)),utl,itl)-R).sum()/R.nnz
        return error
    def get_csrmat(mat,ul,il):
        indx = ul*mat.shape[0]+il
        return sp.csr_matrix((np.take(np.array(mat),indx),(ul,il)),shape=(N,M))
    def costL(U,V):
        tmp = lambdaT*U.dot(V.T)+(1-lambdaT)*S.dot((U.dot(V.T)))
        Rx = get_csrmat(sigmoid(tmp),ul,il)
        cost = 0.5*((R - Rx).power(2)).sum()+0.5*lambdaU*np.linalg.norm(U)**2+0.5*lambdaV*np.linalg.norm(V)**2
        return cost
    def gradient(U,V):
        dU = lambdaU*U
        tmp = lambdaT*U.dot(V.T)+(1-lambdaT)*S.dot((U.dot(V.T)))
        Rv = get_csrmat(dsigmoid(tmp),ul,il)
        Rx = get_csrmat(sigmoid(tmp),ul,il)
        matx = Rv.multiply((Rx-R)).dot(V)
        dU += lambdaT*matx
        dU += (1-lambdaT)*(S.T).dot(matx)
        dV = lambdaV*V
        dV += (Rv.multiply((Rx-R))).T.dot(lambdaT*U+(1-lambdaT)*S.dot(U))
        # print dU,dV
        if np.max(dU)>1:
            dU = dU/np.max(dU)
        if np.max(dV)>1:
            dV = dV/np.max(dV)
        return dU,dV

    def train(U,V):
        res=[]
        steps=10**3
        rate = 0.1
        pregradU = 0
        pregradV = 0
        tol=1e-4
        momentum = 0.9
        stage = max(steps/100 , 1)
        for step in xrange(steps):
            dU,dV = gradient(U,V)
            dU = dU + momentum*pregradU
            dV = dV + momentum*pregradV
            pregradU = dU
            pregradV = dV
            if not step%stage and rate>0.0001:
                rate = 0.95*rate
            U -= rate * dU
            V -= rate * dV
            e = costL(U,V)
            res.append(e)
            if not step%(stage*5):
                print step,e
            if step>100 and abs(sum(res[-3:])-sum(res[-13:-10]))<tol:
                print "====================" 
                print "stop in %d step"%(step)
                print "error is ",e
                print "====================" 
                break
        return U, V
    U = np.random.normal(0,0.01,size=(N,K))
    V = np.random.normal(0,0.01,size=(M,K))
    start = time.time()
    U,V = train(U,V)
    print "=================RESULT======================="
    print 'K:%d,lambdaU:%s, lambdaV:%s,lambdaT:%s' \
            %(K,lambdaU,lambdaV,lambdaT)
    rmse_ans = rmse(U,V,R_test)
    mae_ans = mae(U,V,R_test)
    time_ans = time.time() - start
    print "rmse",rmse_ans
    print "mae",mae_ans
    print "time",time_ans
    return U,V,rmse_ans,mae_ans
def t_yelp(limitu,limiti):
    #data from: http://www.trustlet.org/wiki/Epinions_datasets
    def getdata():
        N,M = limitu,limiti
        max_r = 5.0
        cNum = 8
        T=sp.dok_matrix((N,N))
        print 'get T'
        for line in open('./yelp_data/users.txt','r'):
            u = int(line.split(':')[0])
            uf = line.split(':')[1][1:-1].split(',')
            if len(uf)>1:
                for x in line.split(':')[1][1:-1].split(',')[:-1]:
                    v = int(x)
                    if u<limitu and v<limitu:
                        T[u,v] = 1.0
        T = T.tocsr()
        print 'get R_test'
        utl,itl,rtl = [],[],[]
        for line in open('./yelp_data/ratings-test.txt','r'):
            u,i,r = [int(x) for x in line.split('::')[:3]]
            if u<limitu and i<limiti:
                utl.append(u)
                itl.append(i)
                rtl.append(r/5.0)
        utl,itl = np.array(utl),np.array(itl)
        R_test = sp.csr_matrix((rtl,(utl,itl)),shape=(N,M))
        print 'get R'
        ul,il,rl = [],[],[]
        for line in open('./yelp_data/ratings-train.txt','r'):
            u,i,r = [int(x) for x in line.split('::')[:3]]
            if u<limitu and i<limiti:
                ul.append(u)
                il.append(i)
                rl.append(r/5.0)
        ul,il = np.array(ul),np.array(il)
        R = sp.csr_matrix((rl,(ul,il)),shape=(N,M))
        return R,T,N,M,R_test,ul,il
    R,T,N,M,R_test,ul,il = getdata()
    lambdaU_,lambdaV_,lambdaT_,K_ = 1, 1, 0.5, 5
    U_,V_,rmse_,mae_ = RSTE(R,T,N,M,K_,lambdaU_,lambdaV_,lambdaT_,R,ul,il)
    job_server = pp.Server(6)
    jobs = []
    for lambdaU in [2,5]:
        for lambdaV in [2,5]:
            for lambdaT in [0.3,0.7]:
                jobs.append(job_server.submit(RSTE,(R,T,N,M,K_,lambdaU,lambdaV,lambdaT,R,ul,il),(),("numpy as np","import scipy.sparse as sp","time")))
    job_server.wait()
    for job in jobs:
        U,V,rmse1,mae1 = job()
        if mae1+rmse1<mae_+rmse_:
            mae_,rmse_ = mae1,rmse1
            lambdaU_ = lambdaU
            lambdaT_ = lambdaT
            lambdaV_ = lambdaV
    jobs = []
    for K in [3,7]:
        jobs.append(job_server.submit(RSTE,(R,T,N,M,K,lambdaU_,lambdaV_,lambdaT_,R_test,ul,il),(),("numpy as np","import scipy.sparse as sp","time")))
    job_server.wait()
    for job in jobs:
        U,V,rmse1,mae1 = job()
        if mae1+rmse1<mae_+rmse_:
            mae_,rmse_ = mae1,rmse1
            K_ = K
    print "=========all finish=============="
    print "K,lambdaU,lambdaV,lambdaT",K_,lambdaU_,lambdaV_,lambdaT_
    print "rmse-test",rmse_
    print "map-test",mae_

if __name__ == "__main__":
#   t_epinion()
   t_yelp(1000,20000)