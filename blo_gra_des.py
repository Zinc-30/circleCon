#coding=utf8
'''
    solve the matrix factorization by block gradient descent, which can be applied to large scale datasets
'''
import time
import ctypes

import numpy as np
from scipy.sparse import csr_matrix as cm
from numpy.linalg import norm
from numpy.linalg import svd
from numpy import power

import matplotlib.pyplot as plt

def print_cost(func):
    def wrapper(*args, **kwargs):
        t = time.time()
        res = func(*args, **kwargs)
        print '%s: %.1fs' % (func.__name__, time.time() - t)
        return res
    return wrapper

class MF_BGD(object):

    def __init__(self, data=None, train_data=None, test_data=None, max_iter=200):
        self.K = 4
        self.lamb = 0.0
        self.eps = 10
        self.ite = max_iter
        self.tol = 1e-7
        self.train_ratio = 0.5
        self.filename = 'data/ml-1m-rating.txt'
        if data is None:
            self.load_data()
        else:
            self.data = data
            self.obs_num = len(self.data)
            self.train_data = train_data
            self.train_num = len(self.train_data)
            self.test_data = test_data
            self.test_num = len(self.test_data)
        self.load_lib()

    def load_lib(self):
        part_dot_lib = ctypes.cdll.LoadLibrary('./partXY_blas.so')
        set_val_lib = ctypes.cdll.LoadLibrary('./setVal.so')
        self.part_dot = part_dot_lib.partXY
        self.set_val = set_val_lib.setVal

    def split_data(self):
        rand_inds = np.random.permutation(self.obs_num)
        self.train_num = int(self.obs_num * self.train_ratio)

        self.train_data = self.data[rand_inds[:self.train_num]]
        self.test_data = self.data[rand_inds[self.train_num:]]
        self.test_num = len(self.test_data)
        del rand_inds

    def load_data(self):
        self.data = np.loadtxt(self.filename, dtype=np.float64)
        self.data[:,2] -= self.data[:,2].mean()
        self.data[:,2] /= self.data[:,2].std()
        self.obs_num = len(self.data)
        self.split_data()

    def get_obs_inds(self):
        return self.train_data[:,0].astype(int), self.train_data[:,1].astype(int)

    def part_uv(self, U, V, rows, cols, k):
        num = len(rows)
        output = np.zeros((num,1), dtype=np.float64)

        up = U.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        vp = V.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        op = output.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

        rsp = rows.ctypes.data_as(ctypes.POINTER(ctypes.c_long))
        csp = cols.ctypes.data_as(ctypes.POINTER(ctypes.c_long))

        nc = ctypes.c_int(num)
        rc = ctypes.c_int(k)
        self.part_dot(up, vp, rsp, csp, op, nc, rc)
        return output

    def p_omega(self, mat, rows, cols):
        mat_t = mat.copy()
        mat_t[rows, cols] = 0.0
        return mat - mat_t

    def cal_omega(self, omega, U, V, rows, cols, bias, obs):
        puv = self.part_uv(U, V, rows, cols, self.K)
        puv = obs - puv -  bias
        puvp = puv.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        odp = omega.data.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        nc = ctypes.c_int(self.train_num)
        self.set_val(puvp, odp, nc)

    def obj(self, U, V, omega):
        return 1.0 / 2 * power(norm(omega.data),2) + self.lamb / 2.0 * (power(norm(U,'fro'),2) + power(norm(V,'fro'),2))

    def train_rmse(self, U, V, bias, omega):
        return np.sqrt(power(norm(omega.data),2) / self.train_num)

    def get_grad(self, omega, U, V):
        du = -omega.dot(V) + self.lamb * U
        dv = -omega.T.dot(U) + self.lamb * V
        return du, dv

    def run(self):
        X = cm((self.data[:,2], (self.data[:,0], self.data[:,1]))) #index starting from 0
        M, N = X.shape
        omega = cm((self.train_data[:,2], (self.train_data[:,0], self.train_data[:,1])), shape=(M,N)) #index starting from 0
        trows, tcols = self.test_data[:,0].astype(np.int32), self.test_data[:,1].astype(np.int32)

        U = np.random.rand(M, self.K) * 0.001
        V = np.random.rand(N, self.K) * 0.001
        bias = 0# in reality, bias can also be updated, modified later
        eps_1 = eps_2 = self.eps

        rows, cols = omega.tocoo().row.astype(np.int32), omega.tocoo().col.astype(np.int32)
        obs = omega.copy().data.astype(np.float64).reshape(self.train_num, 1)
        self.cal_omega(omega, U, V, rows, cols, bias, obs)

        objs_1 = [self.obj(U, V, omega)]
        objs_2 = []
        trmses = []
        rmses, maes, costs, acu_cost = [], [], [], []

        run_start = time.time()
        for rnd in range(0, self.ite):
            start = time.time()
            self.cal_omega(omega, U, V, rows, cols, bias, obs)
            du, dv = self.get_grad(omega, U, V)
            l_omega = omega.copy()
            for t1 in range(0, 20):
                #line search
                LU = U - 1.0/eps_1 * du
                LV = V - 1.0/eps_1 * dv
                self.cal_omega(l_omega, LU, LV, rows, cols, bias, obs)
                l_obj = self.obj(LU, LV, l_omega)
                if l_obj < objs_1[rnd]:
                    U, V = LU, LV
                    eps_1 *= 0.95
                    objs_1.append(l_obj)
                    trmses.append(self.train_rmse(U,V,bias,l_omega))
                    break
                else:
                    eps_1 *= 1.5

            if t1 == 19:
                break

            lrate = (objs_1[rnd] - objs_1[rnd+1]) / objs_1[rnd]

            end = time.time()
            print 'iter=%s, obj=%.4f(%.2f%%), ls:((%.4f, %s), (%.4f, %s)), time:%.1fs\n' % (rnd, objs_1[rnd], lrate * 100, eps_1, t1, eps_1, t1, end-start)
            costs.append(round(end-start, 1))
            acu_cost.append(int(end-run_start))

            preds = self.part_uv(U, V, trows, tcols, self.K)
            rmses.append(self.cal_rmse(preds))
            maes.append(self.cal_mae(preds))
            print 'train_rmse=%.4f,rmse=%.4f, mae=%.4f\n' % (trmses[rnd], rmses[rnd], maes[rnd])

            if abs(lrate) < self.tol:
                break

            if objs_1[rnd] < self.tol:
                break

        #inds = range(1, len(objs_1)+1)
        #print 'objs_1', objs_1
        ##plt.plot(objs_1, label='obj')
        #print 'maes', maes
        ##plt.plot(maes, label='mae')
        #print 'rmses', rmses
        #inds = range(1, rnd+2)
        #l1, l2 = plt.plot(inds, rmses, 'r-', inds, trmses, 'g-', label='rmse')
        #l1.set_label('test')
        #l2.set_label('train')
        #plt.ylabel('RMSE')
        #plt.xlabel('iterations')
        #plt.title('movielens-1m')
        #plt.legend()
        #plt.show()
        #print 'costs',costs
        return objs_1, trmses, rmses, acu_cost

    def cal_rmse(self, preds):
        #user or item not occured in train dataset are set to 3 as default
        # beyond the 1,5 also need to set to 1 or 5
        delta = preds - self.test_data[:,2].reshape(preds.shape)
        rmse = np.sqrt(np.square(delta).sum() / self.test_num)
        return rmse

    def cal_mae(self, preds):
        delta = preds - self.test_data[:,2].reshape(preds.shape)
        mae = np.abs(delta).sum() / self.test_num
        return mae

#mf = MF_AGD()
#mf.run()
