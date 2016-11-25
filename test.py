from collections import defaultdict
import numpy as np
from scipy.sparse import csr_matrix as csr
from scipy.sparse import coo_matrix as coo
from scipy.sparse import coo_matrix as dok
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
print set(R.keys())&set([1])
print np.random.normal(size=(2,3))
x= defaultdict(dict)
a =[1,2,3,4]
print a[-4:],a[-2]
row = np.array([0, 0, 1, 2, 2, 5])
col = np.array([0, 2, 2, 0, 1, 5])
data = np.array([1, 2, 3, 4, 5, 6])
m2 = csr((data, (row, col)))
a= [1,2,3,4,5]
b = [1,2,3,4,5]
c = [1,2,3,4,5]
m1 = coo((a,(b,c)))
print m2.toarray()
print m1.toarray()
print m2.todok
x = np.array([[1],[2],[3],[4],[5],[6]]).dot(np.array([[1,2,3,4,5,6]]))
print x
kx = m2.todok().keys()
print kx
print "np",np.array(m2.todok().keys())
x1 = np.sum(kx,axis = 1)
print x1
print np.take(x,x1)
y = coo((np.take(x,x1),kx))
print y.toarray()
