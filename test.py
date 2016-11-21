from collections import defaultdict
import numpy as np
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
print np.random.normal(size=2)