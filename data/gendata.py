# import numpy as np
import operator
# lines = []
# for line in open('ratings.txt','r'):
# 	lines.append(line)
# n = len(lines)
# idlist = np.random.permutation(n)
# k = 0
# f1 = open('ratings-train.txt','w')
# f2 = open('ratings-test.txt','w')
# for i in idlist:
# 	if k<0.8*n:
# 		print >>f1, lines[i][:-1]
# 	else:
# 		print >>f2, lines[i][:-1]
# 	k += 1
# f1.close()
# f2.close()
# print n
ctable = dict()
d = dict()
d['1'] = 3;
d['2'] = 1;
print max(d, key=d.get)
cn = 0;
f3 = open('items-class.txt','w')
c = 
for line in open('items.txt','r'):
	tmps = line.split('::')
	x = tmps[0]
	c = tmps[1][:-1]
	for w in c.split(' '):
		w = w.rstrip().lstrip()
		if w in d:
			d[w] += 1
		else:
			d[w] = 1
	if c in ctable:
		cid = ctable[c]
	else:
		cn += 1
		ctable[c] = cn
		cid = cn
	print >>f3,x,cid
print cn
print len(d)
d = sorted(d.items(),key=operator.itemgetter(1),reverse=1)
for i in range(len(d)):
	if d[i][1]>10:
		print d[i][0]
f3.close()
			
