import numpy as np
from ctw import CTW
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


np.random.seed(0)

N = 40
s1 = [0.5,0.5,0.5, 0.5]
s2 = [0.1,0.1, 0.1, 0.1]

s1 = np.array(s1)
s2 = np.array(s2)

def generate(s, n, context):
	seq = ""
	for i in range(n):
		k = int(context,2)
		u = np.random.rand()
		if u < s[k]:
			seq+= "1"
		else:
			seq+="0"
		context = context[1:]
		context += seq[-1]
	return seq


X = []
Y = []


## Les paires appartiennent au meme cluster, et les impaires au meme
for i in range(N/2):
	X.append(generate(s1, 10, "00"))
	X.append(generate(s2, 10, "00"))

P = []

for i in range(N):
	t = CTW(X[i], "00", 3)
	P.append(t.get_proba())

Z = np.zeros((N,N))
P = np.array(P)

for i in range(N):
	for j in range(N):
		t = CTW(X[i] + X[j], "00", 3)
		p = t.get_proba()
		Z[i,j] = p/(P[i]*P[j])

X = np.array(X)


## K moyenne algorithme

per = np.random.permutation(N)
c1 = P[per[:N/2]]
c2 = P[per[N/2:]]

for i in range(2):
	m1 = np.argmin((c1 - c1.mean())**2)
	m2 = np.argmin((c2 - c2.mean())**2)
	temp = np.array([Z[m1],Z[m2]])
	temp = np.argmin(temp, axis=0)
	temp1 = np.argwhere(temp== 0).flatten()
	temp2 = np.argwhere(temp == 1).flatten()
	if (len(temp1) == 0 or len(temp2) == 0):
		break
	c1 = P[temp1]
	c2 = P[temp2]

print "cluster 1 : " + str(temp1)
print "cluster 2 : " + str(temp2)
print P

for i in range(len(P)):
	if(P[i] > 0.1):
		continue
	if i %2 == 0:
		plt.scatter(1000*P[i], 1000*P[i], color='b')
	else:
		plt.scatter(1000*P[i], 1000*P[i], color='r')

plt.show()










