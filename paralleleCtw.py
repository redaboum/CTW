import numpy as np 
from mpi4py import MPI
import time

t1 = time.time()

N = 100000
d = 2


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

if rank == 0:
	t1 = time.time()

########################################################################
################## Generation des donnees ##############################
########################################################################


if (rank == 0):
	s = ""
	for i in range(N):
		u = np.random.rand();
		if(u < 1./2):
			s += "0"
		else:
			s += "1"
else:
	s = None

s = comm.bcast(s, root=0)
# s = "0100110100"
tree = np.array([0]*(2**d - 1))


########################################################################
################# Parallelisation de l'etape 1, generation de etap #####
########################################################################


nb = 0
for i in range(d + int(np.floor(rank*(N - d)/size)), d + int(np.floor((rank+1)*(N-d)/size))):
	context = s[i-(d-1):i]
	j = 0
	c = len(context) - 1
	nb += 1
	while j < len(tree):
		tree[j] += 1
		if(context[c] == '0'):
			j = 2*(j + 1) - 1
		else:
			j = 2*(j+1)
		if(j > 0):
			c -= 1
tree = comm.allreduce(tree, MPI.SUM)


########################################################################
################# Parallelisation de l'etape 2,  #######################
########################################################################

omega = {}

prof = np.array([0]*(2**d - 1))

def fillProf(l, k):
	if k >= len(l):
		return
	l[k] = 1+l[(k-1)/2]
	fillProf(l, 2*(k+1) - 1)
	fillProf(l, 2*(k+1))

if rank == 0:
	for r in range(size):
		if r < 3:
			omega[r] = r
		else:
			omega[r] = -1
	fillProf(prof, 0)

# Communication bloquante, on attend les infos de la part de root
prof  = comm.bcast(prof, root = 0)
omega = comm.bcast(omega, root = 0)


w = 0,rank  # we add the rank to reduce on max first element 
			# and get the location of the max


nbused = 3
wmax = np.inf
wmaxp = np.inf
omegap = dict(omega)



while nbused < size and wmaxp >= wmax:

	### update de wmax precedent en le nouveau wmax et omegap en omega
	wmaxp = wmax
	omegap = dict(omega)

	### calcul du nouveau travail minimum en dehors de la racine
	if omega[rank] > 0:
		w = tree[omega[rank]]*(d + 1 - prof[omega[rank]]), rank
	wtot = comm.allreduce(w[0], MPI.SUM)
	res = comm.allreduce(w, MPI.MAX)

	### Calcul du travail maximal, max des travail leaf et racine
	wmax = max(res[0], tree[0]*d - wtot)

	### Actualisation de omega, la repartition des processeurs
	if rank == res[1]:
		temp = omega[rank]
		omega[rank] = 2*(temp + 1) - 1
		omega[nbused] = 2*(temp + 1) 

	### Augmenter le nombre de processeur utilise et communiquer omega
	nbused += 1
	omega = comm.bcast(omega, root = res[1])


if( wmaxp < wmax):
	omega = omegap
# if rank == 0:
# 	print omega

########################################################################
################# Parallelisation de l'etape 3  ########################
#################    Finite state machine       ########################
########################################################################


# Calcul de la fonction inverse de omega

ind = omega[rank], rank

ind = comm.allgather(ind)
omegainv = dict(ind)


intToString = [""]*(2**d - 1)
if rank == 0:
	i = 1
	while i < len(intToString):
		if i%2 == 0:
			intToString[i] = "1" + intToString[(i-1)/2] 
		else:
			intToString[i] = "0" + intToString[i/2] 
		i += 1
intToString = comm.bcast(intToString, root = 0)

StringToInt = {}

for i in range(int(np.floor(rank*(2**d - 1)/size)), int(np.floor((rank+1)*(2**d - 1)/size))):
	StringToInt[intToString[i]] = i

	
StringToInt = comm.allgather(StringToInt)
StringToInt = reduce(lambda a, b: dict(a, **b), StringToInt)


# Pas de condition d'arret, on est sur de trouver au moins la racine
def find_pref(context):
	try:
		return omegainv[StringToInt[context]]
	except:
		return find_pref(context[1:])



########################################################################
################# Parallelisation de l'etape 4  ########################
################# Lire les donnees et partager  ########################
#################    avec le bon processeur     ########################
########################################################################


# Parcourir les donnees, trouver les processeurs a qui envoyer
# les elements avec le context restant et faire le travail
#

t3 = time.time()

work = {}
for i in range((d-1) + int(np.floor(rank*(N - (d-1))/size)), (d-1) + int(np.floor((rank+1)*(N-(d-1))/size))):
	context = s[i-(d-1):i]
	r = find_pref(context)
	try:
		work[r].append((s[i], context))
	except:
		work[r] = [((s[i], context))]


data = []
keys = list(work.keys())
keys = comm.allgather(keys)


for k in range(size):
	if rank == k:
		for dest in keys[k]:
			if dest != rank:
				comm.send(work[dest], dest, tag = k)
	if rank != k:
		if rank in keys[k]:
			data.append(comm.recv(source=k, tag =k))




if rank != 0:
	if rank in keys[rank]:
		data.append(work[rank])

t4 = time.time()


########################################################################
################# Parallelisation de l'etape 5  ########################
#################   Getting our hands dirty     ########################
#################   Implementer l'algorithme    ########################
########################################################################


myContext = intToString[omega[rank]]
myDepth = d-len(myContext)
myTree = []


for i in range((2**d-1)):
	myTree.append([0.,0.,0.,0.]) #nbre 0, nbre 1, pc, pe


def find_suff(s, context):
	return s[:-len(context)]

def is_leaf(k):
	return 2*(k+1) >= len(myTree)

# Get list of index for subtree of this processor

list_index = [omega[rank]]*(2**myDepth - 1)

def fill_list_index(u):
	if 2*(u+1)>=len(list_index):
		return
	temp = list_index[u]
	list_index[2*(u+1) - 1] = 2*(temp+1) - 1
	list_index[2*(u+1)] = 2*(temp + 1) 
	fill_list_index(2*(u+1) - 1)
	fill_list_index(2*(u+1))

fill_list_index(0)

# Calcul la probabilite empirique par l'estimateur KT
def proba(a,b):
	if a == 0:
		if b == 0:
			return 1
		else:
			return proba(a, b-1)*(b + 0.5 - 1)/(a + b)
	else:
		return proba(a-1, b)*(a + 0.5 - 1)/(a + b )

def proba_bis(a, b, as1, bs1, p):
	acc = 1
	while a > as1:
		acc *= (a + 0.5 - 1)/(a + b)
		a -= 1
	while b > bs1:
		acc *= (b + 0.5 - 1)/(a + b)
		b -=1
	return acc*p


## Le noeud racine n'a aucune data, attraper cettre erreur !!!
# data =  [da for z in data for da in z]
# if rank == 1:
# 	print "blabla"
# 	print data
# comm.barrier()
# if rank == 2:
# 	print data
# comm.barrier()
# if rank == 3:
# 	print "blabla"
# 	print data
# comm.barrier()
# if rank == 4:
# 	print data

if rank != 0:
	data =  [da for z in data for da in z]


try:
	for da in data:
		elt, context = da
		context = find_suff(context, myContext)

		# Update numbers of 0, numbers of 1
		begin = omega[rank]

		while begin < len(myTree):
			if elt == "0":
				myTree[begin][0] += 1
			else:
				myTree[begin][1] += 1
			if context == "":
				break
			if context[-1] == "0":
				begin = 2*(begin + 1) - 1
			else:
				begin = 2*(begin + 1)


		# Update probability pc and pe
		i = len(list_index) - 1
		begin = list_index[i]
		while i >= 0:
			if(is_leaf(begin)):
				myTree[begin][2] = proba(myTree[begin][0],myTree[begin][1])
				myTree[begin][3] = proba(myTree[begin][0],myTree[begin][1])
			else:
				myTree[begin][2] = proba(myTree[begin][0],myTree[begin][1])
				myTree[begin][3] = 0.5*myTree[2*(begin+1) - 1][3]*\
							myTree[2*(begin+1) +1 - 1][3] + 0.5*myTree[begin][2]
			i -= 1
			begin = list_index[i]

except:
	data = None


myTree = np.array(myTree)
myTree = comm.reduce(myTree, MPI.SUM)



def fill_tree(k):
	### Fill first the children !!
	if(np.all(myTree[2*(k+1) - 1] == np.array([0.,0.,0.,0.]))):
		fill_tree(2*(k+1) - 1)

	if (np.all(myTree[2*(k+1)] == np.array([0.,0.,0.,0.]))):
		fill_tree(2*(k+1))

	### Fill now the father, numbers then probability
	myTree[k][0] = myTree[2*(k+1) - 1][0] + myTree[2*(k+1)][0] 
	myTree[k][1] = myTree[2*(k+1) - 1][1] + myTree[2*(k+1)][1]
	as1,bs1,p1 = myTree[2*(k+1) - 1][0], myTree[2*(k+1) - 1][1]\
							, myTree[2*(k+1) - 1][3]
	as2,bs2,p2 = myTree[2*(k+1)][0], myTree[2*(k+1)][1]\
							, myTree[2*(k+1)][3]

	if (as1 + bs1 > as2 + bs2):
		myTree[k][2] = proba_bis(myTree[k][0],myTree[k][1], as2,bs2,p2)
	else:
		myTree[k][2] = proba_bis(myTree[k][0],myTree[k][1], as1,bs1,p1)

	myTree[k][3] = 0.5*myTree[2*(k+1) - 1][3]*\
							myTree[2*(k+1) +1 - 1][3] + 0.5*myTree[k][2]

if rank == 0:
	fill_tree(0)
	# print myTree
	t2 = time.time()
	print t2 - t1
	print t4 - t3





