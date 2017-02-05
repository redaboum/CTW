from binarytree import tree, pprint, convert, inspect
import numpy as np 



class CTW():
	def __init__(self, seq, context, d):
		self.seq = seq
		self.context = context
		self.l = []
		for i in range((2**d-1)):
			self.l.append([0,0,0,0])
		self.tree = convert(self.l)
		self.update_proba()
		self.high = 1
		self.low = 0	


	def proba(self,a,b):
		if a == 0:
			if b == 0:
				return 1
			else:
				return self.proba(a, b-1)*(b + 0.5 - 1)/(a + b)
		else:
			return self.proba(a-1, b)*(a + 0.5 - 1)/(a + b )


	def update_proba(self):
		ltemp = self.l
		for k in np.arange(len(ltemp)-1, -1, -1):
			if(2*(k+1) + 1 > len(ltemp)):
				ltemp[k][2] = self.proba(ltemp[k][0],ltemp[k][1])
				ltemp[k][3] = self.proba(ltemp[k][0],ltemp[k][1])
			else:
				ltemp[k][2] = self.proba(ltemp[k][0],ltemp[k][1])
				ltemp[k][3] = 0.5*ltemp[2*(k+1) - 1][3]*\
						ltemp[2*(k+1) +1 - 1][3] + 0.5*ltemp[k][2]

	def encode(self, elt, cont):
		point = self.tree
		for c in cont:
			if c == '0':
				point = point.right
			else:
				point = point.left
		p0 = point.value[0]




	def update(self):
		seq = list(self.seq)
		con = list(self.context)
		self.high = 1
		self.low = 0
		for elt in seq:

			#self.encode(elt, con)


			point = self.tree

			if elt == '0':
				point.value[0] += 1
			else:
				point.value[1] += 1

			for c in reversed(con):
				if c == '0':
					point = point.right
					if elt == '0':
						point.value[0] += 1
					else:
						point.value[1] += 1

					
				else:
					point = point.left
					if elt == '0':
						point.value[0] += 1
					else:
						point.value[1] += 1

			self.update_proba()
			con = con[1:]
			con.append(elt)

	def get_proba(self):
		self.update()
		return self.l[0][3]






# t = CTW("0110100", "010", 4)
# t.update()

# acc = 0
# for elt in t.l:
# 	if elt[2] == elt[3]:
# 		print elt
# 		acc += elt[2]

# print acc
