import numpy as np
from random import randint
import random
import pickle
from numpy import array
from numpy import diag
from numpy import dot
from numpy import zeros
from random import randint
from scipy.linalg import svd
from numpy import linalg as LA
import math
from time import time

def mean_normalize_movies(ratings_matrix):	
	for movie in range(len(ratings_matrix[0])):
		#print(movie)
		mean_movie=0
		non_zero=0
		for user in range(len(ratings_matrix)):
			if(ratings_matrix[user][movie]!=0):
				mean_movie+=ratings_matrix[user][movie]
				non_zero+=1
		#print(mean_movie)
		#print(non_zero)
		non_zero=float(non_zero)
		if(non_zero==0):
			continue
		mean_movie=mean_movie/non_zero

		for user in range(len(ratings_matrix)):
			if(ratings_matrix[user][movie]!=0):
				ratings_matrix[user][movie]-=mean_movie

def magnitude_normalize_movies(ratings_matrix):

	for movie in range(len(ratings_matrix[0])):
		sum_movie=0
		for user in range(len(ratings_matrix)):
			sum_movie=sum_movie+((ratings_matrix[user][movie])**2)
		#print(mean_movie)
		#print(non_zero)
		sum_movie=math.sqrt(sum_movie)
		if(sum_movie==0):
			continue
		else:
			for user in range(len(ratings_matrix)):
				ratings_matrix[user][movie]=float(ratings_matrix[user][movie])/sum_movie
	
def test_train_split(train_matrix):
	movie_count=len(train_matrix[0])/3
	user_count=len(train_matrix)/3
	''' the first one third of rows, and first one third of columns will be treated as
	test set'''
	user=0
	while(user<user_count):
		movie=0
		while(movie<movie_count):
			train_matrix[user][movie]=0
			movie+=1
		user+=1

def debug(U,s,VT,sigma):
	print
	print("Left singular vectors are cols\n")
	print(U)
	print
	print("Right singular vectors are rows\n")
	print(VT)
	print
	print("Sigma matrix from the SVD function\n")
	print(diag(s))
	print
	print("Proper sigma matrix\n")
	print(sigma)
	print
	print("Reconstruction without any dimensionality reduction")
	print(np.matmul(U,np.matmul(sigma,VT)))

def energy_fraction(s,frac):
	total_energy=0
	for i in range(len(s)):
		total_energy+=((s[i])*(s[i]))

	collected_energy=0
	req=0
	while(req<len(s)):
		collected_energy+=((s[req])*(s[req]))
		if(collected_energy>(frac*total_energy)):
			break
		req+=1

	req+=1
	return req

def SVD(A):
	A=np.array(A)
	AT=A.transpose()

	cov_1=np.matmul(A,AT)		#cov_1 is A*AT
	cov_2=np.matmul(AT,A)		#cov_2 is AT*A
	w,U_temp=LA.eigh(cov_1)		#w will contain the eigenvalues, U_temp will contain eigenvectors of cov1
	w1,V_temp=LA.eigh(cov_2)     #w will contain the eigenvalues, V_temp will contain eigenvectors of cov1


	
	''' Here begins the calculation for VT '''	
	#print(V_temp)
	#print(w1)

	eigen_to_V_temp={}
	for j in range(len(w1)):
		if(w1[j]>0):
			eigen_to_V_temp[math.sqrt(w1[j])]=V_temp[:,j]
	V=[]
	for key in sorted(eigen_to_V_temp):
		V.append(eigen_to_V_temp[key])
	V=V[::-1]
	V=np.array(V)
	V=V.transpose()
	VT=V.transpose()
	#print(VT)

	''' Here begins the calculation for s'''
	
	s=[]
	w1.sort()					#need to get the eigenvalues in descending order
	w1=w1[::-1]
	for j in range(len(w1)):		#append square roots of positive eigenvalues only
		if(w1[j]>0):
			s.append(math.sqrt(w1[j]))
	
	#print(diag(s))				#diagonalise the list into a matrix

	''' Here begins the calculation for U '''
	
	#print(U_temp)
	#print(w)
	S=diag(s)
	U=np.matmul(A,np.matmul(V, LA.inv(S)))
	#print(U)


	return U,s,VT


################# MAIN SVD FUNCTIONALITY ###################

with open('ratings_matrix.pkl','rb') as f:
	ratings_matrix=pickle.load(f)
print("Matrix loaded")


#Print initial ratings matrix
'''
for i in ratings_matrix:
	print(i)
print
'''
#each row has one user
#each column has one movie
#first mean_normalise the matrix, we need it on princinple

N=len(ratings_matrix)*len(ratings_matrix[0])
threshold=3

#U,s,VT = svd(ratings_matrix,full_matrices=False)
t1=time()
U,s,VT= SVD(ratings_matrix)
print("SVD Calculated")
t2=time()
print("SVD time",t2-t1)

reconstructed=np.matmul(U,np.matmul(diag(s),VT))
diff=ratings_matrix-reconstructed
diff=(diff**2)
sum1=0
for i in diff:
	sum1=sum(i)

print("Reconstruction Error is ", sum1)
print("RMSE is :", math.sqrt(float(sum1)/N))
print("Spearman is : ", 1-6.0*sum1/(N**3-N))
t3=time()
print("Normal SVD",t3-t1)


total_energy=0
frac_energy=0
for i in range(len(s)):
	total_energy+=(s[i]*s[i])

i=0
while(i<len(s)):
	frac_energy+=(s[i]*s[i])
	i+=1
	if(frac_energy>=(0.9*total_energy)):
		break
re_U=U[:,:i]
re_s=s[:i]
re_VT=VT[:i,:]
reconstructed=np.matmul(re_U,np.matmul(diag(re_s),re_VT))
diff=ratings_matrix-reconstructed
diff=(diff**2)
sum1=0
for i in diff:
	sum1=sum(i)
print("Reconstruction Error with 90_percent energy is ", sum1)

print("Reconstruction Error is ", sum1)
print("RMSE is :", math.sqrt(float(sum1)/N))
print("Spearman is : ", 1-6.0*sum1/(N**3-N))
t4=time()
print("90_percent SVD", (t4-t3)+(t2-t1))

K=20
sum_of_scores=0
for user in range(len(ratings_matrix)):
	ordered_movies=np.argsort(reconstructed[user])[::-1]
	i=0
	relevant=0

	while(i<len(ordered_movies) and i<K and reconstructed[user][ordered_movies[i]]>3):
		movie_index=ordered_movies[i]
		i+=1
		if(ratings_matrix[user][movie_index]>3):
			relevant+=1

	if(i==0):
		sum_of_scores+=1
		continue

	relevant=float(relevant)
	relevant/=i
	if(user%100==0):
		print(relevant,i, user)
	sum_of_scores+=relevant

print("Precision on top K: ",sum_of_scores/len(ratings_matrix))
