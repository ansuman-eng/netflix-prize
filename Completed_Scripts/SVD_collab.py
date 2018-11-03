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
import matplotlib.pyplot as plt
from time import time

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
def magnitude_normalize_movies(ratings_matrix):

	for movie in range(len(ratings_matrix[0])):
		sum_movie=0
		for user in range(len(ratings_matrix)):
			sum_movie=sum_movie+((ratings_matrix[user][movie])**2)	
			
		sum_movie=math.sqrt(sum_movie)
		if(sum_movie==0):
			continue
		else:
			for user in range(len(ratings_matrix)):
				ratings_matrix[user][movie]=float(ratings_matrix[user][movie])/sum_movie
def mean_normalize_movies(ratings_matrix):	

	for movie in range(len(ratings_matrix[0])):
		mean_movie=0
		non_zero=0
		for user in range(len(ratings_matrix)):
			if(ratings_matrix[user][movie]!=0):
				mean_movie+=ratings_matrix[user][movie]
				non_zero+=1

		non_zero=float(non_zero)
		if(non_zero==0):
			continue
		mean_movie=mean_movie/non_zero

		for user in range(len(ratings_matrix)):
			if(ratings_matrix[user][movie]!=0):
				ratings_matrix[user][movie]-=mean_movie
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


def test_train_split(train_matrix):
	movie_count=len(train_matrix[0])/3
	user_count=len(train_matrix)/3
	''' the first one fourth of rows, and first one fourth of columns will be treated as
	test set'''
	user=0
	while(user<user_count):
		movie=0
		while(movie<movie_count):
			train_matrix[user][movie]=0
			movie+=1
		user+=1


################MAIN FUNCTIONALITY###################

with open('ratings_matrix.pkl','rb') as f:
	ratings_matrix=pickle.load(f)
print("Matrix loaded")


'''
ratings_matrix=[]

for i in range(600):
	temp=[]
	for j in range(400):
		temp.append(randint(0,5))
	ratings_matrix.append(temp)
#Print initial ratings matrix
'''
'''
for i in ratings_matrix:
	print(i)
print
'''

#each row has one user
#each column has one movie
#mean_normalise the matrix, we need it on princinple
t1=time()
print(t1)
U,s,VT=SVD(ratings_matrix)
print("SVD Done")
t2=time()
print(t2)
ratings_matrix=np.array(ratings_matrix)
movie_to_concept=np.matmul(ratings_matrix.transpose(),U)
movie_to_concept=movie_to_concept.transpose()
magnitude_normalize_movies(movie_to_concept)
similarity_matrix=np.matmul(movie_to_concept.transpose(),movie_to_concept)
print("Movie-Movie Similarity matrix created")

#print the similarity matrix for movies
'''
print("SVD similarity")
for i in similarity_matrix:
	print(i)
print
'''

x_axis=[]
rmse=[]
spear=[]
prec_on_top_k=[]


neigbour_list=[50]
squared_error=0
number_of_test=0

predicted_preferred=[]
for neigbours in neigbour_list:
	print("Considering closest neigbours",neigbours)
	
	for test_user in range(0,len(ratings_matrix)/3):
		preferred_movies=[]
		for test_movie in range(0,len(ratings_matrix[0])/3):
			
			if(ratings_matrix[test_user][test_movie]!=0):
				#Choose to calculate only those ratings inside test set which were there before just to find
				#error				
				
				
				num_score=0
				den_score=0

				ordered_neigbours=np.argsort(similarity_matrix[test_movie])[::-1]
				count=0
				for closest_movie in ordered_neigbours:
					if(closest_movie<len(ratings_matrix[0])/3):
						continue
					#The above if statement ensures that the test-movie isn't compared to any other test-movie

					score=similarity_matrix[test_movie][closest_movie]
					if(ratings_matrix[test_user][closest_movie]!=0):				#Proceed iff the user has a rating for the closest movie
						#print("closest",closest_movie)
						den_score+=score
						num_score+=(score*ratings_matrix[test_user][closest_movie])
						count+=1

					if(count==neigbours):
						break				

				#print(num_score)
				#print(den_score)
				if(den_score==0):
					num_score=0 			#if by a very small chance, den_score becomes 0, we set num_score to be the average
				else:
					num_score=(num_score/den_score)
				preferred_movies.append((num_score,test_movie))
				squared_error+=((num_score-ratings_matrix[test_user][test_movie])**2)
				number_of_test+=1
		predicted_preferred.append(preferred_movies)
print(len(predicted_preferred))
K=20
sum_of_scores=0
for user in range(len(predicted_preferred)):
	predicted_preferred[user].sort()
	predicted_preferred[user]=predicted_preferred[user][::-1]

	i=0
	relevant=0
	print(user,len(predicted_preferred[user]))
	if(len(predicted_preferred[user])==0):
		sum_of_scores+=1
		continue

	while(i<len(predicted_preferred[user]) and predicted_preferred[user][i][0]>3 and i<K ):
		movie_index=predicted_preferred[user][i][1]
		i+=1
		if(ratings_matrix[user][movie_index]>3):
			relevant+=1
			
	if(i==0):
		sum_of_scores+=1
		continue

	relevant=float(relevant)
	relevant/=i
	sum_of_scores+=relevant

print("Precision on top K: ",sum_of_scores/len(predicted_preferred))

t2=time()
print(t2-t1)
print(squared_error)
print(number_of_test)
print(neigbours)
print("RMSE :", math.sqrt(squared_error/number_of_test))
spearman=1- 6.0*(squared_error)/((number_of_test**3)-number_of_test)
print("Spearman Rank score:", spearman )

x_axis.append(neigbours)
rmse.append(math.sqrt(squared_error/number_of_test))
spear.append(spearman)

