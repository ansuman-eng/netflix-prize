import pickle
from random import randint
import math
import numpy as np
from time import time
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


################MAIN FUNCTIONALITY###################

with open('ratings_matrix.pkl','rb') as f:
	ratings_matrix=pickle.load(f)
print("Matrix loaded")
t1=time()
#each row has one user
#each column has one movie
#create a copy matrix
temp_ratings=[]
for i in ratings_matrix:
	temp=[]
	for j in i:
		temp.append(j)
	temp_ratings.append(temp)


mean_normalize_movies(temp_ratings)	#mean_normalise the copy matrix, we need it on princinple
print("Mean normalised the copy matrix")

train_matrix=temp_ratings #now create a dummy matrix - train
test_train_split(train_matrix) #now do the test-train split

magnitude_normalize_movies(train_matrix)	#now magnitude normalize the train_matrix to find the similarity by simple matrix multiplication
print("Magnitude normalized the movie vectors")

train_matrix=np.array(train_matrix)
similarity_matrix=np.matmul(train_matrix.transpose(),train_matrix)		#the similarity matrix for movies
print("Movie-Movie Similarity matrix created")


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
	if(user%100==0):
		print(relevant,i, len(predicted_preferred[user]), user)
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

