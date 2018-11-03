import numpy as np
import random
import pickle
from time import time
import math

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

with open ('ratings_matrix.pkl','rb') as f:
	a=pickle.load(f)
a=np.array(a)
rr=len(a)
cc=len(a[0])
# tt,yy,hh=np.linalg.svd(a,full_matrices=False)
c=3734

t1=time()
print t1
#input starts
# a=np.array([[1,2,4,7,0],
# 			[3,3,3,6,4],
# 			[4,4,4,2,0],
# 			[5,5,5,1,0],
# 			[3,3,3,6,4],
# 			[3,2,2,3,5],
# 			[4,4,4,2,0]])
# rr=7
# cc=5
# c=2
#input ends
# t1=time()
# print t1
aa=a**2
cols=np.empty(shape=(rr,c))
rows=np.empty(shape=(c,cc))
ri=np.empty(shape=(0,0))
ci=np.empty(shape=(0,0))
pr=np.sum(aa,axis=1)
pc=np.sum(aa,axis=0)
pr=pr.astype(float)
pc=pc.astype(float)
pr=pr/np.sum(aa)
pc=pc/np.sum(aa)

cpr=np.cumsum(pr)
cpc=np.cumsum(pc)
pr=np.sqrt(c*pr)
pc=np.sqrt(c*pc)


rows=rows.astype(float)
cols=cols.astype(float)
a=a.astype(float)


for i in range(c):
	rn=np.random.uniform(0,1)
	for j in range(cc):
		if(cpc[j]>rn):
			ci=np.append(ci,j)
			# if i>0:
			# 	cols=np.hstack((cols,a[:,j:j+1]))
			# else:
			# 	cols=a[:,j:j+1]
			break
	rn=np.random.uniform(0,1)
	for k in range(rr):
		if(cpr[k]>rn):
			ri=np.append(ri,k)
			# if i>0:
			# 	rows=np.vstack((rows,a[k:k+1,:]))
			# else:
			# 	rows=a[k:k+1,:]
			break

# ri=np.array([2,1])
# ci=np.array([3,3])

# ci=np.delete(ci,0)
# ri=np.delete(ri,0)
ci=ci.astype(int)
ri=ri.astype(int)
# ci=np.sort(ci)
# ri=np.sort(ri)


k=0
for j in range(c):
	if k > 0:
		rows=np.vstack((rows,a[ri[j]:ri[j]+1,:]))
		cols=np.hstack((cols,a[:,ci[j]:ci[j]+1]))
	else:
		cols=a[:,ci[j]:ci[j]+1]
		rows=a[ri[j]:ri[j]+1,:]
		k=k+1

rows=rows.astype(float)
cols=cols.astype(float)
w=np.zeros(shape=(c,c))
for i in range(c):
	for j in range(c):
		w[i][j]=a[ri[i]][ci[j]]
# w=w.astype(int)

print "Rows and cols before scale"
print rows
print cols
for i in range(c):
	rows[i,:]=rows[i,:]/pr[ri[i]]
	cols[:,i]=cols[:,i]/pc[ci[i]]

# k=2
# for i in range(1,c):
# 	if ri[i]==ri[i-1]:
# 		k=k+1
# 		rows[i,:]=rows[i,:]/np.sqrt(k)
# 		w[i,:]=w[i,:]/np.sqrt(k)
# 	else:
# 		k=2
# k=2
# for i in range(1,c):
# 	if ci[i]==ci[i-1]:
# 		k=k+1
# 		cols[:,i]=cols[:,i]/np.sqrt(k)
# 		w[i,:]=w[i,:]/np.sqrt(k)
# 	else:
# 		k=2



# k=2
# for i in range(c):
# 	for j in range(i+1,c):
# 		if np.array_equal(rows[i,:],rows[j,:]):
# 			k=k+1
# 			rows[j,:]=rows[j,:]/np.sqrt(k)
# 			w[j,:]=w[j,:]/np.sqrt(k)
# 		else:
# 			k=2
# k=2
# for i in range(c):
# 	for j in range(i+1,c):
# 		if np.array_equal(cols[:,i],cols[:,j]):
# 			k=k+1
# 			cols[:,j]=cols[:,j]/np.sqrt(k)
# 			w[j,:]=w[j,:]/np.sqrt(k)
# 		else:
# 			k=2




# print "w"
# print w


aa,bb,vv=np.linalg.svd(w,full_matrices=False)

# ss=0.0
# for i in bb:
# 	ss+=i*i
# kk=0.0
# ll=0
# for i in bb:
# 	kk+=i*i
# 	if(kk/ss>=0.9):
# 		ll=i
# 		break

# bb=bb[0:ll]
# aa=aa[:,0:ll]
# vv=vv[:,0:ll]
bb=np.diag(bb)
for i in range(c):
	if (bb[i][i]>0.01 or bb[i][i]<-0.01):
		bb[i][i]=1/bb[i][i]
aa=np.transpose(aa)
vv=np.transpose(vv)
bb=np.matmul(bb,bb)
u=np.matmul(np.matmul(vv,bb),aa)




# rs,rv,rd=np.linalg.svd(rows,full_matrices=False)
# rvs=0
# print rv
# rvs=np.size(rv)
# rv=np.diag(rv)
# for i in range(rvs):
# 	if rv[i][i]!=0:
# 		rv[i][i]=1/rv[i][i]
# rs=np.transpose(rs)
# rd=np.transpose(rd)
# # rv=np.matmul(rv,rv)
# rowss=np.matmul(np.matmul(rd,rv),rs)

# cs,cv,cd=np.linalg.svd(cols,full_matrices=False)
# cvs=np.size(cv)
# cv=np.diag(cv)
# for i in range(cvs):
# 	if cv[i][i]!=0:
# 		cv[i][i]=1/cv[i][i]
# cs=np.transpose(cs)
# cd=np.transpose(cd)
# # cv=np.matmul(cv,cv)
# colss=np.matmul(np.matmul(cd,cv),cs)
# u=np.matmul(np.matmul(colss,a),rowss)




# print bb
print "CUR"
print cols
print u
print rows

U=cols
s=u
VT=rows

ratings_matrix=a
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



t2=time()
print t2
print t2-t1
# t2=time()
# print t2
# print t2-t1
# print cpc
# print cpr