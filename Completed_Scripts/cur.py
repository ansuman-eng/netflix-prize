import numpy as np
import random
import pickle
from time import time
import math
from pymf.cur import CUR

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

print "reconstrusted"
cur=np.matmul(np.matmul(cols,u),rows)
print cur

rec=a-cur
N=rr*cc
print ('RMSE :',math.sqrt(np.sum(rec**2)/N))
print ('Spearman :',1-6.0*np.sum(rec**2)/(N**3-N))

K=20
sum_of_scores=0
for user in range(len(a)):
	ordered_movies=np.argsort(cur[user])[::-1]
	i=0
	relevant=0

	while(i<len(ordered_movies) and i<K and cur[user][ordered_movies[i]]>3):
		movie_index=ordered_movies[i]
		i+=1
		if(a[user][movie_index]>3):
			relevant+=1

	if(i==0):
		sum_of_scores+=1
		continue

	relevant=float(relevant)
	relevant/=i
	sum_of_scores+=relevant

print("Precision on top K: ",sum_of_scores/len(a))

t2=time()
print t2
print t2-t1
# t2=time()
# print t2
# print t2-t1
# print cpc
# print cpr