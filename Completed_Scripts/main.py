import pickle
count=0

ratings_matrix=[]
for i in range(6040):
	temp=[]
	for j in range(3952):
		temp.append(0)
	ratings_matrix.append(temp)

with open("ratings.dat") as f:
	for line in f:
		count+=1
		if(count%10000==0):
			print(count)
			print(line)		

		user,movie,rating,info=line.split("::")
		user=int(user)-1
		movie=int(movie)-1
		rating=int(rating)
		ratings_matrix[user][movie]=rating

print("Loop over")
with open('ratings_matrix.pkl','wb') as f:
	pickle.dump(ratings_matrix,f)
print("Matrix serialised and dumped")
#offsetting by 1 to make 0 indexable
#users max = 6040
#movies max = 3952