1) The completed scripts are under Completed_Scripts Directory.
2) The raw data is under Data directory.
3) The picked ratings matrix, i.e. ratings_matrix.py is inside the Completed_Scripts.
4) These are the scripts corresponding to the techniques

Dataset: From MovieLens, with ratings from 6000 users for 4000 movies.
Results: Can be found in Design Document. Refer to it.

collaborative_new.py - Collaborative Filtering
collaborative_baseline.py - Collaborative Filtering with baseline approach
main.py - COnvert raw data into ratings_matrix.pkl
SVD.py - SVD Reconstruction with all energy, and 90% energy
SVD_collab.py - SVD + Collaborative filtering
cur.py - CUR with total energy
c90.py - CUR with 90% energy

To execute any python script, just fire up the terminal and execute
python script_name

Dependencies = numpy, time, pickle, random, math, 
