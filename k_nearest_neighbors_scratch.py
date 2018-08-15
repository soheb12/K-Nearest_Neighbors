# python3 k_nearest_lds.py

import numpy as np
from collections import Counter
import pandas as pd
import warnings
import random

def k_nearest_neighbors(data , predict , k=3):
    if(k <= len(data)):
        warnings.warn('K is set to a value less than total voting groups !')
    
    distances = []

    for group in data:
        for feature in data[group]:
            euclidean_dist = np.linalg.norm( np.array(feature) - np.array(predict) )
            distances.append( [euclidean_dist , group] )

    votes = [ i[1] for i in sorted(distances)[:k] ]
    vote_result = Counter(votes).most_common(1)[0][0]  # [ ('r' , 2) , ('k' , 1) ]

    return vote_result

def main():
    df = pd.read_csv("breast-cancer-dataset.txt")
    df.replace("?" , -99999 , inplace = True)
    df.drop( ['id'] , 1 , inplace = True )
    full_data = df.astype(float).values.tolist()
    random.shuffle(full_data)

    test_size = 0.2
    train_set = {2 : [] , 4 : []}
    test_set = {2 : [] , 4 : []}

    train_data = full_data[ :-int( test_size*len(full_data) ) ]
    test_data = full_data[ -int( test_size*len(full_data) ) : ]


    for i in train_data:
        train_set[ i[-1] ].append( i[:-1] ) # last collumn is class so that is the key and store till there
    for i in test_data:
        test_set[ i[-1] ].append( i[:-1] ) 


    correct = 0
    total = 1

    for group in test_set:
        for data in test_set[group]:
            vote = k_nearest_neighbors(test_set , data , k=5)
            if(vote == group):
                correct += 1
            total += 1

    print("Accuracy : " , correct/total)


if __name__ == "__main__":
    main()
