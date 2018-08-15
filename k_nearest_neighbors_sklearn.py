# python3 k_nearest.py

import numpy as np
from sklearn import preprocessing, model_selection, neighbors
import pandas as pd

df = pd.read_csv("breast-cancer-dataset.txt")
df.replace('?' , -9999 , inplace = True)
df.drop(['id'], 1 , inplace = True)

X = np.array( df.drop( ['class'],1 ) )
y = np.array( df['class'] )

X_train, X_test, y_train, y_test = model_selection.train_test_split(X , y , test_size=0.2)

clf = neighbors.KNeighborsClassifier()
clf.fit(X_train , y_train)
accuracy = clf.score(X_test , y_test)

print(accuracy)

example_data = np.array([[10,10,5,5,5,5,5,5,5]])
example_data = example_data.reshape(len(example_data) , -1)

print(clf.predict(example_data))
