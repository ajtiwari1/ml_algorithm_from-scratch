from sklearn.linear_model import LinearRegression 
import numpy as np

import matplotlib.pyplot as plt

x_train=np.array([1, 2, 3, 4], dtype=np.float32).reshape(-1,1)
y_train=np.array([2, 4, 6, 8], dtype=np.float32).reshape(-1,1)
x_test=np.array([10,15], dtype=np.float32).reshape(-1,1)

clf = LinearRegression()
clf.fit(x_train,y_train)
y_pred = clf.predict(x_test)
print(y_pred)


