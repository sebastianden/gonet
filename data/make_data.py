from sklearn.datasets import make_classification
import matplotlib.pyplot as plt
import numpy as np

X,y = make_classification(n_samples=1000, n_features=2, 
n_informative=2, n_redundant=0, n_classes=2, class_sep=2)


plt.scatter(X[:,0],X[:,1],c=y)
plt.show()

# Turn all 0's into -1's
y_save = y
y_save[np.where(y == 0)] = -1

# Stack X and y
X_save = np.hstack([X, y_save.reshape(-1,1)])

# Save to csv
np.savetxt("data1000.csv", X_save, fmt='%.10f',delimiter=',')

