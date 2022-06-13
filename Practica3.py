import numpy as np

P = np.array([[0.7,1.5,2.0,0.9,4.2,2.2,3.6,4.5],
	         [3,5,9,11,0,1,7,6]])

Z = np.vstack([P,[1,1,1,1,1,1,1,1]])

# Valores esperados
T = np.array([[-1,-1,-1,-1,1,1,1,1],
			  [-1,-1,1,1,-1,-1,1,1]])


# algoritmo Widrow-Hoff

R = np.dot(Z,Z.T)/8
H = np.dot(Z,T.T)/8
X = np.linalg.inv(R)@H
W = X[:2,:2].T
b = X[2,:].reshape(-1,1)

print(b)
