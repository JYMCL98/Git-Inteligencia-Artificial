import numpy as np

class TanSig:
	def __call__(self,x):
		return np.tanh(x)

	def derivative(self,x,y):
		return 1.0 - np.square(y)

P = np.array([[0,0,1,1],
			  [0,1,0,1]])


T = np.array([-1,1,1,-1])
Q = 4

n1 = 20  # número de neuronas
ep = 1  # Factpr de escalamiento

W1 = ep*2*np.random.rand(n1,2)-1
b1 = ep*2*np.random.rand(n1,1)-1

W2 = ep*2*np.random.rand(1,n1)-1
b2 = ep*2*np.random.rand(1,1)-1

total_epocas = 100
a2 = np.array(np.zeros((1,Q)))
error_qua_medio = np.array(np.zeros((1,total_epocas)))
alfa = 0.001

for epocas in range(10):
	for q in range(Q):
		
		f = TanSig()
		a1 = f(np.dot(W1,P[:,q].reshape(-1,1))+b1)
		a2[:,q] = f(np.dot(W2,a1)+b2)

		# Retropropagacion de la sensibilidad
		e = T[q]-a2[:,q]
		print(e)
		s2 = -2*(1-(a2[,q]**2))*e
		s1 = (n.diag(1-(a1**2))*W2.T)*s2

		# Actualizacion de pesos sinapticos
		W2 = W2 - alfa*s2*a1.T
		b2 = b2 - alfa*s2
		W1 = W1 - alfa*s1*P[:,q].reshape(-1,1).T
		b1 = b1 - alfa*s1
		sum_error = e**2 + sum_error
		print(sum_error)
	error_qua_medio[epocas] = sum_error/Q

print(error_qua_medio) 


