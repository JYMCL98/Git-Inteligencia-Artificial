# Universidad Autónoma Chapingo
# Departamento de Ingeniería Mecánica Agrícola
# Ingeniería Mecatrónica Agrícola
# Jym Emmanuel Cocotle Lara
# 7° 7

# Librerí­as
# Para vectores y matrices
import numpy as np
# Para graficación
import matplotlib.pyplot as plt

# Función hardlim
def hardlim(n):
	if n[0][0] > 0:
		value1 = 1
	else:
		value1 = -1
	if n[1][0] > 0:
		value2 = 1
	else:
		value2 = -1
	return [[value1],[value2]]

# Entradas
P = np.array([[0.7,1.5,2.0,0.9,4.2,2.2,3.6,4.5],
			  [3,5,9,11,0,1,7,6]])

# Vector aumentado
Z = np.vstack([P,[1,1,1,1,1,1,1,1]])

# Valores esperados : cambia 0 a -1
T = np.array([[-1,-1,-1,-1,1,1,1,1],
			  [-1,-1,1,1,-1,-1,1,1]])

# Algotimo de Widrow-Hoff
R = np.dot(Z,Z.T)/8 # Q=8
H = np.dot(Z,T.T)/8
X = np.linalg.inv(R) @ H
W = X[:2,:2].T
b = X[2:].reshape(-1,1)

# Verificar la solución con el algoritmo perceptrón
e = np.array(np.ones((2,8)))

for q in range(8):
   # Conversión del vector a un vector de 1 columna
	a = hardlim(np.dot(W,P[:,q].reshape(-1,1))+b)
	e[:,q] = (T[:,q].reshape(-1,1)-a).T
	
print(e)

# Graficación 
fig,ax = plt.subplots()

# Ligeros y poco usados
ax.scatter(P[0][0],P[1][0],marker='^')
ax.scatter(P[0][1],P[1][1],marker='^')

#Ligeros y muy usados
ax.scatter(P[0][2],P[1][2],marker='s')
ax.scatter(P[0][3],P[1][3],marker='s')

#Pesados y poco usados
ax.scatter(P[0][4],P[1][4],marker='o')
ax.scatter(P[0][5],P[1][5],marker='o')

#Pesados y muy usados
ax.scatter(P[0][6],P[1][6],marker='*')
ax.scatter(P[0][7],P[1][7],marker='*')

points = np.arange(0,6,0.01)

# Primera neurona
ax.plot(points, (-b[0][0]/W[0,1])-(W[0,0]/W[0,1])*points)
ax.plot(points, (-b[1][0]/W[1,1])-(W[1,0]/W[1,1])*points)
ax.set_xlim([0,6])
ax.set_ylim([-2,14])
plt.show()
