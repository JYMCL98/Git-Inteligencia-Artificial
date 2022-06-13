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

# Función de activación
def hardlim(n):
	if n[0][0] > 0:
		value1 = 1
	else:
		value1 = 0
	
	if n[1][0] > 0:
		value2 = 1
	else:
		value2 = 0

	return [[value1],[value2]]


# Arreglo de entradas
P = np.array([[0.7,1.5,2.0,0.9,4.2,2.2,3.6,4.5], # peso
			  [3,5,9,11,0,1,7,6]]) #frecuencia

# Valores esperados
t = np.array([[0,0,0,0,1,1,1,1],
			  [0,0,1,1,0,0,1,1]])

# error
e = np.array(np.ones((2,8)))

# inicializamos de forma aleatoria
W = 2*np.random.rand(2,2)-1 # matriz de pesos sinápticos
b = 2*np.random.rand(2,1)-1 # vector de polarizaciÃ³n

for epocas in range(40): # número de épocas
	for q in range(8): # número de patrones de prueba
		a = hardlim(np.dot(W,P[:,q].reshape(-1,1))+b) # convierte a vector de 1 columna
		e[:,q] = (t[:,q].reshape(-1,1)-a).T # error
		W += np.dot(e[:,q].reshape(-1,1),P[:,q].reshape(-1,1).T)
		b += e[:,q].reshape(-1,1)

# Graficación del resultado
fig,ax = plt.subplots()

# Ligeros y poco usados
ax.scatter(P[0][0],P[1][0],marker='^')
ax.scatter(P[0][1],P[1][1],marker='^')

# Ligeros y muy usados
ax.scatter(P[0][2],P[1][2],marker='s')
ax.scatter(P[0][3],P[1][3],marker='s')

# Pesados y poco usados
ax.scatter(P[0][4],P[1][4],marker='o')
ax.scatter(P[0][5],P[1][5],marker='o')

# Pesados y muy usados
ax.scatter(P[0][6],P[1][6],marker='*')
ax.scatter(P[0][7],P[1][7],marker='*')


points = np.arange(0,6,0.01)

# Primera neurona
ax.plot(points, (-b[0][0]/W[0,1])-(W[0,0]/W[0,1])*points)
ax.plot(points, (-b[1][0]/W[1,1])-(W[1,0]/W[1,1])*points)
plt.show()
