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
	if n[2][0] > 0:
		value3 = 1
	else:
		value3 = -1
	return [[value1],[value2],[value3]]


# a,b,c,d,e,f,g
P = np.array([[1,1,1,1,1,1,0],  #0
	           [0,1,1,0,0,0,0],  #1
	           [1,1,0,1,1,0,1],  #2
	           [1,1,1,1,0,0,1],  #3
	           [0,1,1,0,0,1,1],  #4
	           [1,0,1,1,0,1,1],  #5
	           [1,0,1,1,1,1,1],  #6
	           [1,1,1,0,0,0,0],  #7
	           [1,1,1,1,1,1,1],  #8
	           [1,1,1,1,0,1,1]]) #9

# Transpuesta de P
P = P.transpose()

# Vector aumentado
Z = np.vstack([P,np.ones(10)])


# Valores esperados
t_pares = [1, -1, 1, -1, 1, -1, 1, -1, 1, -1]
t_mayores_5 = [-1,-1,-1,-1,-1,-1,1,1,1,1]
t_numeros_p = [-1,-1,1,1,-1,1,-1,1,-1,-1]
t_impares = [-1,1,-1,1,-1,1,-1,1,-1,1]

# Valores esperados
T = np.array([t_pares,t_mayores_5,t_numeros_p])

# Algoritmo Adaline
R = np.dot(Z,Z.T)/10 # 10 = #valores esperados
H = np.dot(Z,T.T)/10
X = np.linalg.inv(R) @ H
W = X[:7,:3].T
b = X[7,:3].reshape(-1,1)

print(f"W: {W}")
print(f"b: {b}")

# Verificando la solución 
e = np.ones((3,10))


for q in range(10):
	a = hardlim(np.dot(W,P[:,q].reshape(-1,1))+b)
	e[:,q] = (T[:,q].reshape(-1,1)-a).T

print(f"Error: {e}")