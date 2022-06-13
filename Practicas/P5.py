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

# Ruta del archivo
data_path = "Data/NumerosManuscritos.csv"
# Establecemos a P con los datos del archivo
P = np.loadtxt(data_path,delimiter=",")
# Convertimos el archivo en un arreglo
P = np.array(P)

Z = np.vstack([P,np.ones((1,5000))])

# Valores esperados
T = np.vstack([np.ones((1,500)),-np.ones((9,500))]) # Agregar filas
for i in range(1,10):
	T = np.hstack([T,np.vstack([-np.ones((i,500)),np.ones((1,500)),-np.ones((9-i,500))])]) # agregar columnas

# Red neuronal Adaline
# 5000 = número de datos
R = np.dot(Z,Z.T)/5000
# 5000 = número de datos
H = np.dot(Z,T.T)/5000
# Multiplicación de matrices pseudoinversa
X = np.linalg.pinv(R) @ H

# Matriz de pesos sinápticos
W = X[0:400,:].T
# Vector de polarización
b = X[400,:].reshape(-1,1)

# Visualizar qué tan bien lo hizo
index = np.ones((1,5000))
neurona_sal = np.ones((1,5000))

for q in range(5000):
	a = np.dot(W,P[:,q]).reshape(-1,1)+b
   # Función de activación
	neurona_sal[:,q] = np.amax(a)
	posicion = np.where(a==neurona_sal[:,q])
	index[:,q] = posicion[0]

y = np.zeros((1,500))
for j in range(1,10):
	y = np.hstack([y,j*np.ones((1,500))])

numeros_aciertos = np.sum(y==index)
porcentaje_aciertos = (numeros_aciertos/5000)*100
print(f"{porcentaje_aciertos} % de aciertos")

for k in range(10):
	indice = np.round((4999*np.random.rand(1)+1),0)
	numero_reconocido = index[:,int(indice)]
	print(f"Número reconocido: {numero_reconocido}")

	pixels = P[:,int(indice)].reshape(20,20).T
	plt.imshow(pixels, cmap="gray")
	plt.title("Número reconocido: "+str(int(numero_reconocido)))
	plt.show()
