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

# Función tangente hiperbólica
def tanh(x):
	value = (np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))
	return value

# Ruta del archivo
data_path = "Data/NumerosManuscritos.csv"
P = np.loadtxt(data_path,delimiter=",")
P = np.array(P)

Q = 500
T = np.vstack([np.ones((1,500)),-np.ones((9,500))])

for i in range(1,10):
	T = np.hstack([T,np.vstack([np.ones((i,500)),np.ones((1,500)),-np.ones((9-i,500))])])

n1 = 25
ep = 0.1

# Matrices de pesos sinápticos y vectores de polarización
W1 = ep*(2*np.random.rand(n1,400)-1)
b1 = ep*(2*np.random.rand(n1,1)-1)
W2 = ep*(2*np.random.rand(10,n1)-1)
b2 = ep*(2*np.random.rand(10,1)-1)

alpha = 0.01

total_epocas = 100
error_cuadratico_medio = np.array(np.zeros((1,total_epocas)))

for epocas in range(total_epocas):
	sum_error = 0
	for q in range(Q):
		# Propagación hacia adelante de la entrada a la salida
		a1 = tanh(np.dot(W1,P[:,q].reshape(-1,1))+b1)
		a2 = tanh(np.dot(W2,a1)+b2)
		# Error
		e = T[:,q].reshape(-1,1) - a2

		# Propagación hacia atrás de las sensibilidades
		s2 = -2*np.diag(1-a2**2)*e
		s1 = np.diag(1-a1**2)*(W2.T@s2)

		# Actualizamos los parámetros
		W2 = W2 - alpha*s2*a1.T
		b2 = b2 - alpha*s2

		W1 = W1 - alpha*s1*P[:,q].reshape(-1,1).T
		b1 = b1 - alpha*s1

		sum_error = e.T*e
	error_cuadratico_medio[:,epocas] = sum(sum_error.reshape(-1,1))/Q


# Visualizar el error cuadrático medio
x = np.arange(0,total_epocas,1)
fig,ax = plt.subplots()

ax.plot(x,error_cuadratico_medio.reshape(-1,1))
plt.show()

index = np.ones((1,5000))
neurona_sal = np.ones((1,5000))
for q in range(5000):
	a1 = tanh(np.dot(W1,P[:,q].reshape(-1,1))+b1)
	a = np.amax(np.dot(W2,a1)+b2)
	neurona_sal[:,q] = a
	posicion = np.where(a==neurona_sal[:,q])
	index[:,q] = posicion[0]

# Valores reales
y = np.zeros((1,500))
for j in range(1,10):
	y = np.hstack([y,j*np.ones((1,500))])

numero_aciertos = np.sum(y==index)
print(f"Total aciertos: {str(numero_aciertos)}")
porcentaje_aciertos = (numero_aciertos/5000)*100
print(f"{str(porcentaje_aciertos)} % aciertos")

for k in range(10):
	indice = np.round((4999*np.random.rand(1)+1),0)
	print(indice)
	numero_reconocido = index[:,int(indice)]
	print(f"Número reconocido: {numero_reconocido}")
	pixels = P[:,int(indice)].reshape(20,20).T
	plt.imshow(pixels,cmap="gray")
	plt.title("Número reconocido: "+str(int(numero_reconocido)))
	plt.show()
