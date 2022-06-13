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

# Función de activación tangente sigmoidal 
def tanh(x):
	value = (np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))
	return value

# Ruta del archivo
data_path = ("Data/Clasificacion100.csv") 
P = np.loadtxt(data_path,delimiter=",")
P = np.array(P) # (2 filas, 200 columnas)
Q = 200

# Valores esperados
T = np.ones((1,200))
T[:,100:] = -1

# Valores iniciales
n = 20 # Número de neuronas
ep = 1 # Factor de escalamiento

# Parámetros
# Matrices de pesos sinapticos y vectores de polarización
W1 = ep*2*np.random.rand(n,2)-ep
b1 = ep*2*np.random.rand(n,1)-ep

W2 = ep*2*np.random.rand(1,n)-ep
b2 = ep*2*np.random.rand(1,1)-ep


alfa = 0.01
numero_epocas = 500
error_qua_medio = np.array(np.zeros((1,numero_epocas)))
a2 = np.array(np.zeros((1,Q)))



for epocas in range(numero_epocas):
	sum_error = 0
	for q in range(Q):
		q = np.random.randint(0,200)

		# Propagación hacia delante (desde la entrada hasta la salida)
		a1 = tanh(np.dot(W1,P[:,q].reshape(-1,1))+b1)
		a2[:,q] = tanh(np.dot(W2,a1)+b2)

		# Cálculo de la sensibilidad
		e = T[:,q]-a2[:,q]
		s2 = -2*(1-a2[:,q]**2)*e
		s1 = (np.diag(1-a1**2)*W2.T)*s2

		# Actualizando los pesos sinapticos
		W2 = W2 -alfa*s2*a1.T
		b2 = b2 -alfa*s2

		W1 = W1 -alfa*s1*P[:,q].reshape(-1,1).T
		b1 = b1 -alfa*s1

		sum_error = e**2+sum_error
	error_qua_medio[:,epocas] = sum_error/Q

print(f"Error cuadrático medio: {error_qua_medio}")

# Visualizamos el error cuadrático medio
x = np.arange(0,numero_epocas,1)
fig, ax = plt.subplots()
ax.set_title("Error cuadrático medio")
ax.plot(x,error_qua_medio.reshape(-1,1))
ax.set(xlabel="# Epocas",ylabel = "MSE")

# Verificamos el algoritmo de backpropagation
a_verificacion = np.array(np.zeros((1,Q)))
for q in range(Q):
	a_verificacion[:,q] = tanh(np.dot(W2,tanh(np.dot(W1,P[:,q].reshape(-1,1))+b1))+b2)

x2 = np.arange(0,200,1)
fig2,ax2 = plt.subplots()
ax2.scatter(x2[100:],a_verificacion[:,100:].reshape(-1,1),color='red')
ax2.scatter(x2[:100],a_verificacion[:,:100].reshape(-1,1),color='blue')

# Gráficas de contorno
u = np.linspace(-15,15,50)
v = np.linspace(-15,15,50)
z = np.array(np.zeros((50,50)))

for i in range(50):
	for j in range(50):
		z[i,j] = tanh(np.dot(W2,tanh(np.dot(W1,[[u[i]],[v[j]]])+b1))+b2)

fig3,ax3 = plt.subplots()
ax3.scatter(P[0][100:],P[1][100:],marker="x",color='red')
ax3.scatter(P[0][:100],P[1][:100],marker="*",color='blue')
ax3.contour(u,v,z.T,linewidths=np.arange(-0.9,0,0.9))

plt.show()

