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

# Función tangente sigmoidal
class TanSig:
	def __call__(self, x):
		return np.tanh(x)

	def deriv(self,x,y):
		return 1.0 - np.square(y)

# tangente hiperbólica
def tanh(x):
	t = (np.exp(x) - np.exp(-x))/(np.exp(x)+np.exp(-x))
	return t


# Entradas
P = np.array([[0,0,1,1],
			    [0,1,0,1]])

# Valores esperados
T = np.array([-1,1,1,-1]) # cambiamos 0 por -1
# Número de entrada de datos
Q = 4
# Número de neuronas
n1 = 34
# Epsilon: rango de valores iniciales
ep = 1 # parámetro que afectará a W y b iniciales

# Matriz de pesos sinápticos 1
W1 = ep*2*np.random.rand(n1,2)-1
# Vector de polarización 1
b1 = ep*2*np.random.rand(n1,1)-1

# Matriz de pesos sinápticos 2
W2 = ep*2*np.random.rand(1,n1)-1
# Vector de polarización 2
b2 = ep*2*np.random.rand(1,1)-1

total_epocas = 11000
a2 = np.array(np.zeros((1,Q)))
error_cuadratico_medio = np.array(np.zeros((1,total_epocas)))
alpha = 0.001

for epocas in range(total_epocas):
	sum_error = 0
	for q in range(Q):
		# Progagación de la entrada a la salida
		a1 = tanh(np.dot(W1,P[:,q].reshape(-1,1))+b1)
		a2[:,q] = tanh(np.dot(W2,a1)+b2)

		# Retropropagación de la sensibilidad
		e = T[q]-a2[:,q]
        
		# Sensibilidad 2
		s2 = -2*(1-(a2[:,q]**2))*e
        
		# Sensibilidad 1
		s1 = (np.diag(1-(a1**2))*W2.T)*s2
        
		# Actualización de pesos sinapticos (W) y vectores de polarización (b)
		W2 = W2 - alpha*s2*a1.T
		b2 = b2 - alpha*s2
		W1 = W1 - alpha*s1*P[:,q].reshape(-1,1).T
		b1 = b1 - alpha*s1
        
		# error cuadrático medio
		sum_error = e**2 + sum_error
        
	error_cuadratico_medio[:,epocas] = sum_error/Q

# Error cuadrático medio
print(f"EQM: {error_cuadratico_medio}")

a_verificacion = np.array(np.zeros((1,Q)))

# Verificamos el resultado
for q in range(Q):
	a_verificacion[:,q] = tanh(np.dot(W2,tanh(np.dot(W1,P[:,q].reshape(-1,1))+b1))+b2)

print(f"Valores esperados: {T}")
print(f"Valores de NN: {a_verificacion}")

# Frontera de decisión
# Gráfica de contorno
u = np.linspace(-2,2,100)
v = np.linspace(-2,2,100)
z = np.array(np.zeros((100,100)))

for i in range(100):
	for j in range(100):
		z[i,j] = tanh(np.dot(W2,(tanh(np.dot(W1,[[u[i]],[v[j]]])+b1)))+b2)

x = np.arange(0,total_epocas,1)

fig,(ax1,ax2) = plt.subplots(1,2)

ax1.set_title('Error cuadrático medio')
ax1.plot(x,error_cuadratico_medio.reshape(-1,1))
ax1.set(xlabel='#épocas',ylabel='Error')

ax2.set_title('Compuerta lógica XOR')
ax2.contour(u, v, z.T, 5, linewidths = np.arange(-0.9, 0, 0.9))
ax2.scatter(P[0][0],P[1][0], marker='o')
ax2.scatter(P[0][1],P[1][1], marker='o')
ax2.scatter(P[0][2],P[1][2], marker='o')
ax2.scatter(P[0][3],P[1][3], marker='o')

# Límites de los ejes
ax2.set_xlim([-0.5,1.5])
ax2.set_ylim([-0.5,1.5])

plt.show()
