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

# Función de activación sigmoidal
def sigmod(x):
    value = 1/(1+np.exp(-x))
    return value

# Ruta del archivo
data_path = ("data/RegresionBursatil.csv") 
Data = np.loadtxt(data_path,delimiter=",")
Data = np.array(Data) # datos de entrada, datos deseados

P = Data[0]
T = Data[1]
Q = 31

# Número de neuronas de la primera capa
n1 = 30
# Número de neuronas de la segunda capa
n2 = 40
ep = 1

# Matrices de pesos sinápticos y vectores de polarización
W1 = ep*2*np.random.rand(n1,1)-ep
b1 = ep*2*np.random.rand(n1,1)-ep
W2 = ep*2*np.random.rand(n2,n1)-ep
b2 = ep*2*np.random.rand(n2,1)-ep
W3 = ep*2*np.random.rand(1,n2)-ep
b3 = ep*2*np.random.rand(1,1)-ep

alfa = 0.001
total_epocas = 4000
error_qua_medio = np.array(np.zeros((1,total_epocas)))

for epocas in range(total_epocas):
    sum_error = 0
    for q in range(Q):
        # Algoritmo de propagación hacia adelante
        a1 = sigmod(np.dot(W1,P[q].reshape(-1,1))+b1)
        a2 = sigmod(np.dot(W2,a1)+b2)
        a3 = np.dot(W3,a2)+b3
        
        # Retropropagacion hacia atras o de las sensibilidades
        e = T[q]-a3
        s3 = -2*e
        s2 = (np.diag((1-a2)*a2)*W3.T)*s3
        s1 = np.diag((1-a1)*a1)*(W2.T@s2)
        
        # Actualizar los pesos sinápticos y las polarizaciones
        W3 = W3-alfa*s3*a2.T
        b3 = b3-alfa*s3
        W2 = W2-alfa*s2*a1.T
        b2 = b2-alfa*s2
        W1 = W1-alfa*s1*P[q].T
        b1 = b1-alfa*s1
        
        sum_error = e**2+sum_error
    error_qua_medio[:,epocas] = sum_error/Q
        

# Verificando la solución de la red neuronal Backpropagation
p = np.arange(0,6,0.01)
a3 = np.array(np.zeros((1,np.shape(p)[0])))

for q in range(np.shape(p)[0]):
    a1 = sigmod(np.dot(W1,p[q])+b1)
    a2 = sigmod(np.dot(W2,a1)+b2)
    a3[:,q] = np.dot(W3,a2)+b3
    
# Gráfica
fig, ax = plt.subplots()
ax.scatter(Data[0],Data[1])
ax.plot(p,a3.reshape(-1,1),c='red')

x = np.arange(0,total_epocas,1)
fig2, ax2 = plt.subplots()
ax2.plot(x,error_qua_medio.reshape(-1,1))

plt.show()
