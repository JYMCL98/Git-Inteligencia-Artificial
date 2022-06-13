# -*- coding: utf-8 -*-
"""
Created on Wed Dec 15 16:08:20 2021

@author: jymcl
"""

import numpy as np
import matplotlib.pyplot as plt

def tanh(x):
    value = (np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))
    return value

data_path = "Data/NumerosManuscritos.csv"
P = np.loadtxt(data_path,delimiter=",")
P = np.array(P)

Q = 5000

T = np.vstack([np.ones((1,5000)),-np.ones((9,5000))])

for i in range(1,10):
    T = np.hstack([T,np.vstack([-np.ones((i,500)),np.ones((1,500)),-np.ones((9-i,500))])])

# Definimos el número de neuronas
n1 = 25 # número de neuronas
ep = 0.1 # Factor de escalamiento

W1 = ep*(2*np.random.rand(n1,400)-1)
b1 = ep*(2*np.random.rand(n1,1)-1)
W2 = ep*(2*np.random.rand(10,n1)-1)
b2 = ep*(2*np.random.rand(10,1)-1)

alfa = 0.1
total_epocas = 100



for epocas in range(total_epocas):
    for q in range(Q):
        # Propagación de la entrada hacia la salida
        a1 = tanh(np.dot(W1,P[:,q].reshape(-1,1))+b1)
        a2 = tanh(np.dot(W2,a1)+b2)
        
        # Retropropagación de las sensibilidades
        e = T[:,q]-a2
        s2 = -2*np.diag(1-a2**2)*e
        s1 = np.diag(1-a1**2)*(W2.T@s2)
        
        # Actualizacion de pesos sinapticos y polarizaciones
        W2 = W2-alfa*s2*a1.T
        b2 = b2-alfa*s2
        
        W1 = W1-alfa*s1*P[:,q].reshape(-1,1).T
        b1 = b1-alfa*s1
        
        sum_error = e.T*e
    
    error_qua_medio[:,epocas] = sum(sum_error.reshape(-1,1))/Q
    
# Visualizar el error cuadratico medio

x = np.arange(0,total_epocas,1)
fig,ax = plt.subplots()
plt.show()

# Verificación
for q in range(Q):
    a1 = tanh(np.dot(W1,P[q]))

