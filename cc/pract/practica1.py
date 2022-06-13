# Importamos la librería a utilizar
import numpy as np

# Función escalón
def hardlim(n):
	if n>0:
		value = 1
	else:
		value = 0

	return value

'''
# Entradas
# a,b,c,d,e,f,g
P = [[1,1,1,1,1,1,0],  #0
	 [0,1,1,0,0,0,0],  #1
	 [1,1,0,1,1,0,1],  #2
	 [1,1,1,1,0,0,1],  #3
	 [0,1,1,0,0,1,1],  #4
	 [1,0,1,1,0,1,1],  #5
	 [1,0,1,1,1,1,1],  #6
	 [1,1,1,0,0,0,0],  #7
	 [1,1,1,1,1,1,1],  #8
	 [1,1,1,1,0,1,1]]  #9
'''

P = [[20,255,214,170],  #Manzana
	 [40,225,213,150],  #Naranja
	 [1,1,0,1,1,0,1],  #2
	 [1,1,1,1,0,0,1],  #3
	 [0,1,1,0,0,1,1],  #4
	 [1,0,1,1,0,1,1],  #5
	 [1,0,1,1,1,1,1],  #6
	 [1,1,1,0,0,0,0],  #7
	 [1,1,1,1,1,1,1],  #8
	 [1,1,1,1,0,1,1]]  #9


# Valores esperados
t_manzanas = [1,0,1,1,0,0,1]
t_pares = [1,0,1,0,1,0,1,0,1,0]
t_mayores_5 = [0,0,0,0,0,0,1,1,1,1]
t_impares = [0,1,0,1,0,1,0,1,0,1]
t_numeros_p = [0,0,1,1,0,1,0,1,0,0]

t = t_mayores_5

e = np.ones(10) # Error

W = 2*np.random.rand(1,7)-1 # Matriz de pesos sinápticos
b = 2*np.random.rand(1)-1  # Vector de polarización


for epocas in range(1000):
	for q in range(10):
		a = hardlim(np.dot(W,P[q])+b) # neurona perceptrón
		e[q] = t[q]-a
		W = W+np.dot(e[q],P[q]).T  # Transpuesta
		b = b+e[q]


print(e)
print(W)
print(b)
