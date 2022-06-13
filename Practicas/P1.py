# Universidad Autónoma Chapingo
# Departamento de Ingeniería Mecánica Agrícola
# Ingeniería Mecatrónica Agrícola
# Jym Emmanuel Cocotle Lara
# 7° 7

#Librería para manejo de vectores y arreglos
import numpy as np

# Función escalón
def hardlim(n): 
	if n > 0:
		value = 1
	else:
		value = 0

	return value

# Entradas
# a,b,c,d,e,f,g
P = [[1,1,1,1,1,1,0], # 0
	 [0,1,1,0,0,0,0],  # 1
	 [1,1,0,1,1,0,1],  # 2
	 [1,1,1,1,0,0,1],  # 3
	 [0,1,1,0,0,1,1],  # 4
	 [1,0,1,1,0,1,1],  # 5
	 [1,0,1,1,1,1,1],  # 6
	 [1,1,1,0,0,0,0],  # 7
	 [1,1,1,1,1,1,1],  # 8
	 [1,1,1,1,0,1,1]]  # 9


# Valores esperados
t_pares = [1,0,1,0,1,0,1,0,1,0]
t_mayores_5 = [0,0,0,0,0,0,1,1,1,1]
t_numeros_p = [0,0,1,1,0,1,0,1,0,0]
t_impares = [0,1,0,1,0,1,0,1,0,1]

t = t_pares


# Error
e = np.ones(10) 

# Matriz de pesos sinápticos
W = 2*np.random.rand(1,7)-1
# Vector de polarización
b = 2*np.random.rand(1)-1

# Establecemos un for que reccorra el número de épocas
for epocas in range(500): 
   # Establecemos un for que reccorra el número de patrones de prueba
	for q in range(10):
		a = hardlim(np.dot(W,P[q])+b) # Salida de la neurona perceptrón
		e[q] = t[q]-a # Error
		W += np.dot(e[q],P[q]).T # Transpuesta
		b += e[q]

# Resultados
print(f"W: {W}")
print(f"b: {b}")
print(f"e: {e}")