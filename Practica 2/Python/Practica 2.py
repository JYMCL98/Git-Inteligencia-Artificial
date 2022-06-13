# Identificador de frutas
# Jym Emmanuel Cocotle Lara
# 7° 7

import numpy  as np # Librería para vectores y matrices

# Funcion escalón
def hardlim(n):
    if n > 0:
        value = 1
    else:
        value = 0
    return value

   # R,  G,  B , P  Valores obtenidos a partir del sensor de color y la galga extensiométrica
P =[[5,  10, 13, 126.12],  # Mandarina 1
    [4,  6,  9,  142.9],   # Manzana 1
    [5,  11, 13, 148.58],  # Mandarina 2
    [5,  7,  9,  134.08],  # Manzana 2
    [7,  14, 16, 128.85],  # Mandarina 3
    [6,  9,  12, 142.47],  # Manzana 3
    [5,  11, 13, 158.33],  # Mandarina 4
    [3,  3,  7,  142.89],  # Manzana 4
    [4,  11, 13, 112.5],   # Mandarina 5
    [3,  5,  7,  140.69],  # Manzana 5
    [6,  11, 13, 90.46],   # Mandarina 6
    [2,  3,  5,  145.4],   # Manzana 6
    [5,  11, 13, 117.28],  # Mandarina 7
    [2,  2,  5,  148.06],  # Manzana 7
    [6,  11, 13, 84.87],   # Mandarina 8
    [2,  2,  6,  149.6],   # Manzana 8
    [5,  11, 15, 104.4],   # Mandarina 9
    [3,  5,  8,  142.61],  # Manzana 9
    [3,  6,  10, 147.21],  # Mandarina 10
    [2,  5,  8,  154.75]]  # Manzana 10

# Valores esperados
t = [0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1] # se activa la neurona cuando hay una manzana

# Error
e = np.ones(20)              
W = 2*np.random.rand(1,4)-1  # Matriz de pesos sinápticos
b = 2*np.random.rand(1)-1  #Vector de polarización

for epocas in range(50000):
    for q in range(20):
        a = hardlim(np.dot(W,P[q])+b) # Función de activación
        e[q] = t[q] - a
        W += np.dot(e[q],P[q]).T
        b += e[q]

print(W)
print(b)
print(e)

