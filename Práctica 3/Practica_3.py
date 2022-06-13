# Universidad Autónoma Chapingo
# Mecatrónica Agrícola
# Jym Emmanuel Cocotle Lara

# Librerías
import numpy as np 
import matplotlib.pyplot as plt
import math

# Ruta del archivo
ruta = ("C:/Users/jymcl/Downloads/mnist_test.csv") 
archivo = open(ruta)

P = np.loadtxt(ruta, delimiter = ',')
P = np.array(P)

pixel = np.ones((784,8920))
label = np.ones((784,8920))
print(np.shape(P))

contador = 0
for i in range (10):
    for j in range (8920):
        label = P[j,0]
        if label == i:
            for k in range (784):
                pixel[k,contador] = P[j,k]
            contador+=1

    # Una vez ordenado se crea un archivo con los datos ordenados
    ordenado = np.savetxt('ordenado.csv', pixel, fmt='%i', delimiter=',')

# Actualizamos la ruta
ruta = "ordenado.csv"

P = np.loadtxt(ruta,delimiter = ",")
P = np.array(P)
Z = np.vstack([P,np.ones((1,8920))])

#Valores esperados 
T = np.vstack([np.ones((1,892)),-np.ones((9,892))])

for i in range(1,10):
    T = np.hstack([T,np.vstack([-np.ones((i,892)),np.ones((1,892)),-np.ones((9-i,892))])])


#################Adaline##################
R = np.dot(Z,Z.T)/8920
H = np.dot(Z,T.T)/8920
X = np.linalg.pinv(R)@H

W = X[:784,:].T
b = X[784,:].reshape(-1,1)

index = np.ones((1,8920))
neurona_sal = np.ones((1,8920))

for q in range(8920):
    a = np.dot(W,P[:,q]).reshape(-1,1)+b
    neurona_sal[:,q] = np.amax(a)
    posicion = np.where(a == neurona_sal[:,q])
    index[:,q] = posicion[0]

# Valores reales
y = np.zeros((1,892))
for j in range(1,10):
    y = np.hstack([y,j*np.ones((1,892))])

numero_aciertos = np.sum(y == index)
porcentaje_aciertos = (numero_aciertos/8920)*100

print(f"Total de números identificados: {str (numero_aciertos)}")
print(f"Porcentaje de aciertos: {str(porcentaje_aciertos)} %")

for k in range(10):
    # Números aleatorios para mostrar
    indice = np.round((8919*np.random.rand(1)+1),0)
    print(f"Índice: {indice}")
    numero_reconocido = index[:,int(indice)]
    print(f"Número reconocido: {numero_reconocido}")
    pixels = P[:,int(indice)].reshape(28,28)
    plt.imshow(pixels, cmap = 'gray')
    plt.title(f'Número: {str(int(numero_reconocido))}')
    plt.show()
