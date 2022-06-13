# -*- coding: utf-8 -*-
"""
Created on Sat Dec 11 20:39:24 2021

@author: jymcl
"""

# Importamos las librerías
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from PIL import Image
import os
import cv2

ruta_archivo = ("D:/Desktop/Data/Chayote.csv")

tamanio_img = 80
tamanio = 497
num_iteraciones = 200
alpha = 0.01

print("////////////////// Algoritmo descenso por gradiente /////////////")
# Cargamos el archivo
data = pd.read_csv(ruta_archivo)
# Lo convertimos en un arreglo
data = np.array(data)
# Obtenemos las dimensiones del archivo
m, n = data.shape
# Ordenamos aleatoriamente los datos
np.random.shuffle(data)

dimension_1 = tamanio_img
dimension_2 = tamanio_img
dimension_img = dimension_1*dimension_2
# Datos que no se entrenan
data_dev = data[0:tamanio].T
Y_dev = data_dev[0] 
X_dev = data_dev[1:n] 
X_dev = X_dev / 255.0 

# Datos de entrenamiento
data_train = data[tamanio:m].T
Y_train = data_train[0] # etiquetas
X_train = data_train[1:n] # valores de cada imagen
X_train = X_train / 255.0 # Convertimos a escala 0 - 1
_,m_train = X_train.shape # Obtenemos el número de imágees "m" una vez transpuesta la capa de entrada

# Parámetros iniciales
def parametros_iniciales(): 
    # Inicializamos aleatoriamente los valores de los pesos sinápticos y los vectores de polarización
    # Valores entre -0.5 y 0.5 de dos dimensiones
    W1 = np.random.rand(2, dimension_img) - 0.5
    b1 = np.random.rand(2, 1) - 0.5
    W2 = np.random.rand(2, 2) - 0.5
    b2 = np.random.rand(2, 1) - 0.5
    return W1, b1, W2, b2

# función de activación ReLU
def ReLU(Z):
    return np.maximum(Z, 0) # si Z es menos a 0, se vuelve cero, en caso contrario Z no cambia su valor

# ´Función softmax convierte la variables en probabilidades entre 0 y 1
def softmax(Z):
    A = np.exp(Z) / sum(np.exp(Z))
    return A

# Propagación hacia adelante consigue una predicción a la salida
def propagacion_adelante(W1, b1, W2, b2, X):
    Z1 = W1.dot(X) + b1
    A1 = ReLU(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2

# Derivada de la función ReLU
def ReLU_deriv(Z):
    return Z > 0

# Retrocedemos para calcular el error
def one_hot(Y):
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    one_hot_Y[np.arange(Y.size), Y] = 1
    one_hot_Y = one_hot_Y.T
    return one_hot_Y

# Propagación hacia atrás
def propagacion_atras(Z1, A1, Z2, A2, W1, W2, X, Y): 
    one_hot_Y = one_hot(Y)
    dZ2 = A2 - one_hot_Y
    dW2 = 1 / m * dZ2.dot(A1.T)
    db2 = 1 / m * np.sum(dZ2)
    dZ1 = W2.T.dot(dZ2) * ReLU_deriv(Z1)
    dW1 = 1 / m * dZ1.dot(X.T)
    db1 = 1 / m * np.sum(dZ1)
    return dW1, db1, dW2, db2

# Actualizamos los parámetros
def actualiza_parametros(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * db1    
    W2 = W2 - alpha * dW2  
    b2 = b2 - alpha * db2    
    return W1, b1, W2, b2

def obtiene_predicciones(A2):
    return np.argmax(A2, 0) # Obtiene el valor máximos

def obtiene_precision(predictions, Y):
    print(predictions, Y)
    return np.sum(predictions == Y) / Y.size

# El descenso por grandiente trata de 
# minimizar una función que mide el error de predicción del modelo en el conjunto de datos
def descenso_gradiente(X, Y, alpha, iterations):
    W1, b1, W2, b2 = parametros_iniciales()
    for i in range(iterations):
        Z1, A1, Z2, A2 = propagacion_adelante(W1, b1, W2, b2, X)
        dW1, db1, dW2, db2 = propagacion_atras(Z1, A1, Z2, A2, W1, W2, X, Y)
        W1, b1, W2, b2 = actualiza_parametros(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)
        if i % 10 == 0:
            #print(f"Iteración: {i} ")
            predictions = obtiene_predicciones(A2)
            print(f"Precisión: {obtiene_precision(predictions, Y)}")
    return W1, b1, W2, b2

def descenso_2(X,Y,alpha,iterations):
    W1,b1,W2,b2 = descenso_gradiente(X_train, Y_train, alpha, num_iteraciones)
    for j in range(iterations):
        Z1, A1, Z2, A2 = propagacion_adelante(W1, b1, W2, b2, X)
        dW1, db1, dW2, db2 = propagacion_atras(Z1, A1, Z2, A2, W1, W2, X, Y)
        W1, b1, W2, b2 = actualiza_parametros(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)
        if j % 10 == 0:
            #print(f"Iteración: {i} ")
            predictions = obtiene_predicciones(A2)
            print(f"Precisión: {obtiene_precision(predictions, Y)}")
    return W1, b1, W2, b2

W1, b1, W2, b2 = descenso_gradiente(X_train, Y_train, alpha, num_iteraciones)

W1, b1, W2, b2 = descenso_2(X_train, Y_train, alpha, num_iteraciones)

def hace_predicciones(X, W1, b1, W2, b2):
    _, _, _, A2 = propagacion_adelante(W1, b1, W2, b2, X)
    predictions = obtiene_predicciones(A2)
    return predictions

def prueba_predicciones(index, W1, b1, W2, b2):
    current_image = X_train[:, index, None]
    # Meter aquí el video
    prediction = hace_predicciones(X_train[:, index, None], W1, b1, W2, b2)
    label = Y_train[index]
    print(f"Deducción: {prediction}")
    print(f"Etiqueta: {label}")
    
    current_image = current_image.reshape((dimension_1, dimension_2)) * 255
    plt.gray()
    plt.imshow(current_image, interpolation='nearest')
    
    if( prediction == 1 and label == 1 ):
        plt.title("Es guayaba")
    elif (prediction == 0 and label == 0):
        plt.title("No es guayaba")
    else:
        plt.title("No detectó correctamente")
    plt.show()
        
for i in range(20):
    prueba_predicciones(i, W1, b1, W2, b2)

dev_predictions = hace_predicciones(X_dev, W1, b1, W2, b2)
obtiene_precision(dev_predictions, Y_dev)

print("/////////////////  Video ////////////////////////////////////////")

video = cv2.VideoCapture(0)
if not video.isOpened():
    print("No hay cámara")
    exit()

while(video.isOpened()):
    # Captura trama a trama
    ret, frame = video.read()
    if ret == True:
        frame_original = frame
        escala_grises = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        frame_2 = np.array((cv2.resize(escala_grises,(dimension_1,dimension_2))))
        frame_2 = frame_2.reshape(-1,1)
        # le pasamos los frames del video en tiempo real en lugar de las imágenes de la carpeta
        # después del entrenamiento
        
        X_ = np.array(frame_2, dtype="int64")
        P = X_/255.0
        prediccion_1 = hace_predicciones(P,W1,b1,W2,b2)
        print(f"Deducción: {prediccion_1}")
        
        fruta_reconocida = prediccion_1
        if fruta_reconocida == 1:
            fruta = "Guayaba"
            x_imagen = 100
            color = (153,255,255) # BGR
        elif fruta_reconocida == 0:
            fruta = "No guayaba"
            x_imagen = 200
            color = (0,0,0)
        frame_original = cv2.putText(frame_original,fruta,(x_imagen,50), cv2.FONT_HERSHEY_COMPLEX, 1.5, color, 2)
        # Mostramos la ventana
        cv2.imshow("Frame", frame_original)

        # Presiona ESC para salir
        if cv2.waitKey(1) == 27:
            print("//////////////// Termina el video ///////////////////////////////")
            break

video.release()

# Closes all the frames
cv2.destroyAllWindows()
#