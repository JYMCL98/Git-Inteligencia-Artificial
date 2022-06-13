
import numpy as np
#from numpy import asarray
#import cv2
from PIL import Image
import os

##################### Lectura ################################
# Creamos arreglos para almacenar las imagenes y las etiquetas
categorias = []
categorias2 = []
# labels = []
# imagenes = []
# Carpeta de entrenamiento
categorias = os.listdir('Data/training')
categorias2 = os.listdir("Data/detection")
#print(f"Categorias: {categorias}")
#print(f"Categorias2: {categorias2}")

tamanio_img = 80
def convertir_imagenes(categorias,tamanio_img1,tamanio_img2):
    # Contadores
    x = 0
    y = 0
    labels = []
    imagenes = []
    for direccion in categorias:
        # Lee todas las imágenes de la carpetas
        for imagen in os.listdir('Data/training/'+ direccion):
            img2 = Image.open('Data/training/' + direccion + '/' + imagen).resize((tamanio_img,tamanio_img)) # Redimensionamos la imagen a 28x28
            img2 = np.array(img2) # Convertimos en arreglo a la imagen redimensionada
            imagenes.append(img2) # Agregamos el valor leído al arreglo imagenes
            labels.append(x) # Agregamos una etiqueta a la imagen
            # Toma las fotos de cada carpeta
            if y == 1400: # Número de elementos de la carpeta
                break
            y += 1
        y = 0
        # Lee solo 1 carpeta
        if x == 1:
            break
        x += 1

    labels = np.array(labels) # Convertimos en arreglo las etiquetas
    #print(np.shape(labels))
    print(f"Etiquetas: {labels}")

    imagenes = np.array(imagenes) # Convertimos en arreglo las imagenes (3 canales)
    #imagenes1 = imagenes
    imagenes = imagenes[:,:,:,0] 
    #print(f"Imagenes: {imagenes}") # Extraemos los valores de cada pixel en escala 0 - 255 (un canal)



   

    X = []
    Y = []
    Data = []
    data_path = ("data/frutas.csv")
    Data = np.loadtxt(data_path, delimiter = ",")
    
    Data = np.array(Data).T
    #print(Data)
    Y = Data[0]
    X = Data[:][1:].T
    
    return X,Y


(X_train,y_train) = convertir_imagenes(categorias, tamanio_img,tamanio_img) #resize

(X_test,y_test) = convertir_imagenes(categorias2, tamanio_img,tamanio_img)
numero_filtros = 8
tamanio_filtros = 3
tamanio_maxpool = 2

tamanio_softmax = (tamanio_img - tamanio_filtros + 1)/tamanio_maxpool
#conv = Conv_op(numero_filtros,tamanio_filtros)
#pool = Max_Pool(tamanio_maxpool)
#softmax = Softmax(tamanio_softmax * tamanio_softmax * numero_filtros , 5)