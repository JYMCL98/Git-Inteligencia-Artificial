# Universidad Autónoma Chapingo
# Departamento de Ingeniería Mecánica Agrícola
# Ingeniería Mecatrónica Agrícola
# Jym Emmanuel Cocotle Lara
# 7° 7
# ------Proyecto Final------
# ------Base de datos-------

# Librerías utilizadas
import numpy as np
from PIL import Image
import os

# Creamos arreglos para almacenar las imagenes y las etiquetas
categorias = []
# Ubicación de las imágenes
categorias = os.listdir('D:/Desktop/Data/Data2')
tam_img = 80


def class_img(categorias,tam_img1,tam_img2):
    # Contadores
    x = 0
    y = 0
    labels = []
    imagenes = []
    for direccion in categorias:
        # Lee todas las imágenes de la carpetas en la ubicación
        for imagen in os.listdir('D:/Desktop/Data/Data2/'+ direccion):
            # Redimensionamiento de la imágen
            img2 = Image.open('D:/Desktop/Data/Data2/' + direccion + '/' + imagen).resize((tam_img,tam_img))
            # Convertimos en arreglo a la imagen redimensionada
            img2 = np.array(img2)
            imagenes.append(img2)
            # Agregamos el identificador a la imagen
            labels.append(x)
            if y == 500:
                break
            y += 1
        y = 0
        if x == 1:
            break
        x += 1
    
    # Convertimos en arreglo las etiquetas
    labels = np.array(labels)
    imagenes = np.array(imagenes)
    # Extraemos los valores de cada píxel (0-255)
    imagenes = imagenes[:,:,:,0] 
    archivo = open("D:/Desktop/Data/Chayotes.csv","w")
    total_img = np.size(imagenes[:,0,0])

    for j in range(total_img):
        # Colocación de identificador
        archivo.write(str(labels[j]))
        archivo.write(",")
        for k in range(tam_img): # 
            for l in range(tam_img):
                # Convertimos a escala 0 - 1
                pixels = imagenes[j,k,l]
                archivo.write(str(pixels))
                if k < (tam_img-1) or l < (tam_img-1):
                    # Separación de cada valor
                    archivo.write(",")
        archivo.write("\n")
    
    print("Archivo ordenado")
    archivo.close()

class_img(categorias, tam_img,tam_img)
