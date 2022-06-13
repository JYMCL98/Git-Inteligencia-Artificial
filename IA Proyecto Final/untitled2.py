# -*- coding: utf-8 -*-
"""
Created on Tue Apr 26 09:13:49 2022

@author: jymcl
"""

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


direc = 'D:/Documents/Data/Martelloscope/trainval'
# Creamos arreglos para almacenar las imagenes y las etiquetas
categorias = []
# Ubicación de las imágenes
categorias = os.listdir(direc)
tam_img_x = 274
tam_img_y = 182


def class_img(categorias,tam_img_x,tam_img_y):
    # Contadores
    x = 0
    y = 0
    labels = []
    imagenes = []
    for direccion in categorias:
        # Lee todas las imágenes de la carpetas en la ubicación
        for imagen in os.listdir(direc+'/'+ direccion):
            # Redimensionamiento de la imágen
            img2 = Image.open(direc+'/' + direccion + '/' + imagen).resize((tam_img_x,tam_img_y))
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
    archivo = open(direc+"/plants_2.csv","w")
    total_img = np.size(imagenes[:,0,0])

    for j in range(total_img):
        # Colocación de identificador
        archivo.write(str(labels[j]))
        archivo.write(",")
        for k in range(tam_img_y): # 
            for l in range(tam_img_x):
                # Convertimos a escala 0 - 1
                pixels = imagenes[j,k,l]
                archivo.write(str(pixels))
                if k < (tam_img_x-1) or l < (tam_img_y-1):
                    # Separación de cada valor
                    archivo.write(",")
        archivo.write("\n")
    
    print("Archivo ordenado")
    archivo.close()

class_img(categorias, tam_img_x,tam_img_y)
