
import numpy as np
from PIL import Image
import os

categorias = []
categorias = os.listdir('D:/Desktop/Data')
print(f"Categorias: {categorias}")

tamanio_img = 80
def convertir_imagenes(categorias,tamanio_img1,tamanio_img2):
    # Contadores
    x = 0
    y = 0
    labels = []
    imagenes = []
    for direccion in categorias:
        # Lee todas las imágenes de la carpetas
        for imagen in os.listdir('D:/Desktop/Data/'+ direccion):
            img2 = Image.open('D:/Desktop/Data/' + direccion + '/' + imagen).resize((tamanio_img,tamanio_img)) # Redimensionamos la imagen a 28x28
            img2 = np.array(img2) # Convertimos en arreglo a la imagen redimensionada
            imagenes.append(img2) # Agregamos el valor leído al arreglo imagenes
            labels.append(x) # Agregamos una etiqueta a la imagen
            # Toma las fotos de cada carpeta
            if y == 1494: # Número de elementos de la carpeta
                break
            y += 1
        y = 0
        # Lee solo 1 carpeta
        if x == 1:
            break
        x += 1

    labels = np.array(labels)
    print(f"Etiquetas: {labels}")

    imagenes = np.array(imagenes)
    imagenes = imagenes[:,:,:,0] 
   
    ################ Ordenamiento de datos ################################
    archivo = open("D:/Desktop/Data/frutass.csv","w")

    total_imagenes = np.size(imagenes[:,0,0])

    for j in range(total_imagenes): # Número de imagenes
        # Pone la etiqueta del número de imagen que es
        archivo.write(str(labels[j]))
        archivo.write(",")
        for k in range(tamanio_img): # 
            for l in range(tamanio_img):
                # Convertimos a escala 0 - 1
                pixels = imagenes[j,k,l] #/255 descomentar esto si no jala xD
                archivo.write(str(pixels))
                if k < (tamanio_img-1) or l < (tamanio_img-1):
                    archivo.write(",") # Coloca una coma para cada valor, menos al último valor
        archivo.write("\n")
    
    print("Archivo ordenado")
    archivo.close()

    X = []
    Y = []
    Data = []
    data_path = ("D:/Desktop/Data/frutass.csv")
    Data = np.loadtxt(data_path, delimiter = ",")
    
    Data = np.array(Data).T
    #print(Data)
    Y = Data[0]
    X = Data[:][1:].T
    
    return X,Y

