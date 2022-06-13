# -*- coding: utf-8 -*-
"""
Created on Wed Dec  8 12:49:44 2021

@author: jymcl
"""

from PIL import Image
import numpy as np
import time



import numpy as np
from PIL import Image

image = Image.open("D:/Documents/Chapingo/7° semestre/Inteligencia Artificial/Proyecto Final/Data/detection/no_real/1.jpg")
np_array = np.array(image)

pil_image=Image.fromarray(np_array)
pil_image.show()


'''

im = Image.open("D:/Documents/Chapingo/7° semestre/Inteligencia Artificial/Proyecto Final/Data/detection/no_real/1.jpg")

pixels = im.load()

        
print(pixels)




import numpy as np
from PIL import Image
img = Image.open("D:/Documents/Chapingo/7° semestre/Inteligencia Artificial/Proyecto Final/Data/detection/no_real/1.jpg")
imgArray = np.asarray(img)
print(imgArray.shape)




imagen = "D:/Documents/Chapingo/7° semestre/Inteligencia Artificial/Proyecto Final/Data/detection/no_real/1.jpg"

imagenes = np.array(imagen) # Convertimos en arreglo las imagenes (3 canales)
#imagenes1 = imagenes
#imagenes = imagenes[:,:,:,0] 
#print(f"Imagenes: {imagenes}") # Extraemos los valores de cada pixel en escala 0 - 255 (un canal)

print(imagenes)

'''