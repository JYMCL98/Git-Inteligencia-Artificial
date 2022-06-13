# UNIVERSIDAD AUTÓNOMA CHAPINGO
# DEPARTAMENTO DE INGENIERÍA MECÁNICA AGRÍCOLA
# INGENIERÍA MECATRÓNICA AGRÍCOLA
# INTELIGENCIA ARTIFICAL
# Convolutional Neural Networks
# LUIS ANGEL SANCHEZ RODRIGUEZ 
# 7°7

# Importamos las librerías
import numpy as np
from numpy import asarray
import cv2
import matplotlib.pyplot as plt

"""
# Importamos la imagen
img = cv2.imread('D:/RESPALDO/LAPTOP/Python/images/yuan_baneina.jpg', cv2.IMREAD_GRAYSCALE)/255 # Leemos la imagen
# Mostramos la imagen en escala de grises
plt.imshow(img, cmap = 'gray')
plt.show()
print(img.shape) # Muestra las dimensiones de la imagen

"""

# Convolución
class Conv_op:
    # Definimos los filtros requeridos para la convolución
    def __init__(self, num_filters, filter_size):
        self.num_filters = num_filters
        self.filter_size = filter_size
    # Inicializamos aleatoriamente los valores de los filtros
        self.conv_filter = np.random.randn(num_filters, filter_size, filter_size)/(filter_size * filter_size)
    # Extrae los parches de la imagen
    def image_region(self,image): #generator function
        height, width = image.shape
        self.image = image
    # Extraemos las medidades de los parches
        for j in range(height - self.filter_size + 1):
            for k in range(width - self.filter_size +1):
      	# almacena el valor de los parches en una nueva variable de imagen
                image_patch = image[j:(j+self.filter_size),k:(k+self.filter_size)]
                yield image_patch, j, k 

    def forward_prop(self,image):
        height, width = image.shape
        conv_out = np.zeros((height - self.filter_size + 1, width - self.filter_size+1, self.num_filters))
        for image_path, i,j in self.image_region(image):
    	# Multiplicamos el filtro por los parches de la imagen
            conv_out[i,j]= np.sum(image_path * self.conv_filter, axis=(1,2))
        return conv_out
    
    # Propagación hacia atrás 
    # dL_dout viene del max pooling
    def back_prop(self, dL_dout, learning_rate):
        dL_dF_params = np.zeros(self.conv_filter.shape)
        for image_patch, i,j in self.image_region(self.image):
            for k in range(self.num_filters):
                dL_dF_params[k] += image_patch*dL_dout[i,j,k]

    # Actualizamos los parámetros del filtro
        self.conv_filter -= learning_rate*dL_dF_params
        return dL_dF_params


# Operación Maxpool
class Max_Pool:
    # Definimos el tamaño del filtro requerido
    def __init__(self, filter_size):
        self.filter_size = filter_size
    
    # Reducirá el tamaño de la imagen dividiéndola entre el tamaño del filtro
    def image_region(self, image):
        new_height = image.shape[0] // self.filter_size
        new_width = image.shape[1] // self.filter_size
        self.image = image
        # Extraemos los parches de la imagen nueva que se calcula en la función de arriba
        for i in range(new_height):
            for j in range(new_width):
                image_patch = image[(i*self.filter_size):(i*self.filter_size + self.filter_size), (j*self.filter_size):(j*self.filter_size + self.filter_size)]
                yield image_patch, i, j

    # Módulo de propagación hacia adelante
    def forward_prop(self,image):
        height, width, num_filters = image.shape
        output = np.zeros((height // self.filter_size, width // self.filter_size, num_filters))

        for image_patch, i, j in self.image_region(image):
            output[i,j] = np.amax(image_patch, axis = (0,1))

        return output

    # Módulo de propagación hacia atrás
    # este módulo requerirá una de las entradas del módulo de propagación hacia atrás de la clase Softmax
  	# el valor de dL_dout es el que viene de la clase Softmax
    def back_prop(self, dL_dout):
        dL_dmax_pool = np.zeros(self.image.shape)
        for image_patch, i, j in self.image_region(self.image):
            height, width, num_filters = image_patch.shape
            maximum_val = np.amax(image_patch,axis = (0,1))

            for i1 in range(height):
                for j1 in range(width):
                    for k1 in range(num_filters):
                        if image_patch[i1,j1,k1] == maximum_val[k1]:
                            dL_dmax_pool[i*self.filter_size + i1, j*self.filter_size + j1, k1]=dL_dout[i,j,k1]
            return dL_dmax_pool

# Operación Softmax
class Softmax:
    # definimos los pesos y las bias de
    def __init__(self, input_node, softmax_node):
        self.weight = np.random.randn(input_node, softmax_node)/input_node
        self.bias = np.zeros(softmax_node)

    # multiplicamos los pesos por las biases y generando salidas desde las capas ocultas
    # compactamos el cubo
    def forward_prop(self, image):
        self.orig_im_shape = image.shape #used in backprop
        image_modified = image.flatten()
        self.modified_input = image_modified #to be used in backprop
        output_val = np.dot(image_modified, self.weight) + self.bias
        self.out = output_val
        exp_out = np.exp(output_val)

        # transformamos la salida en las salidas probables dadas
        return exp_out/np.sum(exp_out, axis=0)

    def back_prop(self, dL_dout, learning_rate):
        for i, grad in enumerate(dL_dout):
            if grad ==0:
                continue

            transformation_eq = np.exp(self.out)
            S_total = np.sum(transformation_eq)

            #gradients with respect to out (z)
            dy_dz = -transformation_eq[i]*transformation_eq / (S_total **2)
            dy_dz[i] = transformation_eq[i]*(S_total -  transformation_eq[i]) / (S_total **2)

            #gradients of totals against weights/biases/input
            dz_dw =  self.modified_input
            dz_db = 1
            dz_d_inp = self.weight

            #gradients of loss against totals
            dL_dz = grad * dy_dz

            #gradients of loss against weights/biases/input
            dL_dw = dz_dw[np.newaxis].T @ dL_dz[np.newaxis]
            dL_db = dL_dz * dz_db
            dL_d_inp = dz_d_inp @ dL_dz

      # Actualizamos los pesos y las biases
        self.weight -= learning_rate *dL_dw
        self.bias -= learning_rate * dL_db

        return dL_d_inp.reshape(self.orig_im_shape)


"""
# Implementación de la operación de Convolución
# se le pasan el número de filtros (18) y el tamaño del filtro (7x7)
conn = Conv_op(18,7)
out = conn.forward_prop(img) # Llamamos a la convolución hacia adelante y le pasamos la imagen que tenemos
print(out.shape) # Imprimimos los valores de la salida
# Tenemos una imagen de (225,225), lo convolucionamos con 18 filtros de medida 7x7
# (225-7+1)*(225-7+1), # número de imágenes
# (219,219,18)

# imprimimos las imágenes aplicando la convolución
for i in range(0,18):
    plt.imshow(out[:,:,i],cmap='gray')
    plt.show()

//////////////////////////////////////////////////////////////////////////////////////////
# Implementación de la operación de Max pooling
# El parámetro es el tamaño del filtro que es usado en la operación de Max pooling (4x4)
conn2 = Max_Pool(4)
out2 = conn2.forward_prop(out) # Llamamos a la salida de la operación de convolución
# Imprimimos las medidas de la salida
print(out2.shape) #(54, 54, 18) # la salida es reducida puesto que se divide entre el filtro
# 219/4,219/4,número de imágenes (canales)
# el objetivo fue retener los bordes presentes en la imagen para calcular las características de la imagen

# imprimimos las imágenes aplicando el max pool
for i in range(0,18):
    plt.imshow(out2[:,:,i],cmap='gray')
    plt.show()

/////////////////////////////////////////////////////////////////////////////////////////////////////////////
# implementación de la operación softmax
conn3 = Softmax(54*54*18,10) # (tamaño de la imagen*tamaño de la imagen*tamaño de la imagen, número de salidas)
# transformamos el cubo que venía de la operación max pool en un arreglo de una dimensión
out3 = conn3.forward_prop(out2) # le pasamos la imagen que viene de la operación max pool
print(out3)


"""

from keras.datasets import WiderFace
(X_train, y_train), (X_test, y_test) = WiderFace.load_data()
# 2500 imagenes
# imágenes del entrenamiento
train_images = X_train[:2500]
train_labels = y_train[:2500]
# imágenes de la prueba
test_images = X_test[:2500]
test_labels = y_test[:2500]

num_test = len(test_images)


"""
# número de pasos = número de imágenes
ruta_imagenes = glob.glob("dataset_train/*.jpg")
train_images = []
for img in ruta_imagenes:
	# cargamos imágenes de entrenamiento en blanco y negro
	n = cv2.imread(img, 0) 
	# modificamos el tamaño
	n = cv2.resize(n, (125,125), interpolation=cv2.INTER_LINEAR)
	# agregamos la imagen ajustada al arreglo de imágenes de entrenamiento
	train_images.append(n)
"""

# llamamos a las 3 clases
conv = Conv_op(8,3) #número de filtros, tamaño del filtro
# (28x28x1)-->(28-3+1,28-3+1,1*8) = 26x26x8

# reducimos la imagen entre 2
# (26x26x8)-->(26/2,26/2,8)
pool = Max_Pool(2) # tamaño del filtro

# aplanamos el volumen en un vector de una dimensión
softmax = Softmax(13*13*8, 10)


# empleando la propagación hacia adelante en una red neuronal convolucional
# le pasamos el número de imágenes y obtenemos una salida como 10 clases

def cnn_forward_prop(image, label):
    #  alimentamos a la imagen con la operación de convolución hacia adelante
    out_p = conv.forward_prop((image /255) - 0.5)
    # le pasamos el parámetro a la operación de max pool
    out_p = pool.forward_prop(out_p)
    # nuevamente le pasamos el parámetro a la función de softmax
    out_p = softmax.forward_prop(out_p)

    # calculamos la pérdida de la entropía y la precisión 
    cross_ent_loss = -np.log(out_p[label])
    accuracy_eval = 1 if np.argmax(out_p) == label else 0

    return out_p, cross_ent_loss, accuracy_eval

# vamos a entrenar a la CNN a través de la propagación hacia atrás
# obtenemos los resultados de la salida y alimentamos hacia atrás el error a las capas anteriores
def training_cnn(image, label, learn_rate = 0.005):
    # hacia adelante
    # calculamos la salida, la pérdida y la precisión 
    out, loss, acc = cnn_forward_prop(image, label)

    # calculamos el gradiente inicial
    gradient = np.zeros(10)
    gradient[label] = -1 / out[label]

    # propagación hacia atrás
    # le pasamos el valor del gradiente inicial a las funciones
    grad_back = softmax.back_prop(gradient, learn_rate) 
    grad_back = pool.back_prop(grad_back)
    grad_back = conv.back_prop(grad_back, learn_rate)

    # retornamos la pérdida y la precisión
    return loss, acc



# entrenamos a la CNN el número de épocas que veamos conveniente (4)
for epocas in range(4):
    print('//////////////////////////// Época %d //////////////////////////////'% (epocas +1))

    #shuffle the training data
    # 1500 imágenes se dividen en parches, cada parche tiene 100 imágenes
    shuffle_data = np.random.permutation(len(train_images))
    train_images = train_images[shuffle_data]
    train_labels = train_labels[shuffle_data]

    # entrenamiento de la red 
    loss = 0.0
    num_correct = 0

    for i, (im, label) in enumerate(zip(train_images, train_labels)):
        # por cada 100 iteraciones, hacemos 0 a la pérdida y el número de imágenes correctas
        if i % 100 == 0:
            print('%d/%d pasos: Pérdida promedio: %.3f Precisión: %d%%' %(i+1, num_test,loss/100, num_correct))
            loss = 0
            num_correct = 0

        # calculamos el entrenamiento de la red llamando a la función training_cnn  
        l1, accu = training_cnn(im, label)
        loss += l1
        num_correct +=accu


# Empleo de la CNN
print('///////////////// Probando...\n\t\t\t\t  Probando... gggg ////////////////////////')
loss = 0
num_correct = 0

for im, label in zip(test_images, test_labels):
    # utilizaos la progagación hacia adelante 
    _, l1, accu = cnn_forward_prop(im, label)
    loss += l1
    num_correct += accu

precision = (num_correct / num_test)*100
perdida = loss /num_test
print('Pérdida: ', perdida)
print(f'Precisión: {precision:.2f} %')

# Mostramos rostros de entrada y de prueba
"""
num_row = 3
num_col = 3
fig,axes = plt.subplots(num_row,num_col,figsize=(1.5*num_col,2*num_row))

i=0

for k in range(10):
	indice = np.round((4999*np.random.rand(1)+1),0) # modificar por número de datos -1
	#print(indice)
	numero_reconocido = index[:,int(indice)]
	#print(numero_reconocido)
	pixels = P[:,int(indice)].reshape(20,20).T
	plt.imshow(pixels,cmap="gray")
	plt.title("Reconocido: "+str(int(numero_reconocido)))
	

plt.show()

"""
