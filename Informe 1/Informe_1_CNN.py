# Universidad Autónoma Chapingo
# Ingeniería Mecatrónica Agrícola
# Jym Emmanuel Cocotle Lara 7° 7

# Importamos las librerías a ocupar
import os
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt


# Directorio donde se encuentran las imagenes que se ocuparán para entrenar y para probar a la CNN
Data_train = "Data/training"  # Imagenes para entrenar
Data_det = "Data/detection"  # Imagenes para probar"


# Convolución
class Convolucion:
    # Filtros requeridos para la convolución
    def __init__(self, num_filt, tam_filt):
        self.num_filt = num_filt
        self.tam_filt = tam_filt
        self.conv_filt = np.random.randn(num_filt, tam_filt, tam_filt)/(tam_filt * tam_filt)
    
    # Obtenemos los parches de ruta de las imagenes
    def image_region(self,image):
        height, width = image.shape
        self.image = image
        # Extraemos las medidas de los parches
        for j in range(height - self.tam_filt + 1):
            for k in range(width - self.tam_filt +1):
                # Se crea una imagen a partir de los valores de los parches.
                image_patch = image[j:(j+self.tam_filt),k:(k+self.tam_filt)]
                yield image_patch, j, k 

    # Propagación hacia adelante
    def forward_prop(self,image):
        height, width = image.shape
        sal_conv = np.zeros((height - self.tam_filt + 1, width - self.tam_filt+1, self.num_filt))
        for image_path, i,j in self.image_region(image):
            # Multiplicación de los filtros con el parche de la imagen
            sal_conv[i,j]= np.sum(image_path * self.conv_filt, axis=(1,2))
        return sal_conv
    
    # Propagación hacia atrás
    def back_prop(self, dL_dout, learning_rate):
        dL_dF_params = np.zeros(self.conv_filt.shape)
        for image_patch, i,j in self.image_region(self.image):
            for k in range(self.num_filt):
                dL_dF_params[k] += image_patch*dL_dout[i,j,k]

    # Actualización de los parámetros del filtro
        self.conv_filt -= learning_rate*dL_dF_params
        return dL_dF_params


# Max-Pooling
class Max_Pool:
    # Definición del tamaño del filtro
    def __init__(self, tam_filt):
        self.tam_filt = tam_filt
    
    # Reducción del tamaño de la imagen
    def image_region(self, image):
        # Nueva altura
        new_height = image.shape[0] // self.tam_filt
        # Nuevo ancho
        new_width = image.shape[1] // self.tam_filt
        self.image = image
        # Extracción de los parches de la imagen nueva que se calcula en la función de convolución
        for i in range(new_height):
            for j in range(new_width):
                image_patch = image[(i*self.tam_filt):(i*self.tam_filt + self.tam_filt), (j*self.tam_filt):(j*self.tam_filt + self.tam_filt)]
                yield image_patch, i, j

    # Propagacion hacia adelante
    def forward_prop(self,image):
        height, width, num_filt = image.shape
        output = np.zeros((height // self.tam_filt, width // self.tam_filt, num_filt))

        for image_patch, i, j in self.image_region(image):
            output[i,j] = np.amax(image_patch, axis = (0,1))

        return output

    # Propagación hacia atrás
    def back_prop(self, dL_dout):
        dL_dmax_pool = np.zeros(self.image.shape)
        for image_patch, i, j in self.image_region(self.image):
            height, width, num_filt = image_patch.shape
            maximum_val = np.amax(image_patch,axis = (0,1))

            for i1 in range(height):
                for j1 in range(width):
                    for k1 in range(num_filt):
                        if image_patch[i1,j1,k1] == maximum_val[k1]:
                            dL_dmax_pool[i*self.tam_filt + i1, j*self.tam_filt + j1, k1]=dL_dout[i,j,k1]
            return dL_dmax_pool

# Softmax
class Softmax:

    def __init__(self, input_node, softmax_node):
        # Se inicializa el peso a partir de valores aleatorios
        self.weight = np.random.randn(input_node, softmax_node)/input_node
        # Se crea un sesgo inicial a partir de una matriz de ceros
        self.bias = np.zeros(softmax_node)

    # Se multiplican los pesos por las bases
    def forward_prop(self, image):
        self.orig_im_shape = image.shape
        image_modified = image.flatten()
        self.modified_input = image_modified
        output_val = np.dot(image_modified, self.weight) + self.bias
        self.out = output_val
        exp_out = np.exp(output_val)
        # Transformación de la salida en las salidas probables dadas
        return exp_out/np.sum(exp_out, axis=0)

    # Propagación hacia atrás
    def back_prop(self, dL_dout, learning_rate):
        for i, grad in enumerate(dL_dout):
            if grad ==0:
                continue
            # 
            transformation_eq = np.exp(self.out)
            S_total = np.sum(transformation_eq)

            # Gradiente con especto a la salida Z
            dy_dz = -transformation_eq[i]*transformation_eq / (S_total **2)
            dy_dz[i] = transformation_eq[i]*(S_total -  transformation_eq[i]) / (S_total **2)

            # Gradiente del total con pesos y entradas
            dz_dw =  self.modified_input
            dz_db = 1
            dz_d_inp = self.weight

            # Gradiente total contra el perdido
            dL_dz = grad * dy_dz

            # Gradiente de pérdida contra pesos, bases y entradas
            dL_dw = dz_dw[np.newaxis].T @ dL_dz[np.newaxis]
            dL_db = dL_dz * dz_db
            dL_d_inp = dz_d_inp @ dL_dz

      # Actualizamos los pesos y las bases
        self.weight -= learning_rate *dL_dw
        self.bias -= learning_rate * dL_db

        return dL_d_inp.reshape(self.orig_im_shape)


def dataset_function(directorio,ancho,alto):
    dataset = []
    label = ["real","no_real"]
    for categoria in label:
        rostro = label.index(categoria)
        ruta_imagen = os.path.join(directorio,categoria)
            
        for file_name in os.listdir(ruta_imagen):
            img = cv2.imread(os.path.join(ruta_imagen,file_name),cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (ancho,alto))
            dataset.append([img,rostro])
    X = []
    Y = []
    for caracteristicas,label in dataset:
        X.append(caracteristicas)
        Y.append(label)

    X = np.array(X).reshape(-1,ancho,alto)
    Y = np.array(Y)

    # Retornamos los valores X,Y
    return X,Y


data = pd.read_csv('D:/Desktop/Data/Chayote.csv')
data = np.array(data)
m, n = data.shape
np.random.shuffle(data) # shuffle before splitting into dev and training sets

tamanio = 494

data_dev = data[0:tamanio].T
Y_dev = data_dev[0]
X_dev = data_dev[1:n]
X_dev = X_dev / 255.

data_train = data[tamanio:m].T
Y_train = data_train[0]
X_train = data_train[1:n]
X_train = X_train / 255.
_,m_train = X_train.shape



(X_train,y_train) = dataset_function(Data_train, 80,80) #resize

(X_test,y_test) = dataset_function(Data_det, 80,80)


train_images = X_train[0:1400]
train_labels = y_train[0:1400]


test_images = X_test[0:20]
test_labels = y_test[0:20]


num_test = len(test_images)
num_train = len(train_images)

# 8 filtros de 3x3
conv = Convolucion(8,3)
# Operación de encontrar el número mayor
pool = Max_Pool(2)
# (80 - tamaño de filtro + 1/maxpool)*, numero de imagenes a compactar
softmax = Softmax(39*39*8,5)



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
    # Se calcula la salida, la pérdida y la precisión 
    out, loss, acc = cnn_forward_prop(image, label)

    # Se calcula el gradiente inicial
    gradient = np.zeros(10)
    gradient[label] = -1 / out[label]

    # propagación hacia atrás
    # le pasamos el valor del gradiente inicial a las funciones
    grad_back = softmax.back_prop(gradient, learn_rate) 
    grad_back = pool.back_prop(grad_back)
    grad_back = conv.back_prop(grad_back, learn_rate)

    return loss, acc

# Se entrena a la CNN de acuerdo con el número de épocas que se quieran

for epocas in range(1):
    print("Epoca numero: %d "% (epocas +1))

    # 1400 imágenes se dividen en parches y cada parche tiene 100 imágenes
    shuffle_data = np.random.permutation(len(train_images))
    train_images = train_images[shuffle_data]
    train_labels = train_labels[shuffle_data]

    loss = 0.0
    num_correct = 0

    for i, (im, label) in enumerate(zip(train_images, train_labels)):
        # 
        if i % 100 == 0:

            print('No. pasos: %d/%d Pérdida promedio: %.3f Precisión: %d%%' %(i+1, num_train,loss/100, num_correct))
            loss = 0
            num_correct = 0

        # Se calcula el entrenamiento de la red llamando a la función antes creada
        l1, accu = training_cnn(im, label)
        loss += l1
        num_correct +=accu

# Puesta a prueba de la CNN
print("Puesta a prueba...")
loss = 0 # Variable para mostrar la perdida
num_correct = 0 # Variable conocer el número de imagenes correctas
n = 0 # Identificador del número de imagen



for i,(im, label) in enumerate(zip(test_images, test_labels)):
    __, l1, accu = cnn_forward_prop(im, label)
    loss += l1
    num_correct += accu
     
    if (accu == 1):
        plt.imshow(X_test[n],cmap="gray")
        plt.title("Sí existe un rostro")
        plt.show()

    else:
        plt.imshow(X_test[n],cmap="gray")
        plt.title("No existe un rostro")
        plt.show()
    
    n+=1

# Cálculo de la precisión final
precision = (num_correct / num_test)*100
# cálculo de la perdida final
perdida = loss /num_test
# Se visualiza la perdida y la precisión
print("Precisión:", precision)
print("\nPérdida: ", perdida)

