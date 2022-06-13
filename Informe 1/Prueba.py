# Universidad Autónoma Chapingo
# 


# Importamos las librerías
import numpy as np
from numpy import asarray
import cv2
import matplotlib.pyplot as plt
import os


# Directorio donde se encuentran las imagenes que se ocuparán para entrenar y para probar a la CNN
directorio1 = "Data/training"  # Imagenes para entrenar
directorio2 = "Data/detection"  # Imagenes para probar"


# Convolución
class Conv_op:
# Definimos los filtros requeridos para la convolución
	def __init__(self, num_filters, filter_size):
		self.num_filters = num_filters
		self.filter_size = filter_size
		# Inicializamos aleatoriamente los valores de los filtros
		self.conv_filter = np.random.randn(num_filters,filter_size,filter_size)/(filter_size*filter_size)

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
		conv_out = np.zeros((height-self.filter_size+1,width-self.filter_size+1,self.num_filters))
		for image_path, i,j in self.image_region(image):
			# Multiplicamos el filtro por los parches de la imagen
			conv_out[i,j]= np.sum(image_path*self.conv_filter,axis=(1,2))

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
		new_height = image.shape[0]//self.filter_size
		new_width = image.shape[1]//self.filter_size
		self.image = image
		# Extraemos los parches de la imagen nueva que se calcula en la función de arriba
		for i in range(new_height):
			for j in range(new_width):
				image_patch = image[(i*self.filter_size):(i*self.filter_size+self.filter_size),(j*self.filter_size):(j*self.filter_size+self.filter_size)]
				yield image_patch, i, j

	# Módulo de propagación hacia adelante
	def forward_prop(self,image):
		height, width, num_filters = image.shape
		output = np.zeros((height//self.filter_size, width//self.filter_size,num_filters))
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
							dL_dmax_pool[i*self.filter_size+i1,j*self.filter_size + j1,k1]=dL_dout[i,j,k1]
			return dL_dmax_pool

# Operación Softmax
# la función de Softmax se encarga de pasar a probabilidad (entre 0 y1) a las neuronas de salida
class Softmax:
	# definimos los pesos y las bias de
	def __init__(self, input_node, softmax_node):
		self.weight = np.random.randn(input_node,softmax_node)/input_node
		self.bias = np.zeros(softmax_node)

	# multiplicamos los pesos por las biases y generando salidasdesde las capas ocultas
	# compactamos el cubo
	def forward_prop(self, image):
		self.orig_im_shape = image.shape #used in backprop
		image_modified = image.flatten()
		self.modified_input = image_modified #to be used in backprop
		output_val = np.dot(image_modified,self.weight)+self.bias
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

			# Gradiente con especto a la salida Z
			dy_dz = -transformation_eq[i]*transformation_eq/(S_total**2)
			dy_dz[i] = transformation_eq[i]*(S_total-transformation_eq[i])/(S_total**2)

			# Gradiente del total con pesos y entradas
			dz_dw = self.modified_input
			dz_db = 1
			dz_d_inp = self.weight

			# Gradiente total contra el perdido
			dL_dz = grad*dy_dz

			# Gradiente de pérdida contra pesos, biases y entradas
			dL_dw = dz_dw[np.newaxis].T@dL_dz[np.newaxis]
			dL_db = dL_dz*dz_db
			dL_d_inp = dz_d_inp@dL_dz

		# Actualizamos los pesos y las biases
		self.weight -= learning_rate*dL_dw
		self.bias -= learning_rate*dL_db

		return dL_d_inp.reshape(self.orig_im_shape)


def dataset_function(directorio,ancho,alto):
	dataset = []
	label = ["caras","no_caras"]
	for categoria in label:
		rostro = label.index(categoria)
		ruta_imagen = os.path.join(directorio,categoria)

		for file_name in os.listdir(ruta_imagen):
			img = cv2.imread(os.path.join(ruta_imagen,file_name),cv2.IMREAD_GRAYSCALE)
			#img = cv2.resize(img, (ancho,alto))
			dataset.append([img,rostro])
	X = []
	Y = []
	for caracteristicas,label in dataset:
		X.append(caracteristicas)
		Y.append(label)

	X = np.array(X).reshape(-1,ancho,alto)
	Y = np.array(Y)

	return X,Y

(X_train,y_train) = dataset_function(directorio1, 125,125) #resize

(X_test,y_test) = dataset_function(directorio2, 125,125)

# imás del entrenamiento (número de imágenes que tengamos)
train_images = X_train[0:2041]  # suma total de caras y no caras
train_labels = y_train[0:2041]

# imágenes de la prueba (número de imágenes que tengamos)
test_images = X_test[0:20]
test_labels = y_test[0:20]

num_test = len(test_images)
num_train = len(train_images)

# 8 filtros de 3x3
conv = Conv_op(8,3)
# operacion de encontrar el número mayor
pool = Max_Pool(2)
# (80 - tamaño de filtro + 1/maxpool)*, numero de imagenes a compactar
softmax = Softmax(39*39*8,5)

# empleando la propagación hacia adelante en una red neuronalconvolucional
# le pasamos el número de imágenes y obtenemos una salida como 10clases
def cnn_forward_prop(image, label):
	# alimentamos a la imagen con la operación de convolución haciaadelante
	out_p = conv.forward_prop((image/255)-0.5)
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
	# Propagación hacia atras de la salida, la pérdida y la precisión
	out, loss, acc = cnn_forward_prop(image, label)

	# Se calcula el gradiente inicial
	gradient = np.zeros(10)
	gradient[label] = -1 / out[label]

	# Propagación hacia atras
	# le pasamos el valor del gradiente inicial a las funciones
	grad_back = softmax.back_prop(gradient, learn_rate)
	grad_back = pool.back_prop(grad_back)
	grad_back = conv.back_prop(grad_back, learn_rate)

	# retornamos la pérdida y la precisión
	return loss, acc

# entrenamos a la CNN el número de épocas que veamos conveniente (#épocas)
for epocas in range(1):
	print("Época numero: %d "% (epocas +1))

	# 1500 imágenes se dividen en parches, cada parche tiene 100 imágenes
	# El número de imagenes totales y de cada parche puede ser cambiado
	shuffle_data = np.random.permutation(len(train_images))
	train_images = train_images[shuffle_data]
	train_labels = train_labels[shuffle_data]

	# entrenamiento de la red
	loss = 0.0
	num_correct = 0

	for i, (im, label) in enumerate(zip(train_images, train_labels)):
		# por cada 100 iteraciones, hacemos 0 a la pérdida y el número de imágenes correctas
		if i % 100 == 0:
			print('%d/%d pasos: Pérdida promedio: %.3f Precisión: %d%%' %(i+1, num_train,loss/100, num_correct))
			loss = 0
			num_correct = 0

		# Se calcula el entrenamiento de la red
		l1, accu = training_cnn(im, label)
		loss += l1
		num_correct +=accu

# Empleo de la CNN
print("Puesta a prueba")

loss = 0 # Variable para calcular la pérdida
num_correct = 0 # Variable para calcular el número de imágenes correctas
n = 0

for i,(im, label) in enumerate(zip(test_images, test_labels)):
	# Se calcula la propagación hacia adelante
	__, l1, accu = cnn_forward_prop(im, label)
	loss += l1
	num_correct += accu

	if (accu == 1):
		plt.imshow(X_test[n],cmap="gray")
		plt.title(f"Sí existe un rostro")
		plt.show()

	else:
		plt.imshow(X_test[n],cmap="gray")
		plt.title(f"No existe un rostro")
		plt.show()

	n+=1

precision = (num_correct / num_test)*100
perdida = loss /num_test
print('Pérdida: ', perdida)
print(f'Precisión: {precision:.2f} %')