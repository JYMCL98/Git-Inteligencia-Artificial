# Universidad Autónoma Chapingo
# Departamento de Ingeniería Mecánica Agrícola
# Ingeniería Mecatrónica Agrícola
# Jym Emmanuel Cocotle Lara
# 7° 7
# ------Proyecto Final------
# ------CNN-----------------

# Librerías utilizadas
import pandas as pd
import numpy as np
import cv2 as cv

# Creamos una clase para una capa de convolución con filtros de 3x3
class Convolucion_3x3:
    def __init__(self, num_filtros):
        self.num_filtros = num_filtros
        # Dividimos por 9 para reducir la varianza de nuestros valores iniciales
        self.filters = np.random.randn(num_filtros, 3, 3) / 9
    
    # Generamos todas las regiones de imagen de 3x3 posibles.
    # La imagen es una matriz numérica de dos dimensiones.
    def itera_reg(self, image):
        h, w = image.shape
    
        for i in range(h - 2):
          for j in range(w - 2):
            im_region = image[i:(i + 3), j:(j + 3)]
            yield im_region, i, j
    
    # Realizamos un paso hacia adelante de la capa de convolucion usando los 
    # datos de entrada.
    # Donde la entrada es una matriz numérica de 2 dimensiones
    def forward_prop(self, input):
        self.last_input = input
    
        h, w = input.shape
        output = np.zeros((h - 2, w - 2, self.num_filtros))
    
        for im_region, i, j in self.itera_reg(input):
          output[i, j] = np.sum(im_region * self.filters, axis=(1, 2))
        
        # Se devuelve una matriz numérica de 3 dimensiones (h, w, num_filtros).
        return output
    
    # Se realiza una propagación hacia atras de la capa de convolución
    # Donde d_L_d_out es el gradiente de pérdida para las salidas de esta capa.
    # y learn_rate es un flotante.
    def backprop(self, d_L_d_out, learn_rate):
        d_L_d_filtros = np.zeros(self.filters.shape)
    
        for im_region, i, j in self.itera_reg(self.last_input):
          for f in range(self.num_filtros):
            d_L_d_filtros[f] += d_L_d_out[i, j, f] * im_region
    
        # Se actualizan los filtros
        self.filters -= learn_rate * d_L_d_filtros

        # No se devuleve ningún valor ya que usamos Convolucion_3x3 
        # como la primera capa en nuestra CNN.
        return None

# Creamos una clase denominada Maxpool 2, donde 2 hace referencia a 2x2 que 
# son las dimensiones que se tomaran para esta accion.
# Es decir, recorreremos cada una de nuestras 1439 imÃ¡genes de características 
# obtenidas anteriormente de 80x80 px de izquierda-derecha, arriba-abajo 
# pero en vez de tomar de a 1 pixel, tomaremos de 2 en 2
# e iremos preservando el valor más alto
class MaxPool_2:
    
    def itera_reg(self, image):
        h, w, _ = image.shape
        new_h = h // 2
        new_w = w // 2
    
        for i in range(new_h):
          for j in range(new_w):
            im_region = image[(i * 2):(i * 2 + 2), (j * 2):(j * 2 + 2)]
            yield im_region, i, j
    
    
    # Realizamos un pase hacia adelante de la capa maxpool usando los datos de entrada.
    def forward_prop(self, input):
        # La entrada es una matriz de 3 dimensiones (h, w, número de filtros)
        self.last_input = input
    
        h, w, num_filtros = input.shape
        output = np.zeros((h // 2, w // 2, num_filtros))
    
        for im_region, i, j in self.itera_reg(input):
            output[i, j] = np.amax(im_region, axis=(0, 1))
            
        # Retornamos una matriz de 3 dimensiones (h/2,w/2,número de filtros)
        return output
    
    # Realizamos un paso hacia atras de la capa maxpool.
    def backprop(self, d_L_d_out):
        d_L_d_input = np.zeros(self.last_input.shape)
    
        for im_region, i, j in self.itera_reg(self.last_input):
            h, w, f = im_region.shape
            amax = np.amax(im_region, axis=(0, 1))
    
        for i2 in range(h):
            for j2 in range(w):
                for f2 in range(f):
                    # Si este píxel era el valor máximo. copiamos su gradiente
                    if im_region[i2, j2, f2] == amax[f2]:
                        d_L_d_input[i * 2 + i2, j * 2 + j2, f2] = d_L_d_out[i, j, f2]
        return d_L_d_input

# Creamos la clase Softmax
class Softmax:

    def __init__(self, input_len, nodes):
        self.weights = np.random.randn(input_len, nodes) / input_len
        self.biases = np.zeros(nodes)

    def forward_prop(self, input):

        self.last_input_shape = input.shape
        input = input.flatten()
        self.last_input = input
        input_len, nodes = self.weights.shape
        totals = np.dot(input, self.weights) + self.biases
        self.last_totals = totals
        
        exp = np.exp(totals)
        # Retornamos una matriz de una dimensiÃ³n que contiene los valores de 
        # probabilidad respectivos.
        return exp / np.sum(exp, axis=0)

    def backprop(self, d_L_d_out, learn_rate):

        for i, gradient in enumerate(d_L_d_out):
            if gradient == 0:
                continue
    
            # Valores exponenciales
            t_exp = np.exp(self.last_totals)
    
            # Suma de todos los valores exponenciales
            S = np.sum(t_exp)
    
            # Gradiente de cada valor de salida out[i] contra los totales
            d_out_d_t = -t_exp[i] * t_exp / (S ** 2)
            d_out_d_t[i] = t_exp[i] * (S - t_exp[i]) / (S ** 2)
    
            # Gradiente de los valores totales contra ponderaciones, sesgos y entradas
            d_t_d_w = self.last_input
            d_t_d_b = 1
            d_t_d_inputs = self.weights
    
            # Gradiente de las pérdidas contra los valores totales
            d_L_d_t = gradient * d_out_d_t

            # radiente de las pérdidas contra ponderaciones, sesgos y entradas
            d_L_d_w = d_t_d_w[np.newaxis].T @ d_L_d_t[np.newaxis]
            d_L_d_b = d_L_d_t * d_t_d_b
            d_L_d_inputs = d_t_d_inputs @ d_L_d_t
    
            # Actualización de las ponderaciones y sesgos
            self.weights -= learn_rate * d_L_d_w
            self.biases -= learn_rate * d_L_d_b
          
            # Devolvemos el gradiente de pérdida para las entradas de esta capa
            return d_L_d_inputs.reshape(self.last_input_shape)

# Numero total de imagenes (recordando que es un array y la posición "0" cuenta)
total_img = 999
# Resolución de las imagenes utilizadas
resol = 80
# Ubicación del archivo
data = pd.read_csv('D:/Desktop/Data/Chayotes.csv')
data2 = np.array(data)
m, n = data2.shape

data3 = data.drop(data.columns[[0]], axis='columns')
data4 = np.array(data3)

train_images1 = data4.reshape(total_img,resol,resol)
train_images = np.uint8(train_images1)

data_train = data2[0:m].T
train_labels1 = data_train[0]
train_labels = np.uint8(train_labels1)

conv = Convolucion_3x3(8) # 8 filtros
pool = MaxPool_2()
softmax = Softmax(39*39*8, 5) # 13x13x8 -> 10


# Creamos una función forward propagation (propagación hacia adelante)
# Completa una propagación hacia adelante de la CNN y calculamos la precisión y
# pérdida de entropí­a cruzada
def forward_prop(image, label):
    # Transformamos la imagen de [0, 255] a [-0,5, 0,5] para que sea más fácil 
    # trabajar con ella.
    out = conv.forward_prop((image / 255) - 0.5)
    out = pool.forward_prop(out)
    out = softmax.forward_prop(out)

    # Calculamos la precisión (acc) y la pérdida (loss)
    loss = -np.log(out[label])
    acc = 1 if np.argmax(out) == label else 0

    return out, loss, acc

# Creamos una función para realizar un paso de entrenamiento donde:
# im = la imagen expresada como una matriz de 2 dimensiones
# label = el número que identifica la imagen (1 = chayote, 0 = no hay chayote)
# alfa = la tasa de aprendizaje (0.005)
def train(im, label, alfa=.005):
    # propagación hacia adelante
    out, loss, acc = forward_prop(im, label)
    
    # Calculamos el gradiente inicial
    gradient = np.zeros(10)
    gradient[label] = -1 / out[label]
    
    # Backpropagation (propagación hacia atras)
    gradient = softmax.backprop(gradient, alfa)
    gradient = pool.backprop(gradient)
    gradient = conv.backprop(gradient, alfa)
  
    # Retornamos los valores de pérdida y precisión
    return loss, acc

print('--- Iniciando la CNN ---')

# Entrenamiento de la CNN con 3 épocas
for epoca in range(3):
    # Mostramos la época en la que nos encontramos
    print('--- Epoca %d ---' % (epoca + 1))

    # Mezclamos los datos de entrenamiento
    permutation = np.random.permutation(len(train_images))
    train_images = train_images[permutation]
    train_labels = train_labels[permutation]

    loss = 0.0
    num_correct = 0
    
    for i, (im, label) in enumerate(zip(train_images, train_labels)):
        if i % 100 == 0:
            print(
                '[Paso %d] 100 pasos dados| pérdida %.3f | Exactitud: %d%%' %
                (i + 1, loss / 100, num_correct)
                )
            loss = 0
            num_correct = 0

        l, acc = train(im, label)
        loss += l
        num_correct += acc

# Inicializamos el video
video=cv.VideoCapture(1) 

if not video.isOpened():
    print("No se puede abrir la camara")
    exit()

# Bucle de la camara
while True:
    # Captura trama a trama
    ret,frame = video.read()
    # Escala de grises
    escala_grises = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
    
    if not ret:
        print("No se puede recibir el frame, el video ha terminado")
        break

    fotog = np.array(cv.resize(escala_grises, (80,80))).reshape(-1,80,80)
    test_images = fotog
    test_labels = np.array([0])
    
    loss = 0
    num_correct = 0
    n = 0

    for i,(im, label) in enumerate(zip(test_images, test_labels)):
        _, l, acc = forward_prop(im, label)
        loss += l
        num_correct += acc
      
        if acc == 1:
            print("Si hay Chayote")
            fruta ="Chayote"
            x_imagen = 100
            color = (0,255,0)
        else:
            print("No hay Chayote")
            fruta = "No hay"
            x_imagen = 50
            color = (0,0,255)

    n+=1
    teste = len(test_images)
    escala_grises = cv.putText(frame,fruta,(x_imagen,50),cv.FONT_HERSHEY_COMPLEX, 1.5,color,1)
    cv.imshow("Camara",escala_grises)
    
    if cv.waitKey(1)==ord('q'):
        break
    
    print('pérdida: ',loss/teste)
    print('Prediccion: ',num_correct)