import mnist
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import pickle
import cv2 as cv


class Conv3x3:
  # A Convolution layer using 3x3 filters.

  def __init__(self, num_filters):
    self.num_filters = num_filters

    # filters is a 3d array with dimensions (num_filters, 3, 3)
    # We divide by 9 to reduce the variance of our initial values
    self.filters = np.random.randn(num_filters, 3, 3) / 9

  def iterate_regions(self, image):
    '''
    Generates all possible 3x3 image regions using valid padding.
    - image is a 2d numpy array.
    '''
    h, w = image.shape

    for i in range(h - 2):
      for j in range(w - 2):
        im_region = image[i:(i + 3), j:(j + 3)]
        yield im_region, i, j

  def forward(self, input):
    '''
    Performs a forward pass of the conv layer using the given input.
    Returns a 3d numpy array with dimensions (h, w, num_filters).
    - input is a 2d numpy array
    '''
    self.last_input = input

    h, w = input.shape
    output = np.zeros((h - 2, w - 2, self.num_filters))

    for im_region, i, j in self.iterate_regions(input):
      output[i, j] = np.sum(im_region * self.filters, axis=(1, 2))

    return output

  def backprop(self, d_L_d_out, learn_rate):
    '''
    Performs a backward pass of the conv layer.
    - d_L_d_out is the loss gradient for this layer's outputs.
    - learn_rate is a float.
    '''
    d_L_d_filters = np.zeros(self.filters.shape)

    for im_region, i, j in self.iterate_regions(self.last_input):
      for f in range(self.num_filters):
        d_L_d_filters[f] += d_L_d_out[i, j, f] * im_region

    # Update filters
    self.filters -= learn_rate * d_L_d_filters

    # We aren't returning anything here since we use Conv3x3 as the first layer in our CNN.
    # Otherwise, we'd need to return the loss gradient for this layer's inputs, just like every
    # other layer in our CNN.
    return None



class MaxPool2:
  # A Max Pooling layer using a pool size of 2.

  def iterate_regions(self, image):
    '''
    Generates non-overlapping 2x2 image regions to pool over.
    - image is a 2d numpy array
    '''
    h, w, _ = image.shape
    new_h = h // 2
    new_w = w // 2

    for i in range(new_h):
      for j in range(new_w):
        im_region = image[(i * 2):(i * 2 + 2), (j * 2):(j * 2 + 2)]
        yield im_region, i, j

  def forward(self, input):
    '''
    Performs a forward pass of the maxpool layer using the given input.
    Returns a 3d numpy array with dimensions (h / 2, w / 2, num_filters).
    - input is a 3d numpy array with dimensions (h, w, num_filters)
    '''
    self.last_input = input

    h, w, num_filters = input.shape
    output = np.zeros((h // 2, w // 2, num_filters))

    for im_region, i, j in self.iterate_regions(input):
      output[i, j] = np.amax(im_region, axis=(0, 1))

    return output

  def backprop(self, d_L_d_out):
    '''
    Performs a backward pass of the maxpool layer.
    Returns the loss gradient for this layer's inputs.
    - d_L_d_out is the loss gradient for this layer's outputs.
    '''
    d_L_d_input = np.zeros(self.last_input.shape)

    for im_region, i, j in self.iterate_regions(self.last_input):
      h, w, f = im_region.shape
      amax = np.amax(im_region, axis=(0, 1))

      for i2 in range(h):
        for j2 in range(w):
          for f2 in range(f):
            # If this pixel was the max value, copy the gradient to it.
            if im_region[i2, j2, f2] == amax[f2]:
              d_L_d_input[i * 2 + i2, j * 2 + j2, f2] = d_L_d_out[i, j, f2]

    return d_L_d_input





class Softmax:
  # A standard fully-connected layer with softmax activation.

  def __init__(self, input_len, nodes):
    # We divide by input_len to reduce the variance of our initial values
    self.weights = np.random.randn(input_len, nodes) / input_len
    self.biases = np.zeros(nodes)

  def forward(self, input):
    '''
    Performs a forward pass of the softmax layer using the given input.
    Returns a 1d numpy array containing the respective probability values.
    - input can be any array with any dimensions.
    '''
    self.last_input_shape = input.shape

    input = input.flatten()
    self.last_input = input

    input_len, nodes = self.weights.shape

    totals = np.dot(input, self.weights) + self.biases
    self.last_totals = totals

    exp = np.exp(totals)
    return exp / np.sum(exp, axis=0)

  def backprop(self, d_L_d_out, learn_rate):
    '''
    Performs a backward pass of the softmax layer.
    Returns the loss gradient for this layer's inputs.
    - d_L_d_out is the loss gradient for this layer's outputs.
    - learn_rate is a float.
    '''
    # We know only 1 element of d_L_d_out will be nonzero
    for i, gradient in enumerate(d_L_d_out):
      if gradient == 0:
        continue

      # e^totals
      t_exp = np.exp(self.last_totals)

      # Sum of all e^totals
      S = np.sum(t_exp)

      # Gradients of out[i] against totals
      d_out_d_t = -t_exp[i] * t_exp / (S ** 2)
      d_out_d_t[i] = t_exp[i] * (S - t_exp[i]) / (S ** 2)

      # Gradients of totals against weights/biases/input
      d_t_d_w = self.last_input
      d_t_d_b = 1
      d_t_d_inputs = self.weights

      # Gradients of loss against totals
      d_L_d_t = gradient * d_out_d_t

      # Gradients of loss against weights/biases/input
      d_L_d_w = d_t_d_w[np.newaxis].T @ d_L_d_t[np.newaxis]
      d_L_d_b = d_L_d_t * d_t_d_b
      d_L_d_inputs = d_t_d_inputs @ d_L_d_t

      # Update weights / biases
      self.weights -= learn_rate * d_L_d_w
      self.biases -= learn_rate * d_L_d_b

      return d_L_d_inputs.reshape(self.last_input_shape)






# We only use the first 1k examples of each set in the interest of time.
# Feel free to change this if you want.

#data = np.array(pd.read_csv('D:/Desktop/Data/Chayote.csv'))
data = pd.read_csv('D:/Desktop/Data/Chayote.csv')
data = np.array(data)
m, n = data.shape
#np.random.shuffle(data) # shuffle before splitting into dev and training sets


data_train = data[0:m].T
Y_train = data_train[0]
X_train = data_train[1:n]


train_images = X_train
train_labels = Y_train

'''

data_train = data[0:m].T

X_train = data_train[1:n]
train_images = X_train.reshape(1493,80,80)
train_labels = data_train[0]
test_images = train_images
test_labels = train_labels

'''


conv = Conv3x3(8)                  # 28x28x1 -> 26x26x8
pool = MaxPool2()                  # 26x26x8 -> 13x13x8
softmax = Softmax(39*39*8, 5) # 13x13x8 -> 10


#tamanio = 494
'''
data_dev = data[0:tamanio].T
Y_dev = data_dev[0]
X_dev = data_dev[1:n]
X_dev = X_dev / 255.

data_train = data[tamanio:m].T
Y_train = data_train[0]
X_train = data_train[1:n]
X_train = X_train / 255.
_,m_train = X_train.shape



train_images = mnist.train_images()[:1000]
train_labels = mnist.train_labels()[:1000]
test_images = mnist.test_images()[:1000]
test_labels = mnist.test_labels()[:1000]
'''


def forward(image, label):
  '''
  Completes a forward pass of the CNN and calculates the accuracy and
  cross-entropy loss.
  - image is a 2d numpy array
  - label is a digit
  '''
  # We transform the image from [0, 255] to [-0.5, 0.5] to make it easier
  # to work with. This is standard practice.
  out = conv.forward((image / 255) - 0.5)
  out = pool.forward(out)
  out = softmax.forward(out)

  # Calculate cross-entropy loss and accuracy. np.log() is the natural log.
  loss = -np.log(out[label])
  acc = 1 if np.argmax(out) == label else 0

  return out, loss, acc

def train(im, label, lr=.005):
  '''
  Completes a full training step on the given image and label.
  Returns the cross-entropy loss and accuracy.
  - image is a 2d numpy array
  - label is a digit
  - lr is the learning rate
  '''
  # Forward
  out, loss, acc = forward(im, label)

  # Calculate initial gradient
  gradient = np.zeros(10)
  gradient[label] = -1 / out[label]

  # Backprop
  gradient = softmax.backprop(gradient, lr)
  gradient = pool.backprop(gradient)
  gradient = conv.backprop(gradient, lr)

  return loss, acc

print('--- Iniciando la CNN ---')

# Train the CNN for 3 epochs
for epoca in range(1):
  print('--- Epoca %d ---' % (epoca + 1))

  # Shuffle the training data
  
  permutation = np.random.permutation(len(train_images))
  train_images = train_images[permutation]
  train_labels = train_labels[permutation]

  # Train!
  loss = 0.0
  num_correct = 0
  
  for i, (im, label) in enumerate(zip(train_images, train_labels)):
    if i % 100 == 0:
      print(
        '[Paso %d] 100 pasos dados| Perdida %.3f | Exactitud: %d%%' %
        (i + 1, loss / 100, num_correct)
      )
      loss = 0
      num_correct = 0

    l, acc = train(im, label)
    loss += l
    num_correct += acc


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
    
    img2 = cv.resize(escala_grises,(80,80))
    X = (img2)
    X = np.array(X).reshape(-1,80,80)
    Y = np.array([0])
    test_images = X
    test_labels = Y
    test_images = X[:1]
    test_labels = Y[:1]
    loss = 0
    num_correct = 0
    n = 0
    
    for i,(im, label) in enumerate(zip(test_images, test_labels)):  #zip
        _, l, acc = forward(im, label)
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
    
    print('Perdida: ',loss/teste)
    print('Precision: ',num_correct/teste)
    
'''  
    fotog = np.array((cv.resize(escala_grises, (80,80))))
    fotog2 = fotog.reshape(-1,1)
    
    XA = np.array(fotog2, dtype='int64')
    P = fotog2/255
    predic = make_predictions(P, W1, b1, W2, b2)
    print("Prediccion: ", predic)    
    
    if predic == 1:
        fruta ="Chayote"
        x_imagen = 100
        color = (0,255,0)
    elif predic == 0:
        fruta = "No hay"
        x_imagen = 50
        color = (0,0,255)
    
    escala_grises = cv.putText(frame,fruta,(x_imagen,50),cv.FONT_HERSHEY_COMPLEX, 1.5,color,1)
    
    cv.imshow("Camara",escala_grises)
    
    if cv.waitKey(1)==ord('q'):
        break

cv.destroyAllWindows()





# Test the CNN
print('\n--- Probando con imagenes ya tomadas ---')
loss = 0
num_correct = 0
for i,(im, label) in enumerate(zip(test_images, test_labels)):  #zip
  _, l, acc = forward(im, label)
  loss += l
  num_correct += acc

# Image.fromarray(array)


num_tests = len(test_images)
print('Perdida:', loss / num_tests)
print('Exactitud:', num_correct / num_tests)

#pil_image=Image.fromarray()
#pil_image.show()

'''