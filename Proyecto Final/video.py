import numpy as np
import pandas as pd
import cv2 as cv

video=cv.VideoCapture(0) 

# Ruta de la base de datos
data = pd.read_csv('D:/Desktop/Data/Chayote.csv')
data = np.array(data)
m, n = data.shape
np.random.shuffle(data)

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


def init_params():
    W1 = np.random.rand(2, 6400) - 0.5
    b1 = np.random.rand(2, 1) - 0.5
    W2 = np.random.rand(2, 2) - 0.5
    b2 = np.random.rand(2, 1) - 0.5
    return W1, b1, W2, b2

def ReLU(Z):
    return np.maximum(Z, 0)

def softmax(Z):
    A = np.exp(Z) / np.sum(np.exp(Z))
    return A
    
def forward_prop(W1, b1, W2, b2, X):
    Z1 = W1.dot(X) + b1
    A1 = ReLU(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2

def ReLU_deriv(Z):
    return Z > 0

def one_hot(Y):
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    one_hot_Y[np.arange(Y.size), Y] = 1
    one_hot_Y = one_hot_Y.T
    return one_hot_Y

def backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y):
    one_hot_Y = one_hot(Y)
    dZ2 = A2 - one_hot_Y
    dW2 = 1 / m * dZ2.dot(A1.T)
    db2 = 1 / m * np.sum(dZ2)
    dZ1 = W2.T.dot(dZ2) * ReLU_deriv(Z1)
    dW1 = 1 / m * dZ1.dot(X.T)
    db1 = 1 / m * np.sum(dZ1)
    return dW1, db1, dW2, db2

def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * db1    
    W2 = W2 - alpha * dW2  
    b2 = b2 - alpha * db2    
    return W1, b1, W2, b2

def get_predictions(A2):
    return np.argmax(A2, 0)

def get_accuracy(predictions, Y):
    print(predictions, Y)
    return np.sum(predictions == Y) / Y.size

def gradient_descent(X, Y, alpha, iterations):
    W1, b1, W2, b2 = init_params()
    for i in range(iterations):
        Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X)
        dW1, db1, dW2, db2 = backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y)
        W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)
        if i % 10 == 0:
            print("Iteracion: ",i)
            predictions = get_predictions(A2)
            print("Precision: ", get_accuracy(predictions, Y))
    return W1, b1, W2, b2

W1, b1, W2, b2 = gradient_descent(X_train, Y_train, 0.1, 10000)

def make_predictions(X, W1, b1, W2, b2):
    _, _, _, A2 = forward_prop(W1, b1, W2, b2, X)
    predictions = get_predictions(A2)
    return predictions


dev_predictions = make_predictions(X_dev, W1, b1, W2, b2)
get_accuracy(dev_predictions, Y_dev)



if not video.isOpened():
    print("No se puede abrir la camara")
    exit()

# Bucle de la camara
while True:
    # Captura trama a trama
    ret,frame = video.read()
            
    if not ret:
        print("No se puede recibir el frame, el video ha terminado")
        break
            
    # Escala de grises
    escala_grises = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
    
    
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