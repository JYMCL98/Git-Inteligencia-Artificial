# Universidad Autónoma Chapingo
# Departamento de Ingeniería Mecánica Agrícola
# Ingeniería Mecatrónica Agrícola
# Jym Emmanuel Cocotle Lara
# 7° 7


import numpy as np
import matplotlib.pyplot as plt

image_size = 28 # width and length
no_of_different_labels = 10 #  i.e. 0, 1, 2, 3, ..., 9
image_pixels = image_size * image_size
data_path = "Data/"
train_data = np.loadtxt(data_path + "mnist_train.csv", delimiter=",")
test_data = np.loadtxt(data_path + "mnist_test.csv",delimiter=",") 
test_data[:10]
print(test_data[:10])

num_row = 2
num_col = 5# plot images
fig, axes = plt.subplots(num_row, num_col, figsize=(1.5*num_col,2*num_row))
i=0
for data in test_data[:10]:  

    label = data[0]

    pixels = data[1:]

    pixels = np.array(pixels, dtype='uint8')
 
    # Reshape the array into 28 x 28 array (2-dimensional array)
    pixels = pixels.reshape((28, 28))
 
    ax = axes[i//num_col, i%num_col]
    ax.imshow(pixels, cmap='gray')
    ax.set_title('Número: {label}'.format(label=label)) 
    plt.tight_layout()
    plt.pause(0.05)
 

    i=i+1

plt.show()
    # break # This stops the loop, I just want to see one
