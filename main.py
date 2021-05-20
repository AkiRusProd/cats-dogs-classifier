import sys
from PyQt5 import QtCore, QtGui, QtWidgets
from QtGen import Ui_MainWindow
from tqdm import tqdm
from numba import njit
import numpy as np
import cv2



app = QtWidgets.QApplication(sys.argv)
MainWindow = QtWidgets.QMainWindow()
ui = Ui_MainWindow()
ui.setupUi(MainWindow)
MainWindow.show()

image_size = 128

RGB_kernels_size = 3
RGB_kernels_count = 64
conv_chann_size = image_size - RGB_kernels_size + 1  
pool_part_size1 = 2
pool_size1 = conv_chann_size // pool_part_size1

kernels_2layer_size = 4
kernels_2layer_count = 192
conv_layer2_size = pool_size1 - kernels_2layer_size + 1
pool_part_size2 = 2
pool_size2 = conv_layer2_size // pool_part_size2

kernels_3layer_size = 3
kernels_3layer_count = 384
conv_layer3_size = pool_size2 - kernels_3layer_size + 1
pool_part_size3 = 2
pool_size3 = conv_layer3_size // pool_part_size3

output_size = 2

flatten_conv_size = kernels_3layer_count * pool_size3 * pool_size3

learning_rate = 0.005

with open('weights/R_conv_layer.csv','r') as f:
    R_chann_weights=np.set_printoptions(precision = 30, suppress = True)
    R_chann_weights= np.genfromtxt(f, delimiter=" ", dtype=np.float64)
with open('weights/G_conv_layer.csv','r') as f:
    G_chann_weights=np.set_printoptions(precision = 30, suppress = True)
    G_chann_weights= np.genfromtxt(f, delimiter=" ", dtype=np.float64)
with open('weights/B_conv_layer.csv','r') as f:
    B_chann_weights=np.set_printoptions(precision = 30, suppress = True)
    B_chann_weights= np.genfromtxt(f, delimiter=" ", dtype=np.float64)



with open('weights/snd_conv_layer.csv','r') as f:
    kernel2_weights=np.set_printoptions(precision = 30, suppress = True)
    kernel2_weights= np.genfromtxt(f, delimiter=" ", dtype=np.float64)
with open('weights/trd_conv_layer.csv','r') as f:
    kernel3_weights=np.set_printoptions(precision = 30, suppress = True)
    kernel3_weights= np.genfromtxt(f, delimiter=" ", dtype=np.float64)
with open('weights/output_layer.csv','r') as f:
    output_weights=np.set_printoptions(precision = 30, suppress = True)
    output_weights= np.genfromtxt(f, delimiter=",",dtype = np.float64)

R_chann_weights= R_chann_weights.reshape((RGB_kernels_count,RGB_kernels_size,RGB_kernels_size))
G_chann_weights= G_chann_weights.reshape((RGB_kernels_count,RGB_kernels_size,RGB_kernels_size))
B_chann_weights= B_chann_weights.reshape((RGB_kernels_count,RGB_kernels_size,RGB_kernels_size))

kernel2_weights = kernel2_weights.reshape((kernels_2layer_count,kernels_2layer_size,kernels_2layer_size))
kernel3_weights = kernel3_weights.reshape((kernels_3layer_count,kernels_3layer_size,kernels_3layer_size))

@njit
def RGB_convolution(chann_layer, input_chann, chann_weights):
    for i in range(RGB_kernels_count):
        for h in range(conv_chann_size):
            for w in range(conv_chann_size):
                chann_layer[i, h, w] = np.sum(input_chann[h:h + RGB_kernels_size, w:w + RGB_kernels_size] * chann_weights[i])

    return chann_layer


@njit
def next_layers_convolution(conv_layer, conv_layer_size, input_layer, kernel_weights, prev_kernels_count, kernels_size,kernel_per_input):
    l = 0
    r = kernel_per_input
    for k in range(prev_kernels_count):
        for i in range(l, r):
            for h in range(conv_layer_size):
                for w in range(conv_layer_size):
                    conv_layer[i, h, w] = np.sum(input_layer[k, h:h + kernels_size, w:w + kernels_size] * kernel_weights[i])
        l = r
        r += kernel_per_input
    return conv_layer

@njit
def pooling(pooling_layer, pool_part_size, pool_size, conv_layer, kernels_count):
    for s in range(kernels_count):
        for h in range(pool_size):
            for w in range(pool_size):
                pooling_layer[s, h, w] = conv_layer[s, h * pool_part_size:h * pool_part_size + pool_part_size,w * pool_part_size:w * pool_part_size + pool_part_size].max()  
               


path=str()
def choose_image():
    global path
    filename = QtWidgets.QFileDialog.getOpenFileName()
    path=filename[0]
    print(path)
    ui.Image.setPixmap(QtGui.QPixmap(filename[0]))
    
    
    

def recognize():
   
    img = cv2.imread(path)
    data = (cv2.resize(img, dsize=(image_size, image_size), interpolation=cv2.INTER_CUBIC))

    B, G, R = np.asarray(cv2.split(data)) / 255 * 0.99 + 0.01  # 1.0/255 #Нормализация входных каналов

    #Подготовка сверточных слоев
    Rconv_layer = Gconv_layer = Bconv_layer = np.zeros((RGB_kernels_count, conv_chann_size, conv_chann_size))
    conv_layer2 = np.zeros((kernels_2layer_count, conv_layer2_size, conv_layer2_size))
    conv_layer3 = np.zeros((kernels_3layer_count, conv_layer3_size, conv_layer3_size))

    pooling_layer1 = np.zeros((RGB_kernels_count, pool_size1, pool_size1))
    pooling_layer2 = np.zeros((kernels_2layer_count, pool_size2, pool_size2))
    pooling_layer3 = np.zeros((kernels_3layer_count, pool_size3, pool_size3))

    #Подготовка пулинговых слоев
    Rconv_layer = RGB_convolution(Rconv_layer, R, R_chann_weights) 
    Gconv_layer = RGB_convolution(Gconv_layer, G, G_chann_weights) 
    Bconv_layer = RGB_convolution(Bconv_layer, B, B_chann_weights) 

    #Свертка
    conv_layer1 = np.maximum((Rconv_layer + Gconv_layer + Bconv_layer), 0)
    pooling(pooling_layer1, pool_part_size1, pool_size1, conv_layer1, RGB_kernels_count)
   
    conv_layer2 = np.maximum(next_layers_convolution(conv_layer2, conv_layer2_size, pooling_layer1, kernel2_weights, RGB_kernels_count,kernels_2layer_size, (kernels_2layer_count // RGB_kernels_count)), 0)
    pooling(pooling_layer2, pool_part_size2, pool_size2, conv_layer2, kernels_2layer_count)

    conv_layer3 = np.maximum(next_layers_convolution(conv_layer3, conv_layer3_size, pooling_layer2, kernel3_weights, kernels_2layer_count,kernels_3layer_size, (kernels_3layer_count // kernels_2layer_count)), 0)
    pooling(pooling_layer3, pool_part_size3, pool_size3, conv_layer3, kernels_3layer_count)

    output_values = np.dot(output_weights, np.array(pooling_layer3.flatten(), ndmin=2).T)

    Exit = 1 / (1 + np.exp(-output_values))

    print(Exit)

    max_output_index = np.argmax(Exit)
    if (max_output_index == 0):
        ui.label.setText("На картинке изображена кошка")
    else:
        ui.label.setText("На картинке изображена собака")




ui.pushButton.clicked.connect(choose_image)
ui.pushButton_2.clicked.connect(recognize)

sys.exit(app.exec_())


    
