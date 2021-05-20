from tqdm import tqdm
from numba import njit
import numpy as np
import os, sys
import random
import cv2

epochs = 1

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

R_chann_weights = np.random.normal(0.0, pow(RGB_kernels_size * RGB_kernels_size, -0.5),(RGB_kernels_count, RGB_kernels_size, RGB_kernels_size))
G_chann_weights = np.random.normal(0.0, pow(RGB_kernels_size * RGB_kernels_size, -0.5),(RGB_kernels_count, RGB_kernels_size, RGB_kernels_size))
B_chann_weights = np.random.normal(0.0, pow(RGB_kernels_size * RGB_kernels_size, -0.5),(RGB_kernels_count, RGB_kernels_size, RGB_kernels_size))

kernel2_weights = np.random.normal(0.0, pow(kernels_2layer_size * kernels_2layer_size, -0.5),(kernels_2layer_count, kernels_2layer_size, kernels_2layer_size))
kernel3_weights = np.random.normal(0.0, pow(kernels_3layer_size * kernels_3layer_size, -0.5),(kernels_3layer_count, kernels_3layer_size, kernels_3layer_size))

output_weights = np.random.normal(0.0, pow(flatten_conv_size, -0.5), (output_size, flatten_conv_size))

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
def train_pooling(pooling_layer, pooling_layer_ind, pool_part_size, pool_size, conv_layer, kernels_count):
    for s in range(kernels_count):
        for h in range(pool_size):
            for w in range(pool_size):
                pool_part = conv_layer[s, h * pool_part_size:h * pool_part_size + pool_part_size,w * pool_part_size:w * pool_part_size + pool_part_size] 
                pooling_layer[s, h, w] = conv_layer[s, h * pool_part_size:h * pool_part_size + pool_part_size,w * pool_part_size:w * pool_part_size + pool_part_size].max() 

              
                for i in range(pool_part_size):
                    for j in range(pool_part_size):
                        if pool_part[i, j] == pooling_layer[s, h, w]:
                            I = int(i + h * pool_part_size)

                            J = int(j + w * pool_part_size)

                pooling_layer_ind[s, I, J] = 1

@njit
def test_pooling(pooling_layer, pool_part_size, pool_size, conv_layer, kernels_count):
    for s in range(kernels_count):
        for h in range(pool_size):
            for w in range(pool_size):
                pooling_layer[s, h, w] = conv_layer[s, h * pool_part_size:h * pool_part_size + pool_part_size,w * pool_part_size:w * pool_part_size + pool_part_size].max()  

@njit()
def pooling_error_convertation(pooling_layer_error, pl_error_wrong_count, prev_kernels, kernel_per_input, pooling_layer):
    l = 0
    r = kernel_per_input
    for k in range(prev_kernels):

        for s in range(l, r):
            pooling_layer_error[k] += pl_error_wrong_count[s] * (pooling_layer[k] > 0)

        l = r
        r += kernel_per_input

    return pooling_layer_error

@njit
def pooling_error_expansion(pooling_layer_error, conv_layer_error, pooling_layer_ind, kernels_count, conv_layer_size):
    for s in range(kernels_count):
        i = 0
        for h in range(conv_layer_size):
            for w in range(conv_layer_size):
                if (pooling_layer_ind[s, h, w] == 1):
                    conv_layer_error[s, h, w] = pooling_layer_error[s, i]
                    i += 1


@njit()
def back_prop(pl_error_pattern, pl_error_wrong_count, prev_pool_size, conv_layer_size, conv_layer_error, kernels, weights_rot_180, kernel_size):
    for i in range(kernels):
        pl_error_pattern[i, kernel_size - 1:conv_layer_size + kernel_size - 1,kernel_size - 1:conv_layer_size + kernel_size - 1] = conv_layer_error[i] #Матрица ошибок нужного размера для прогона по ней весов

    for s in range(kernels):
        weights_rot_180[s] = np.fliplr(weights_rot_180[s])
        weights_rot_180[s] = np.flipud(weights_rot_180[s])

    for s in range(kernels):
        for h in range(prev_pool_size):
            for w in range(prev_pool_size):
                pl_error_wrong_count[s, h, w] = np.sum(pl_error_pattern[s, h:h + kernel_size, w:w + kernel_size] * weights_rot_180[s])

    return pl_error_wrong_count

@njit
def RGB_weights_updating(kernels_count, kernel_weights, kernels_size, input_layer, conv_layer_error, conv_layer_size):
    for i in range(kernels_count):
        for h in range(kernels_size):
            for w in range(kernels_size):
                kernel_weights[i, h, w] -= np.sum(conv_layer_error[i] * input_layer[h:h + conv_layer_size, w:w + conv_layer_size] * learning_rate)

    return kernel_weights


@njit
def weights_updating(prev_kernels_count, kernel_weights, kernels_size, kernel_per_input, input_layer, conv_layer_error,conv_layer_size):
    l = 0
    r = kernel_per_input
    for k in range(prev_kernels_count):
        for i in range(l, r): 
            for h in range(kernels_size):
                for w in range(kernels_size):
                    kernel_weights[i, h, w] -= np.sum(conv_layer_error[i] * input_layer[k, h:h + conv_layer_size, w:w + conv_layer_size] * learning_rate)
        l = r
        r += kernel_per_input
    return kernel_weights





def training(img, targets, R_chann_weights, G_chann_weights, B_chann_weights, output_weights, kernel2_weights,kernel3_weights):
    
    data = (cv2.resize(img, dsize=(image_size, image_size), interpolation=cv2.INTER_CUBIC))

    B, G, R = np.asarray(cv2.split(data)) / 255 * 0.99 + 0.01  # 1.0/255 #Нормализация входных каналов

    #Подготовка сверточных слоев

    Rconv_layer = Gconv_layer = Bconv_layer = np.zeros((RGB_kernels_count, conv_chann_size, conv_chann_size))  
    conv_layer2 = np.zeros((kernels_2layer_count, conv_layer2_size, conv_layer2_size))
    conv_layer3 = np.zeros((kernels_3layer_count, conv_layer3_size, conv_layer3_size))

    #Подготовка пулинговых слоев и массивов, в которых запоминаются индексы максимальных элементов пулинга (понадобится для создания слоя матриц ошибок сверточных слоев)

    pooling_layer1 = np.zeros((RGB_kernels_count, pool_size1, pool_size1))
    pooling_layer1_ind = np.zeros((RGB_kernels_count, conv_chann_size, conv_chann_size))

    pooling_layer2 = np.zeros((kernels_2layer_count, pool_size2, pool_size2))
    pooling_layer2_ind = np.zeros((kernels_2layer_count, conv_layer2_size, conv_layer2_size))

    pooling_layer3 = np.zeros((kernels_3layer_count, pool_size3, pool_size3))
    pooling_layer3_ind = np.zeros((kernels_3layer_count, conv_layer3_size, conv_layer3_size))

    pl1_error_pattern = np.zeros((kernels_2layer_count, pool_size1 + kernels_2layer_size - 1, pool_size1 + kernels_2layer_size - 1))
    pl1_error_wrong_count = np.zeros((kernels_2layer_count, pool_size1, pool_size1))

    pl2_error_pattern = np.zeros((kernels_3layer_count, pool_size2 + kernels_3layer_size - 1, pool_size2 + kernels_3layer_size - 1))
    pl2_error_wrong_count = np.zeros((kernels_3layer_count, pool_size2, pool_size2))

    #Свертка

    Rconv_layer = RGB_convolution(Rconv_layer, R, R_chann_weights)  
    Gconv_layer = RGB_convolution(Gconv_layer, G, G_chann_weights)  
    Bconv_layer = RGB_convolution(Bconv_layer, B, B_chann_weights)  

    conv_layer1 = np.maximum((Rconv_layer + Gconv_layer + Bconv_layer), 0)
    train_pooling(pooling_layer1, pooling_layer1_ind, pool_part_size1, pool_size1, conv_layer1, RGB_kernels_count)
   
    conv_layer2 = np.maximum(next_layers_convolution(conv_layer2, conv_layer2_size, pooling_layer1, kernel2_weights, RGB_kernels_count,kernels_2layer_size, (kernels_2layer_count // RGB_kernels_count)), 0)
    train_pooling(pooling_layer2, pooling_layer2_ind, pool_part_size2, pool_size2, conv_layer2, kernels_2layer_count)
   
    conv_layer3 = np.maximum(next_layers_convolution(conv_layer3, conv_layer3_size, pooling_layer2, kernel3_weights, kernels_2layer_count,kernels_3layer_size, (kernels_3layer_count // kernels_2layer_count)), 0)
    train_pooling(pooling_layer3, pooling_layer3_ind, pool_part_size3, pool_size3, conv_layer3, kernels_3layer_count)

    output_values = np.dot(output_weights, np.array(pooling_layer3.flatten(), ndmin=2).T)

    Exit = 1 / (1 + np.exp(-output_values))

    #Вычисление ошибки выходного слоя
    Exit_Error = -(targets - Exit)

    #Подготовка матриц ошибок сверточных слоев
    conv_layer1_error = np.zeros((RGB_kernels_count, conv_chann_size, conv_chann_size))
    conv_layer2_error = np.zeros((kernels_2layer_count, conv_layer2_size, conv_layer2_size))
    conv_layer3_error = np.zeros((kernels_3layer_count, conv_layer3_size, conv_layer3_size))

    #Подготовка матриц ошибок пулинговых слоев
    pooling_layer1_Error = np.zeros((RGB_kernels_count, pool_size1, pool_size1))
    pooling_layer2_Error = np.zeros((kernels_2layer_count, pool_size2, pool_size2))

    #Вычисление матриц ошибок 3 сверточного слоя
    pooling_layer3_Error = np.dot(output_weights.T, Exit_Error)
    pooling_layer3_Error = pooling_layer3_Error.reshape((kernels_3layer_count, pool_size3 * pool_size3))

    pooling_error_expansion(pooling_layer3_Error, conv_layer3_error, pooling_layer3_ind, kernels_3layer_count,conv_layer3_size)


    #Вычисление матриц ошибок 2 сверточного слоя

    pl2_error_wrong_count = back_prop(pl2_error_pattern, pl2_error_wrong_count, pool_size2, conv_layer3_size,conv_layer3_error, kernels_3layer_count, kernel3_weights, kernels_3layer_size)
    pooling_layer2_Error = pooling_error_convertation(pooling_layer2_Error, pl2_error_wrong_count, kernels_2layer_count,(kernels_3layer_count // kernels_2layer_count), pooling_layer2)

    pooling_layer2_Error = pooling_layer2_Error.reshape((kernels_2layer_count, pool_size2 * pool_size2))
    pooling_error_expansion(pooling_layer2_Error, conv_layer2_error, pooling_layer2_ind, kernels_2layer_size,conv_layer2_size)


    #Вычисление матриц ошибок 1 сверточного слоя
    pl1_error_wrong_count = back_prop(pl1_error_pattern, pl1_error_wrong_count, pool_size1, conv_layer2_size,conv_layer2_error, kernels_2layer_count, kernel2_weights, kernels_2layer_size)
    pooling_layer1_Error = pooling_error_convertation(pooling_layer1_Error, pl1_error_wrong_count, RGB_kernels_count,(kernels_2layer_count // RGB_kernels_count), pooling_layer1)

    pooling_layer1_Error = pooling_layer1_Error.reshape((RGB_kernels_count, pool_size1 * pool_size1))
    pooling_error_expansion(pooling_layer1_Error, conv_layer1_error, pooling_layer1_ind, RGB_kernels_count,conv_chann_size)


    #Обновление весов
    output_weights -= learning_rate * np.dot((Exit_Error * Exit * (1.0 - Exit)),np.array(pooling_layer3.flatten(), ndmin=2))

    kernel3_weights = weights_updating(kernels_2layer_count, kernel3_weights, kernels_3layer_size,(kernels_3layer_count // kernels_2layer_count), conv_layer2, conv_layer3_error,conv_layer3_size)
    kernel2_weights = weights_updating(RGB_kernels_count, kernel2_weights, kernels_2layer_size,(kernels_2layer_count // RGB_kernels_count), conv_layer1, conv_layer2_error,conv_layer2_size)

    Rconv_layer = RGB_weights_updating(RGB_kernels_count, R_chann_weights, RGB_kernels_size, R, conv_layer1_error,conv_chann_size)
    Gconv_layer = RGB_weights_updating(RGB_kernels_count, G_chann_weights, RGB_kernels_size, G, conv_layer1_error,conv_chann_size)
    Bconv_layer = RGB_weights_updating(RGB_kernels_count, B_chann_weights, RGB_kernels_size, B, conv_layer1_error,conv_chann_size)
    pass


path = 'dataset\\train'



inputs_names = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
random_inputs_names = random.sample(inputs_names, len(inputs_names))

for i in range(epochs):
    for j in tqdm(random_inputs_names, desc=str(i + 1)):

        img = cv2.imread(path + '\\' + j)

        targets = np.array([0.01, 0.01]).reshape((2, 1))

        if str(j[0]) == 'c':
            targets[0] = 0.99  # 1
        else:
            targets[1] = 0.99  # 1

        training(img, targets, R_chann_weights, G_chann_weights, B_chann_weights, output_weights, kernel2_weights,kernel3_weights)


def testing(img):
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
    test_pooling(pooling_layer1, pool_part_size1, pool_size1, conv_layer1, RGB_kernels_count)
   
    conv_layer2 = np.maximum(next_layers_convolution(conv_layer2, conv_layer2_size, pooling_layer1, kernel2_weights, RGB_kernels_count,kernels_2layer_size, (kernels_2layer_count // RGB_kernels_count)), 0)
    test_pooling(pooling_layer2, pool_part_size2, pool_size2, conv_layer2, kernels_2layer_count)

    conv_layer3 = np.maximum(next_layers_convolution(conv_layer3, conv_layer3_size, pooling_layer2, kernel3_weights, kernels_2layer_count,kernels_3layer_size, (kernels_3layer_count // kernels_2layer_count)), 0)
    test_pooling(pooling_layer3, pool_part_size3, pool_size3, conv_layer3, kernels_3layer_count)

    output_values = np.dot(output_weights, np.array(pooling_layer3.flatten(), ndmin=2).T)

    Exit = 1 / (1 + np.exp(-output_values))

    return Exit


path = 'dataset\\train'

efficiency = []

for j in tqdm(os.listdir(path)):

    img = cv2.imread(path + '\\' + j)
    targets_ind = 0

    if str(j[0]) == 'c':
        targets_ind = 0
    else:
        targets_ind = 1

    outputs = testing(img)

    max_output_index = np.argmax(outputs)

    if (max_output_index == targets_ind):
        efficiency.append(1)
    else:
        efficiency.append(0)

efficiency_array = np.asarray(efficiency)

performance = (efficiency_array.sum() / efficiency_array.size) * 100

print('Производительность:', performance, '%')

# np.savetxt("efficiency_arr.txt", efficiency_array, delimiter=",")

with open('weights/R_conv_layer.csv', 'w') as outfile:
    outfile.write('# Исходный размер R сверточного слоя: {0}\n'.format(R_chann_weights.shape))
    for data_slice in R_chann_weights:
        outfile.write('# Ядро сверток:\n')
        np.savetxt(outfile, data_slice)
with open('weights/G_conv_layer.csv', 'w') as outfile:
    outfile.write('# Исходный размер G сверточного слоя: {0}\n'.format(G_chann_weights.shape))
    for data_slice in G_chann_weights:
        outfile.write('# Ядро сверток:\n')
        np.savetxt(outfile, data_slice)
with open('weights/B_conv_layer.csv', 'w') as outfile:
    outfile.write('# Исходный размер B сверточного слоя: {0}\n'.format(B_chann_weights.shape))
    for data_slice in B_chann_weights:
        outfile.write('# Ядро сверток:\n')
        np.savetxt(outfile, data_slice)

with open('weights/snd_conv_layer.csv', 'w') as outfile:
    outfile.write('# Исходный размер 2-го сверточного слоя: {0}\n'.format(kernel2_weights.shape))
    for data_slice in kernel2_weights:
        outfile.write('# Ядро сверток:\n')
        np.savetxt(outfile, data_slice)
                    
with open('weights/trd_conv_layer.csv', 'w') as outfile:
    outfile.write('# Исходный размер 3-го сверточного слоя: {0}\n'.format(kernel3_weights.shape))
    for data_slice in kernel3_weights:
        outfile.write('# Ядро сверток:\n')
        np.savetxt(outfile, data_slice)

np.savetxt("weights/output_layer.csv", output_weights, delimiter=",")
