# Arda Mavi

import numpy as np
from scipy.misc import imread, imresize

def get_img(data_path):
    # Getting image array from path:
    img = imread(data_path)
    img = imresize(img, (700, 700, 3))
    return img

def get_data():
    with open('Data/Train_Label.text', 'r') as file:
        all_file = file.read()
    X_1, X_2, Y_1, Y_2 = [], [], [], []
    datasets = all_file.split('\n')
    for data in datasets:
        one_data = data.split(' ')
        img = get_img('Data/Train_Data/'+one_data[0]+'.jpg')
        X_1.append([img, img])
        X_2.append([float(one_data[3]), float(one_data[4]), float(one_data[5])])
        Y_1.append(float(one_data[1]))
        Y_2.append(float(one_data[2]))
    X = np.array([X_1,X_2])
    Y = np.array([Y_1,Y_2])
    return X, Y
