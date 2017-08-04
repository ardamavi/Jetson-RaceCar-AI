# Arda Mavi

import numpy as np
from scipy.misc import imread, imresize

def get_img(data_path):
    # Getting image array from path:
    img = imread(data_path, mode='L')
    img = imresize(img, (500, 500, 1))
    return img

def get_data():
    with open('Data/Train_Label.text', 'r') as file:
        all_file = file.read()
    X_1, X_2, Y = [], [], []
    datasets = all_file.split('\n')
    for data in datasets:
        one_data = data.split(' ')
        img = get_img('Data/Train_Data/'+one_data[0])
        X_1.append(img)
        X_2.append([float(one_data[3]), float(one_data[4]), float(one_data[5])])
        Y.append([float(one_data[1]), float(one_data[2])])
    X_1 = np.array(X_1).reshape(len(datasets), 500, 500, 1)
    X_2 = np.array(X_2).reshape(len(datasets), 3)
    Y = np.array(Y).reshape(len(datasets), 2)
    return X_1, X_2, Y
