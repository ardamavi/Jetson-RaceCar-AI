# Arda Mavi

import sys
import numpy as np
from get_data import get_img
from scipy.misc import imresize
from keras.models import model_from_json

def predict(model, img, lidar_data):
    img = imresize(img, (700, 700, 3))
    img = np.array(img).reshape(1, 700, 700, 3)
    lidar_data = np.array(lidar_data).reshape(1, 3)
    Y = model.predict([img], [lidar_data])
    return Y

if __name__ == '__main__':
    img_dir = sys.argv[1]
    lidar_data = [sys.argv[2], sys.argv[3], sys.argv[4]]
    img = get_img(img_dir)
    # Getting model:
    model_file = open('Data/Model/model.json', 'r')
    model = model_file.read()
    model_file.close()
    model = model_from_json(model)
    # Getting weights
    model.load_weights("Data/Model/weights.h5")
    print(predict(model, img, lidar_data))
