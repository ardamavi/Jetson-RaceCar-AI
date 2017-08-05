# Arda Mavi

import numpy as np
from scipy.misc import imresize
from multiprocessing import Process, Value, Array
from predict import predict, get_ready_model
from Sensor_Data.zed_process import get_zed_data, get_capture, release_capture
from Sensor_Data.lidar_process import get_lidar_data, start_lidar, stop_lidar

zed_data = Array('f', [])
lidar_data = Array('f', [])
data_flow = Value('i', 1)

def zed_data_process(cap):
    while data_flow.value:
        zed_data = np.array(imresize(np.array(get_zed_data(cap)), (500, 500, 1)))

def lidar_data_process():
    while data_flow.value:
        lidar_data = np.array(get_lidar_data())

def ai():
    print('Preparing model ...')
    model = get_ready_model()
    print('Model ready.')

    print('Preparing lidar ...')
    start_lidar()
    print('Lidar ready.')

    print('Preparing camera...')
    cap = get_capture()
    print('Camera ready.')

    data_flow.value = True

    print('Threads starting')
    # Start getting data process:
    camera_process = Process(target=zed_data_process, args=(cap,))
    camera_process.start()
    print('Camera thread start.')

    lidar_process = Process(target=lidar_data_process)
    lidar_process.start()
    print('Lidar thread start.')

    print('AI will start in a short time.')

    while True:
        stereo_img = np.array(zed_data)
        lidar_map = np.array(lidar_data)
        try:
            if stereo_img.size == 1 and lidar_map.size == 1:
                print(predict(model, stereo_img, lidar_map))
        except:
            print('AI Error !')
            break

    data_flow.value = False
    camera_process.join()
    lidar_process.join()

    stop_lidar()

    release_capture(cap)

if __name__ == '__main__':
    ai()
