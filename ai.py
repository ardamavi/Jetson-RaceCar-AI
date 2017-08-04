# Arda Mavi

import numpy as np
from scipy.misc import imresize
from multiprocessing import Process, Value, Array
from predict import predict, get_ready_model
from Sensor_Data.zed_process import get_zed_data, get_capture, release_capture
from Sensor_Data.lidar_process import get_lidar_data, start_lidar, stop_lidar

zed_data = Array('d', [])
lidar_data = Array('d', [])
data_flow = Value('i', True)

def zed_data_process(cap):
    while data_flow.value:
        zed_data.value = np.array(imresize(np.array(get_zed_data(cap)), (500, 500, 1)))

def lidar_data_process():
    while data_flow.value:
        lidar_data.value = np.array(get_lidar_data())

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

    print('Processes starting')
    # Start getting data process:
    camera_process = Process(target=zed_data_process, args=(cap,))
    camera_process.start()
    print('Camera process start.')

    lidar_process = Process(target=lidar_data_process)
    lidar_process.start()
    print('Lidar process start.')

    print('AI will start in a short time.')

    while True:
        while zed_data.value == [] or lidar_data.value == []:
            pass
        try:
            print(predict(model, zed_data.value, lidar_data.value))
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
