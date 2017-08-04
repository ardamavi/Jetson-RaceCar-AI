# Arda Mavi

import numpy as np
from scipy.misc import imresize
from multiprocessing import Process
from predict import predict, get_ready_model
from Sensor_Data.zed_process import get_zed_data, get_capture, release_capture
from Sensor_Data.lidar_process import get_lidar_data, start_lidar, stop_lidar

zed_data = None
lidar_data = None
data_flow = False

def zed_data_process(cap):
    while data_flow:
        zed_data = np.array(imresize(np.array(get_zed_data(cap)), (500, 500, 1)))
        zed_data = True

def lidar_data_process():
    while data_flow:
        lidar_data = np.array(get_lidar_data())
        lidar_data = True

def ai():
    model = get_ready_model()

    start_lidar()
    cap = get_capture()

    data_flow = True

    # Start getting data process:
    Process(target=zed_data_process, args=(cap,)).start()
    Process(target=lidar_data_process).start()

    while True:
        while zed_data != None and lidar_data != None:
            pass
        try:
            print(predict(model, zed_data, lidar_data))
        except:
            print('AI Error: ')
            break

    data_flow = False

    stop_lidar()

    release_capture(cap)

if __name__ == '__main__':
    ai()
