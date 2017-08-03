# Arda Mavi

from multiprocessing import Process
from predict import predict, get_ready_model
from Sensor_Data.zed_process import get_zed_data, get_capture, release_capture
from Sensor_Data.lidar_process import get_lidar_data, start_lidar, stop_lidar

zed_data = None
lidar_data = None

def zed_data_process(cap):
    zed_data = get_zed_data(cap)

def lidar_data_process():
    lidar_data = get_lidar_data()

def ai():
    model = get_ready_model()

    start_lidar()

    cap = get_capture()

    # Start getting data process:
    Process(target=zed_data_process, args=(cap,)).start()
    Process(target=lidar_data_process).start()

    while zed_data != None and lidar_data != None:
        try:
            print(predict(model, zed_data, lidar_data))
        except:
            print('AI Error')
            break

    stop_lidar()

    release_capture(cap)

if __name__ == '__main__':
    ai()
