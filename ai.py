# Arda Mavi

from predict import predict, get_ready_model
from multiprocessing import Process
from Sensor_Data.lidar_process import get_lidar_data
from Sensor_Data.zed_process import get_zed_data

zed_data = []
lidar_data = []

def zed_data_process():
    zed_data = get_zed_data()

def lidar_data_process():
    lidar_data = get_lidar_data()

def ai():
    model = get_ready_model()

    # Start getting data process:
    Process(target=zed_data_process).start()
    Process(target=lidar_data_process).start()

    predict(model, zed_data, lidar_data)
