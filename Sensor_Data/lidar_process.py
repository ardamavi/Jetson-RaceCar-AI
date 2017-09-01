# Arda Mavi
from sweeppy import Sweep
import itertools

def get_lidar_data():
    with Sweep('/dev/ttyUSB0') as lidar:
        while not lidar.get_motor_ready():
            pass
        lidar.start_scanning()
        scans = lidar.get_scans()
        data = []
        for scan in itertools.islice(lidar.get_scans(), 1):
            datas = scan[0]
            data.append([datas[0], datas[1], datas[2]])
        lidar.stop_scanning()
    return data

def start_lidar():
    set_motor_speed(speed=5)
    set_sample_rate(rate=500)
    with Sweep('/dev/ttyUSB0') as lidar:
            lidar.start_scanning()

def stop_lidar():
    set_motor_speed(speed=0)
    set_sample_rate(rate=0)
    with Sweep('/dev/ttyUSB0') as lidar:
        lidar.stop_scanning()

def set_motor_speed(speed=5):
    with Sweep('/dev/ttyUSB0') as lidar:
        if lidar.get_motor_speed() != speed:
            lidar.set_motor_speed(speed)

def set_sample_rate(rate=500):
    with Sweep('/dev/ttyUSB0') as lidar:
        if lidar.get_sample_rate() != rate:
            lidar.set_sample_rate(rate)
