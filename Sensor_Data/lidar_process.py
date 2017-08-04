# Arda Mavi
from sweeppy import Sweep

def get_lidar_data():
    start_lidar()
    with Sweep('/dev/ttyUSB0') as lidar:
        while not lidar.get_motor_ready():
            pass
        lidar.start_scanning()
        scans = lidar.get_scans()
        data = []
        for scan in itertools.islice(sweep.get_scans(), 0, None):
            data.append([scan[0], scan[1], scan[2]])
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
