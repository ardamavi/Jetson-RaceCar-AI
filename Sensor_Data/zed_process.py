# Arda Mavi
import cv2

# Getting capture:
def get_capture(camera=1): # Camera 0 is jetson's embeded camera.
    cap = cv2.VideoCapture(camera)
    return cap

# Release capture:
def release_capture(capture):
    capture.release()

# Take a picture:
def get_zed_data(capture):
    while True:
        ret, img = capture.read()
        img_left = img[0:720, 0:1280]
        img_right = img[0:720, 1280:2560]

        # TODO: Stereo picture
        # stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)
        # img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # disparity = stereo.compute(img_left,img_right)

    return img
