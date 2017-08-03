# Arda Mavi
import cv2

# Getting capture:
def get_capture(camera=1): # Camera 0 is jetson's embeded camera.
    cap = cv2.VideoCapture(camera)
    return cap

# Release capture:
def release_capture(capture):
    capture.release()

stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)

# Take a picture:
def get_zed_data(capture):
    ret, img = capture.read()

    # img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    img_left = img[0:376, 0:672]
    img_right = img[0:376, 672:1344]

    # disparity = stereo.compute(img_left,img_right)

    return disparity
