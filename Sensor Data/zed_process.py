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
    stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)
    while True:
        ret, img = capture.read()
        print(ret)
        #img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img_left = img[0:720, 0:1280]
        img_right = img[0:720, 1280:2560]

        # disparity = stereo.compute(img_left,img_right)

        cv2.imshow('Left',img_left)
        cv2.imshow('Right',img_right)

        if cv2.waitKey(1) == 27: # Decimal 27 = Esc
                break

    cv2.destroyAllWindows()



    return two_image


cap = get_capture()
get_zed_data(cap)
release_capture()