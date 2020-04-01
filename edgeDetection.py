#-----------------
# MAIN FUNCTION
#-----------------

# organize imports
import cv2
import imutils
import numpy as np

# global variables
bg = None

#--------------------------------------------------
# To find the running average over the background
#--------------------------------------------------
def run_avg(image, aWeight):
    global bg
    # initialize the background
    if bg is None:
        bg = image.copy().astype("float")
        return

    # compute weighted average, accumulate it and update the background
    cv2.accumulateWeighted(image, bg, aWeight)

#---------------------------------------------
# To segment the region of hand in the image
#---------------------------------------------
def segment(image, threshold=25):
    global bg
    # find the absolute difference between background and current frame
    diff = cv2.absdiff(bg.astype("uint8"), image)

    # threshold the diff image so that we get the foreground
    thresholded = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)[1]

    # get the contours in the thresholded image
    (_, cnts, _) = cv2.findContours(thresholded.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    segmented = cnts
    return (thresholded, segmented)

if __name__ == "__main__":
    # initialize weight for running average
    aWeight = 0.8

    # get the reference to the webcam
    camera = cv2.VideoCapture(0)

    # keep looping, until interrupted
    while(True):
        # get the current frame
        (_, frame) = camera.read()

        # resize the frame
        frame = imutils.resize(frame, width=700)

        # flip the frame so that it is not the mirror view
        frame = cv2.flip(frame, 1)

        # convert the roi to grayscale and blur it
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (7, 7), 0)

        # to get the background, keep looking till a threshold is reached
        # so that our running average model gets calibrated
        run_avg(gray, aWeight)

        # segment the hand region
        image = segment(gray, 7)

        (thresholded, segmented) = image

        cv2.drawContours(frame, segmented, -1, (200, 0, 255))
        # display the frame with segmented hand
        cv2.imshow("Thresholded", thresholded)

        cv2.imshow("Video Feed", frame)

        # observe the keypress by the user
        keypress = cv2.waitKey(1) & 0xFF

        # if the user pressed "q", then stop looping
        if keypress == ord("q"):
            break

# free up memory
camera.release()
cv2.destroyAllWindows()
