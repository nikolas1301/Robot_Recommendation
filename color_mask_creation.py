import cv2
import numpy as np

def color_mask(frame, hsv_low=[0, 0, 0], hsv_up=[179, 255, 255]):
    '''(numpy.ndarray, array, array) -> numpy.ndarray

    frame - input image: 8-bit unsigned, 16-bit unsigned ( CV_16UC... ), or single-precision floating-point.
    hsv_low - array with lower bound hsv values.
    hsv_up - array with upper bound hsv values.
    Return - output image: pixel values of 0 and 1.

    Creats a mask based on HSV color space, defined by the lower and upper bounds.
    '''

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) 

    # Define the hsv lower and higher value to creat the mask
    lower_hsv = np.array(hsv_low)
    upper_hsv = np.array(hsv_up)

    # This creates a mask of the chose color.
    mask = cv2.inRange(hsv, lower_hsv, upper_hsv)

    #morphological filtering
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)))

    return mask