import cv2
import numpy as np
import def_blob_param

# Setup SimpleBlobDetector parameters.
params = cv2.SimpleBlobDetector_Params()

def_blob_param.blob_param(params)

# Create a detector with the parameters
detector = cv2.SimpleBlobDetector_create(params)

class Object:
    def __init__(self, name, hsv_low, hsv_up):
        self.name = name
        self.hsv_low = hsv_low
        self.hsv_up = hsv_up

    def mask_creation(self, frame):
        '''(numpy.ndarray) -> numpy.ndarray

        frame - input image: 8-bit unsigned, 16-bit unsigned ( CV_16UC... ), or single-precision floating-point.
        hsv_low - array with lower bound hsv values.
        hsv_up - array with upper bound hsv values.
        Return - output image: pixel values of 0 and 1.

        Creats a mask based on HSV color space, defined by the lower and upper bounds.
        '''

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) 

        # Define the hsv lower and higher value to creat the mask
        lower_hsv = np.array(self.hsv_low)
        upper_hsv = np.array(self.hsv_up)

        # This creates a mask of the chose color.
        mask = cv2.inRange(hsv, lower_hsv, upper_hsv)

        #morphological filtering
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)))

        return mask

    def obj_detect (self, mask):
        '''
        Detect the blobs on the mask and plot it
        '''
        # Detect blobs.
        keypoints = detector.detect(mask)

        # Draw detected blobs as red circles.
        # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures
        # the size of the circle corresponds to the size of blob
        mask = cv2.drawKeypoints(mask, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        return keypoints, mask

    def obj_plot (self, frame, keypoints):
        '''
        Draw the object on the camera
        '''

        #Go through keypoints, get the center of the blobs and it size,
        #transform into UInt16 and draw the circles
        for keypoints in keypoints:
            center = (np.uint16(np.around(keypoints.pt[0])), np.uint16(np.around(keypoints.pt[1])))
            # circle center
            cv2.circle(frame, center, 1, (0, 0, 255), 3)
            # circle outline
            radius = np.uint16(np.around(keypoints.size/2.0))
            cv2.circle(frame, center, radius, (0, 255, 0), 3)
            # object name
            cv2.putText(frame, self.name, (center[0]+radius,center[1]-int(0.8*radius)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2, cv2.LINE_AA)

