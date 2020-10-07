import cv2

def blob_param(params):
    '''
    (cv2.SimpleBlobDetector_Params()) -> None
    
    Define the blobs parameters to the ones that I'm using on this project.

    '''
    # Filter by color
    params.filterByColor = True
    params.blobColor = 255

    # Filter by Area.
    params.filterByArea = True
    params.minArea = 250
    params.maxArea = 10000000

    # Filter by Circularity
    params.filterByCircularity = True
    params.minCircularity = 0.08

    # Filter by Convexity
    params.filterByConvexity = True
    params.minConvexity = 0.87

    # Filter by Inertia
    params.filterByInertia = True
    params.minInertiaRatio = 0.1

