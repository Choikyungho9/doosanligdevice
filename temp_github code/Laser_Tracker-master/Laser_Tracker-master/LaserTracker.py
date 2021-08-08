#!/usr/bin/python
# THis code uses openCV and python to track the position of a laser beam spot. 

import cv2
import numpy as np
import cv2.cv as cv
import time
import numpy as np
from numpy import unravel_index

kernel = np.ones((5,5),np.uint8)

# create video capture
cap = cv2.VideoCapture(0)
width=640
height=480 

#Reduce the size of video to 320x240 so rpi can process faster
cap.set(3,width)
cap.set(4,height)

time.sleep(2)

# Creating a windows for later use
cv2.namedWindow('closing')
cv2.namedWindow('tracking')

# My experimental grayscale threshold values.
gmn = 18
gmx = 253

#initialize centroid variables for later in program
cx = 0
cy = 0

#remember when we started
start_time = time.time() 
#if limited run is desired, set number of seconds here
max_time = start_time + int(60) 

#if limited run is desired, set while(max_time) > time.time():  
while(1):
   # read the frames
    _,frame = cap.read()

    # convert to grayscale
    tracking = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    #apply thresholding to grayscale frames. 
    ret, thresh = cv2.threshold(tracking, gmn, gmx, 0)

    # find contours in the threshold image
    contours,hierarchy = cv2.findContours(thresh,cv2.RETR_LIST,cv2.CHAIN_APPROX_TC89_L1)

    # finding contour with maximum area and store it as best_cnt
    max_area = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > max_area:
            max_area = area
            best_cnt = cnt

            # finding centroids of best_cnt and draw a circle there
            M = cv2.moments(best_cnt)
            cx,cy = int(M['m10']/M['m00']), int(M['m01']/M['m00'])
            #print centroid coordinates in terminal. These could also be outputted to a file.
            print cx   
            print cy
            break
        else:
            cx = 'X value not found'
            cy = 'Y value  not found'
            break
 

    #Used to (half) confirm the results printed in the terminal. Comment these out for faster performance. 
    cv2.imshow('closing',frame)
    cv2.imshow('tracking', tracking)

    # Show it, if key pressed is 'Esc', exit the loop
    if cv2.waitKey(33)== 27:
        break

# Clean up everything before leaving
cv2.destroyAllWindows()
cap.release()

