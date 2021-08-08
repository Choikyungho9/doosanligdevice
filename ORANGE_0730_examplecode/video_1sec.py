import cv2
import numpy as np
import time
import pygame
from pygame import Surface
import sys
import time
import os
#import smbus
from pygame.locals import QUIT, Rect
import serial
import select
import sys
# Create a VideoCapture object
cap = cv2.VideoCapture(0)

# Check if camera opened successfully
if (cap.isOpened() == False):
    print("Unable to read camera feed")

# Default resolutions of the frame are obtained.The default resolutions are system dependent.
# We convert the resolutions from float to integer.
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
print(frame_height, frame_width)
#input("...")
#frame_width = 1280
#frame_height = 720

# Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.
out = cv2.VideoWriter('test4.avi', cv2.VideoWriter_fourcc(*'XVID'), 10, (frame_width, frame_height))

t_end = time.time() + 100 * 0.16
while time.time() < t_end:
    ret, frame = cap.read()
    if ret == True:
        # Write the frame into the file 'output.avi'
        out.write(frame)
        # Display the resulting frame
        cv2.imshow('frame', frame)
        # Press Q on keyboard to stop recording
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    # Break the loop
    else:
        break
    # When everything done, release the video capture and video write objects
cap.release()
out.release()

# Closes all the frames
cv2.destroyAllWindows()
