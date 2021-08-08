import cv2
import numpy as np

img = cv2.imread('pic2/file1.jpg')
print(img.shape)
mask = np.zeros(img.shape[:2], dtype="uint8")
px0 = 220
px1 = 420
py0 = 120
py1 = 360
cv2.rectangle(mask, (px0, py0), (px1, py1), (255, 255, 255), -1)
masked = cv2.bitwise_and(img, img, mask=mask)
print(masked.shape)
cv2.imshow('test', masked)

cv2.waitKey(0)
cv2.destroyAllWindows()