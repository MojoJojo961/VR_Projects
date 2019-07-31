import cv2
import numpy as np

image = cv2.imread('img2.jpg', 0)
image = cv2.resize(image, None, fx=.5, fy=.5, interpolation=cv2.INTER_AREA)
equ = cv2.equalizeHist(image)

res = np.hstack((image, equ))
cv2.imshow('stacked images', res)
cv2.waitKey(0)
cv2.destroyAllWindows()
