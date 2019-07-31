import cv2
import numpy as np

image = cv2.imread('img2.jpg')
image = cv2.resize(image, None, fx=.3, fy=.3, interpolation=cv2.INTER_AREA)
img = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)

l,a,b = cv2.split(img)

cv2.imshow("BGR image",image)
cv2.imshow("Lab image",img)
cv2.imshow('L* img', l)
cv2.imshow('a* img', a)
cv2.imshow('b* img', b)


cv2.waitKey(0)
cv2.destroyAllWindows()
