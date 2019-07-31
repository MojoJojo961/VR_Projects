import cv2
import numpy as np

image = cv2.imread('img3.jpg')
image = cv2.resize(image, None, fx=.15, fy=.15, interpolation=cv2.INTER_AREA)
img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

h,s,v = cv2.split(img)
b,g,r = cv2.split(image)

#lower_blue = np.array([110,50,50])
#upper_blue = np.array([130,255,255])

#mask = cv2.inRange(img, lower_blue, upper_blue)
#res = cv2.bitwise_and(img, img, mask = mask)

cv2.imshow("BGR image",image)
cv2.imshow("HSV image",img)
cv2.imshow('h img', h)
cv2.imshow('s img', s)
cv2.imshow('v img', v)
#cv2.imshow('h mask img', mask)
#cv2.imshow('h res img', res)


cv2.waitKey(0)
cv2.destroyAllWindows()
