import cv2
import numpy as np

image = cv2.imread('img2.jpg')
image = cv2.resize(image, None, fx=.3, fy=.3, interpolation=cv2.INTER_AREA)
img = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)

h, l, s = cv2.split(img)

#lower_blue = np.array([110,50,50])
#upper_blue = np.array([130,255,255])

#mask = cv2.inRange(img, lower_blue, upper_blue)
#res = cv2.bitwise_and(img, img, mask = mask)

cv2.imshow("BGR image",image)
cv2.imshow("HSV image",img)
cv2.imshow('h img', h)
cv2.imshow('s img', s)
cv2.imshow('l img', l)
#cv2.imshow('h mask img', mask)
#cv2.imshow('h res img', res)


cv2.waitKey(0)
cv2.destroyAllWindows()
