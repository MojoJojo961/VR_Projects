import cv2
import numpy as np

image = cv2.imread('img5.jpg')

image = cv2.resize(image, None, fx=.5, fy=.5, interpolation=cv2.INTER_AREA)
#blur = cv2.blur(image, (2,2))
blur1 = cv2.GaussianBlur(image, (1,1), cv2.BORDER_DEFAULT)
blur2 = cv2.GaussianBlur(image, (3,3), cv2.BORDER_DEFAULT)
blur3 = cv2.GaussianBlur(image, (5,5), cv2.BORDER_DEFAULT)
blur4 = cv2.GaussianBlur(image, (7,7), cv2.BORDER_DEFAULT)

i = np.hstack((image,blur2))
i = np.hstack((i, blur3))
i = np.hstack((i, blur4))

cv2.imshow("image1", i)
#cv2.imshow("image2", np.hstack((image, blur2)))
#cv2.imshow("image3", np.hstack((image, blur3)))
#cv2.imshow("image4", np.hstack((image, blur4)))
cv2.waitKey(0)
cv2.destroyAllWindows()
