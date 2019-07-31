import cv2
import numpy as np

image = cv2.imread('img3.jpg')

image = cv2.resize(image, None, fx=.3, fy=.3, interpolation=cv2.INTER_AREA)

row,col,ch = image.shape
s_vs_p = 0.5
amount = 0.2
out = np.copy(image)
# Salt mode
num_salt = np.ceil(amount * image.size * s_vs_p)
coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
out[coords] = 1
# Pepper mode
num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
out[coords] = 0

i = np.hstack((image,out))

# median filter
blur = cv2.medianBlur(out, 3) 

cv2.imshow("image1", i)
cv2.imshow("filter", np.hstack((i, blur)))
cv2.waitKey(0)
cv2.destroyAllWindows()
