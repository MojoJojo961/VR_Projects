import cv2
import numpy as np

image = cv2.imread('img2.jpg', 0)

mu = np.mean(image)
sigma = np.std(image)

img = np.divide(np.subtract(image, mu), sigma)
print(image.shape)

cv2.imshow("image", image)
cv2.imshow("whitened image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
