import cv2

image = cv2.imread('img2.jpg')

image = cv2.resize(image, None, fx=.3, fy=.3, interpolation=cv2.INTER_AREA)
img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
img1 = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
img2 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)

cv2.imshow("BGR image", image)
cv2.imshow("Gray image", img)
cv2.imshow("gray1", img2)
cv2.waitKey(0)
cv2.destroyAllWindows()
#cv2.imwrite('gray_img2.jpg', image)

