import cv2

image = cv2.imread('img3.jpg')
image = cv2.resize(image, None, fx=.3, fy=.3, interpolation=cv2.INTER_AREA)

b,g,r = cv2.split(image)

cv2.imshow("blue", b)
cv2.imshow("orig", image)
cv2.imshow("red", r)
cv2.imshow("green", g)

k = cv2.waitKey(0)
if k==27:
    cv2.destroyAllWindows()
elif k == ord('s'):
    cv2.imwrite('blue_img2.jpg', b)
    cv2.imwrite('red_img2.jpg', r)
    cv2.imwrite('green_img2.jpg', g)
    cv2.destroyAllWindows()
