import cv2
import numpy as np
import matplotlib.pyplot as plt

image1 = cv2.imread('Panorama/institute2.jpg')
img1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)

image2 = cv2.imread('Panorama/institute1.jpg')
img2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

# get the sift object
sift = cv2.xfeatures2d.SIFT_create()

# find the keypoints and descriptors
keypt1, desc1 = sift.detectAndCompute(img1, None)
keypt2, desc2 = sift.detectAndCompute(img2, None)

# match the similar features using Brute Force Matcher
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
matches = bf.knnMatch(desc1, desc2, k=2)

# store all the good matches as per Lowe's ratio test.
good = []
for m in matches:
    if m[0].distance < 0.5*m[1].distance:
        good.append(m)
matches = np.asarray(good)

# Align the images
# Homography matrix requires at least 4 matches and is needed for the transformation
if len(matches[:,0]) >= 4:
    src = np.float32([keypt1[m.queryIdx].pt for m in matches[:,0]]).reshape(-1,1,2)
    dest = np.float32([keypt2[m.trainIdx].pt for m in matches[:,0]]).reshape(-1,1,2)

    H, masked = cv2.findHomography(src,dest,cv2.RANSAC, 5.0)
else:
    raise AssertionError("Can't find enough keypoints")

# Warp and stitch the image
dest = cv2.warpPerspective(image1, H, (image2.shape[1] + image1.shape[1], image2.shape[0]))

plt.subplot(122), plt.imshow(dest), plt.title('Warped Image')
plt.show()
plt.figure()

dest[0:image2.shape[0], 0:image2.shape[1]] = image2

plt.imshow(dest)
plt.show()
