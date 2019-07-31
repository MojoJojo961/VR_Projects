import cv2
import numpy as np
import matplotlib.pyplot as plt
import math

def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    match_mask_color = 255
    cv2.fillPoly(mask, vertices, match_mask_color)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def draw_lines(img, lines, color=[0,255,0], thickness=3):
    line_img = np.zeros(
            (img.shape[0],
             img.shape[1],
             3
            ),
            dtype = np.uint8
    )
    img = np.copy(img)
    if lines is None:
        return

    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(line_img, (x1,y1), (x2,y2), color, thickness)

    img = cv2.addWeighted(img, 0.8, line_img, 1.0, 0.0)
    return img

def show_image(image):
    plt.figure()
    plt.imshow(image)
    plt.show()

def pipeline(image):
    
    height = image.shape[0]
    width = image.shape[1]
    region_of_interest_vertices = [
            (0, height),
            (width/2, height/2),
            (width, height),
    ]

    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur_img = cv2.GaussianBlur(gray_img, (7,7), 0)
    cannyed_img = cv2.Canny(blur_img, 50, 125)
    cropped_img = region_of_interest(cannyed_img,
            np.array([region_of_interest_vertices], np.int32),
            )
    show_image(cropped_img)

    lines = cv2.HoughLinesP(
            cropped_img,
            rho = .2,
            theta = np.pi/180,
            threshold = 20,
            lines = np.array([]),
            minLineLength = 30,
            maxLineGap = 40
    )

    left_line_x = []
    left_line_y = []
    right_line_x = []
    right_line_y = []

    for line in lines:
        print(line)
        for x1,y1,x2,y2 in line:
            slope = (y2 - y1) / (x2 - x1)
            print(slope)
            if math.fabs(slope) < 0.5:
                continue
            if slope <= 0:
                left_line_x.extend([x1,x2])
                left_line_y.extend([y1,y2])
            else:
                right_line_x.extend([x1,x2])
                right_line_y.extend([y1,y2])

    min_y = int(image.shape[0] * (3 / 5)) # just below the horizon
    max_y = int(image.shape[0])           # bottom of the image

    if len(left_line_y)>0:
        poly_left = np.poly1d(np.polyfit(
            left_line_y,
            left_line_x,
            deg = 1
        ))

        left_start_x = int(poly_left(max_y))
        left_end_x = int(poly_left(min_y))
    else:
        left_start_x = 0
        left_end_x = 0
  
    if(len(right_line_y)>0):
        poly_right = np.poly1d(np.polyfit(
            right_line_y,
            right_line_x,
            deg = 1
        ))
    
        right_start_x = int(poly_right(max_y)) 
        right_end_x = int(poly_right(min_y))
    else:
        right_start_x = 0
        right_end_x = 0

    line_image = draw_lines(image, 
            [[
                [left_start_x, max_y, left_end_x, min_y],
                [right_start_x, max_y, right_end_x, min_y],
            ]],
            thickness=3,
        )

    cv2.imshow("road lane marked image", line_image)
    k = cv2.waitKey(0)
    if k==27:
        cv2.destroyAllWindows()
    elif k==ord('s'):
        cv2.imwrite("road_lane_marked_image.jpg", line_image)
        cv2.destroyAllWindows()


# read the image
image = cv2.imread('road_lane2.jpg')
#image = cv2.resize(image, None, fx=.2, fy=.2, interpolation = cv2.INTER_AREA)
print(image.shape)

pipeline(image)
