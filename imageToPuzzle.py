## Could make a neural net to see if a puzzle is straight up or if its like ilted and conditionally add a border to the sraight up one

from turtle import color
import cv2
import numpy as np

image_path = './tilted_puzzle.png'
image_height = 450
image_width = 450

# Preprocessing the image:

def preprocess(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.GaussianBlur(img_gray, (5, 5), 1)
    img_threshold = cv2.adaptiveThreshold(img_blur, 255, 1, 1, 11, 2)
    return img_threshold

img = cv2.imread(image_path)
img = cv2.resize(img, (image_width, image_height))
#img = cv2.copyMakeBorder(img, 40, 40, 40, 40, cv2.BORDER_CONSTANT, value=[0, 0, 0]) # adds border
img_blank = np.zeros((image_height, image_width, 3), np.uint8) #MIGHT NOT NEED THIS
img_threshold = preprocess(img)


# Finding Contours:

img_contour = img.copy()

img_big_contour = img.copy()
contours, hierarchy = cv2.findContours(img_threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
img_2 = cv2.drawContours(img_contour, contours, -1, (0, 255, 0), 3)

#cv2.imshow("hi", img_2)
#cv2.waitKey(8000)

# Finding sudoku puzzle from detected contours:

def biggest_contour(contours): # change this function up
    biggest = np.array([])
    max_area = 0
    for i in contours:
        area = cv2.contourArea(i)
        if area > 50:
            perimator = cv2.arcLength(i, True)
            approx = cv2.approxPolyDP(i, 0.02 * perimator, True) # Checks how many corners there are
            if area > max_area and len(approx) == 4: # compares area to max_area and checks if the shape we have has four corners
                biggest = approx
                max_area = area
    return biggest, max_area

def order_points(points):
    points = points.reshape((4, 2))
    new_points = np.zeros((4, 1, 2), dtype=np.int32)
    add = points.sum(1)
    new_points[0] = points[np.argmin(add)]
    new_points[3] = points[np.argmax(add)]
    diff = np.diff(points, axis=1)
    new_points[1] = points[np.argmin(diff)]
    new_points[2] = points[np.argmax(diff)]
    return new_points

biggest, max_area = biggest_contour(contours)

if biggest.size != 0:
    biggest = order_points(biggest)
    cv2.drawContours(img_big_contour, biggest, -1, (0, 255, 0), 10)
    pts1 = np.float32(biggest)
    pts2 = np.float32([[0, 0], [image_width, 0], [0, image_height], [image_width, image_height]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    image_warp_coloured = cv2.warpPerspective(img, matrix, (image_width, image_height))
    image_detected_digits = img_blank.copy() # Might not need this and anything to do with blank image
    image_warp_coloured = cv2.cvtColor(image_warp_coloured, cv2.COLOR_BGR2GRAY)

# Splitting Image into individual sqaures

def split_boxes(img):
    rows = np.vsplit(img, 9)
    boxes = []

    for r in rows:
        cols = np.hsplit(r, 9)
        for box in cols:
            boxes.append(box)
    return boxes

image_solved_digits = img_blank.copy()
boxes = split_boxes(image_warp_coloured)
cv2.imshow("box", boxes[4])
cv2.waitKey(10000)
# numbers = get_prediction(boxes, model)
# image_detected_digits  display_numbers(image_detected_digits, numbers, color=(255, 0, 255))
# numbers = np.asarray(numbers)
# position_array = np.where(numbers > 0, 0, 1)

