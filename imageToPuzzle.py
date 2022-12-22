import cv2
import numpy as np
import tensorflow as tf
from sudokuSolver import *

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
img_blank = np.zeros((image_height, image_width, 3), np.uint8) 
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

def initialize_prediction_model():
    model = tf.keras.models.load_model('new_model.h5')
    return model

def get_prediction(boxes, model):
    result = []
    for image in boxes:
        # Prep Img
        img = np.asarray(image)
        img = img[4:img.shape[0] - 4, 4:img.shape[1] - 4]
        img = cv2.resize(img, (28, 28))
        img = img/255
        img = img.reshape(1, 28, 28, 1)
        # Get Prediction
        predictions = model.predict(img)
        class_index = np.argmax(predictions, axis=-1)
        probability_val = np.amax(predictions)
        #print(class_index, probability_val)
        # Save to Result
        if probability_val > 0.6:
            result.append(class_index[0])
        else:
            result.append(0)
    return result

model = initialize_prediction_model()

image_solved_digits = img_blank.copy()
boxes = split_boxes(image_warp_coloured)
numbers = get_prediction(boxes, model)

numbers = np.array(numbers)
board = np.reshape(numbers, (9,9))
print(board)
print(solve(board))