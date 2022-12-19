## Could make a neural net to see if a puzzle is straight up or if its like ilted and conditionally add a border to the sraight up one

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

cv2.imshow("hi", img_2)
cv2.waitKey(8000)
