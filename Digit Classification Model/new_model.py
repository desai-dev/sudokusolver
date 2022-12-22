import os
import cv2
import numpy as np  
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
import matplotlib.pyplot as plt

# ###############################
path = 'C:/Users/Dev/Desktop/Sudoku Solver/Digit Classification Model/Digit Database'
# ###############################

images = []
class_num = []
myList = os.listdir(path)
num_of_classes = len(myList)

for i in range(10):
    myPicList = os.listdir(path+"/"+str(i))
    for j in myPicList:
        cur_img = cv2.imread(path+"/"+str(i)+"/"+j)
        cur_img = cv2.resize(cur_img, (28, 28))
        images.append(cur_img)
        class_num.append(i)

images = np.array(images)
class_num = np.array(class_num)

# #########################################
test_ratio = 0.2
val_ratio = 0.2
# #########################################

X_train, X_test, y_train, y_test = train_test_split(images, class_num, test_size=test_ratio)
X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=val_ratio)

for i in range(10):
    cv2.imshow("img", X_train[i])
    cv2.waitKey(0)
    print(y_train[i])

def preprocessing(img):
    img = cv2.resize(img, (28, 28))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2. equalizeHist(img)
    img = img/255
    return img

X_train = np.array(list(map(preprocessing, X_train)))
X_test = np.array(list(map(preprocessing, X_test)))
X_validation = np.array(list(map(preprocessing, X_validation)))

X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)
X_validation = X_validation.reshape(X_validation.shape[0], X_validation.shape[1], X_validation.shape[2], 1)

y_train = to_categorical(y_train, num_of_classes) # One Hot encode; this could be an error
y_test = to_categorical(y_test, num_of_classes)
y_validation = to_categorical(y_validation, num_of_classes)


model = Sequential()
model.add((Conv2D(60, (5,5), input_shape=(28, 28, 1), activation='relu')))
model.add((Conv2D(60, (5,5), activation='relu')))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add((Conv2D(30, (3, 3), activation='relu')))
model.add((Conv2D(30, (3, 3), activation='relu')))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dense(500, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy']) 

history = model.fit(X_train, y_train, epochs=3, validation_data=(X_validation, y_validation))

model.save('new_model.h5')
# img = cv2.imread('C:/Users/Dev/Desktop/Sudoku Solver/Digit Classification Model/sample.png')
# img = preprocessing(img)
# test =[]
# test.append(img)
# test = np.array(test)
# test = np.reshape(test, (1, 28, 28, 1))

# prediction = model.predict(test)
# print(prediction)

# plt.figure(1)
# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.legend(['training', 'validation'])
# plt.title('Loss')
# plt.xlabel('epoch')

# plt.figure(2)
# plt.plot(history.history['accuracy'])
# plt.plot(history.history['val_accuracy'])
# plt.legend(['training', 'validation'])
# plt.title('Accuracy')
# plt.xlabel('epoch')

# plt.show()

score = model.evaluate(X_test, y_test, verbose=0)
print('Test Loss = ', score[0])
print('Test Accuracy = ', score[1])