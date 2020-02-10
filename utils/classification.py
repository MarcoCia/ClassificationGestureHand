import tensorflow as tf
import tflearn
from tflearn.layers.conv import conv_2d,max_pool_2d
from tflearn.layers.core import input_data,dropout,fully_connected
from tflearn.layers.estimator import regression
import numpy as np
from PIL import Image
import cv2
import imutils

# global variables
bg = None

def resizeImage(imageName):
    basewidth = 100
    img = Image.open(imageName)
    wpercent = (basewidth/float(img.size[0]))
    hsize = int((float(img.size[1])*float(wpercent)))
    img = img.resize((basewidth,hsize), Image.ANTIALIAS)
    img.save(imageName)

def copy(image):
    z = np.asarray(image)
    temp = np.zeros(shape=(z.shape[0],z.shape[1]))
    for i in range(z.shape[0]):
        for j in range(z.shape[1]):
            temp[i][j] = float(z[i][j])
    return temp


def run_avg(image, aWeight):
    global bg
    # initialize the background
    if bg is None:
        bg = copy(image)
        return

    # compute weighted average, accumulate it and update the background
    cv2.accumulateWeighted(image, bg, aWeight)

def back(image):
    im = np.zeros(shape=(image.shape[0], image.shape[1]))
    for i in range(im.shape[0]):
        for j in range(im.shape[1]):
            im[i][j] = 120
    return im


def segment(image, threshold=25):
    global bg
    # find the absolute difference between background and current frame
    """
    im = cv2.imread("paper.jpg")
    im = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)

    im = cv2.resize(im, (image.shape[1],image.shape[0]), interpolation=cv2.INTER_AREA)
    print(im.shape)
    print(image.shape)
    """
    im = back(image)

    diff = cv2.absdiff(im.astype("uint8"), image)

    # threshold the diff image so that we get the foreground
    thresholded = cv2.threshold(diff,
                                threshold,
                                255,
                                cv2.THRESH_BINARY)[1]
    """
    # get the contours in the thresholded image
    (_, cnts, _) = cv2.findContours(thresholded.copy(),
                                    cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)

    # return None, if no contours detected
    if len(cnts) == 0:
        return
    else:
        # based on contour area, get the maximum contour which is the hand
        segmented = max(cnts, key=cv2.contourArea)
        return (thresholded, segmented)
    """
    return thresholded

def getPredictedClass(model):
    # Predict
    image = cv2.imread('Temp.png')
    image = cv2.resize(image,(89,100), interpolation= cv2.INTER_AREA)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    print("Gra: ",gray_image.shape)
    prediction = model.predict([gray_image.reshape(89, 100, 1)])
    return np.argmax(prediction), (np.amax(prediction) / (prediction[0][0] + prediction[0][1] + prediction[0][2]))


def prediction(frame, p1, p2, model):



    frame = cv2.flip(frame, 1)
    top, right, bottom, left = p1[1], p2[0], p2[1], p1[0]

    #print(top,bottom,left,right)

    roi = frame[top:bottom, left:right]

    """
    print("p1",p1)
    print("p2",p2)
    print("ROI",roi)
    print("frame",frame.shape)
    """
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)

    #run_avg(gray, 0.5)

    # segment the hand region
    thresholded = segment(gray)

    # check whether hand region is segmented



    #(thresholded, segmented) = hand
        # if yes, unpack the thresholded image and
        # segmented region
        # draw the segmented region and display the frame
    #cv2.drawContours(frame, [segmented + (right, top)], -1, (0, 0, 255))
    cv2.imwrite('Temp.png', thresholded)
    resizeImage('Temp.png')
    predictedClass, confidence = getPredictedClass(model)
    predizione = showStatistics(predictedClass, confidence)

    return predizione

               
def showStatistics(predictedClass, confidence):

    textImage = np.zeros((300,512,3), np.uint8)
    className = ""

    if predictedClass == 0:
        className = "Swing"
    elif predictedClass == 1:
        className = "Wave"
    else:
        className = ""
    return className
	


