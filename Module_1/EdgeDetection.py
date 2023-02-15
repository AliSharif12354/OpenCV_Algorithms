import cv2 as cv
import numpy as np
import math

#Ali Hassan Sharif
#101142782
#COMP 4102 Assignment 1

#nonMaximumSuppression takes in magnitude gradient image, orientation gradient image, and an integer threshold to apply maximum suppression
def nonMaximumSuppression (magSobel, oriSobel, thresh):
    temp = np.copy(magSobel)
    #Loop through image
    for i in range(thresh, len(magSobel) - thresh):
        for j in range(thresh, len(magSobel[i]) - thresh):
            #Each if statement is for a different case of angle rounding, and we apply non maximum suppression in the correct orientation of the magnitutde gradient
            if(oriSobel[i, j] >= 0 and oriSobel[i, j] < 22.5): 
                if((magSobel[i, j] > magSobel[i, j + thresh]) and (magSobel[i, j] > magSobel[i, j - thresh])):
                    temp[i, j] = magSobel[i, j]
                else:
                    temp[i, j] = 0

            if(oriSobel[i, j] >= 22.5 and oriSobel[i,j] < 67.5): 
                if((magSobel[i,j] > magSobel[i + thresh,j + thresh]) and (magSobel[i, j] > magSobel[i- thresh, j - thresh])):
                    temp[i,j] = magSobel[i,j]
                else:
                    temp[i,j] = 0

            if(oriSobel[i,j] >= 67.5 and oriSobel[i,j] < 112.5): 
                if((magSobel[i,j] > magSobel[i + thresh,j]) and (magSobel[i,j] > magSobel[i - thresh,j])):
                    temp[i,j] = magSobel[i,j]
                else:
                    temp[i,j] = 0

            if(oriSobel[i,j] >= 112.5): 
                if((magSobel[i,j] > magSobel[i - thresh, j + thresh]) and (magSobel[i , j] > magSobel[i + thresh, j - thresh])):
                    temp[i,j] = magSobel[i,j]
                else:
                    temp[i,j] = 0
            
    return temp

#Generates 8 stick kernals of size 5x5
def createSticks():
    one = np.array([[1/5, 0, 0, 0, 0],
                [0, 1/5, 0, 0, 0],
                [0, 0, 1/5, 0, 0],
                [0, 0, 0, 1/5, 0],
                [0, 0, 0, 0, 1/5],])
    two = np.array([[0, 0, 0, 0, 0],
                [1/5, 1/5, 0, 0, 0],
                [0, 0, 1/5, 0, 0],
                [0, 0, 0, 1/5, 1/5],
                [0, 0, 0, 0, 0],])
    three = np.array([[0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [1/5, 1/5, 1/5, 1/5, 1/5],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],])
    four = np.array([[0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1/5],
                [0, 1/5, 1/5, 1/5, 0],
                [1/5, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],])
    five = np.array([[0, 0, 0, 0, 1/5],
                [0, 0, 0, 1/5, 0],
                [0, 0, 1/5, 0, 0],
                [0, 1/5, 0, 0, 0],
                [1/5, 0, 0, 0, 0],])
    six = np.array([[0, 0, 0, 1/5, 0],
                [0, 0, 0, 1/5, 0],
                [0, 0, 1/5, 0, 0],
                [0, 1/5, 0, 0, 0],
                [0, 1/5, 0, 0, 0],])
    seven = np.array([[0, 0, 1/5, 0, 0],
                [0, 0, 1/5, 0, 0],
                [0, 0, 1/5, 0, 0],
                [0, 0, 1/5, 0, 0],
                [0, 0, 1/5, 0, 0],])
    eight = np.array([[0, 1/5, 0, 0, 0],
                [0, 1/5, 0, 0, 0],
                [0, 0, 1/5, 0, 0],
                [0, 0, 0, 1/5, 0],
                [0, 0, 0, 1/5, 0],])
    return np.array([one, two, three, four, five, six, seven, eight])

#Get the neighbourhood of (5x5 matrix) at pixel (x,y)
def getN(img, x, y): 
    temp = np.ones((5,5), np.float32)
    for i in range (x-2, x+2):
        for j in range(y-2, y+2):
            temp[i - x + 2][j - y + 2] = img[i][j]
    return temp

#Apply sticks filter to a given image
def sticksFilter(img):
    sticks = createSticks() #generate stick kernals
    output = np.copy(img) #Output variable
    for i in range(5, len(img) - 5):
        for j in range(5, len(img[i]) - 5):
            difference = 0; #Variable to keep track of which sticks kernel has the maximum difference
            stick = 0; #Variable to keep track of which stick kernel is the best fit
            for x in range(len(sticks)): 
                imgN = getN(img, i, j) #Get the 5x5 neighbourhood of the img[i][j] pixel 
                temp = np.ones((5, 5)) 

                #the following calculations are to find the maximum difference sticks kernel
                for k in range(0, 5):
                    for l in range(0, 5):
                        temp[k][l] = sticks[x][k][l] * imgN[k][l] 
                diff = np.subtract(temp, imgN)
                if(np.abs((np.sum(diff))) > difference):
                    difference = np.sum(diff)
                    stick = sticks[x]

            #The following calculations are to use the stick kernal we found and improve contrast
            for k in range(i - 2, i + 2):
                for l in range(j - 2, j + 2):
                    #Check if the stick kernal is 0 at current kernel to improve contrast
                    #+100 and -100 are arbitrary values I picked, specifications did not mention how to contrast specifically
                    if (stick[k - i + 2][l - j + 2] != 0): 
                        output[k][l] = img[k][l] + 100
                    else:
                        output[k][l] = img[k][l] - 100

    return output;


def edgeDetection (img0, sigma):
    hsize = 2* math.ceil(3 * sigma) + 1 #Calculate kernal size
    img0 = cv.cvtColor(img0, cv.COLOR_BGR2GRAY) #Make image black and white
    smoothImg = cv.GaussianBlur(img0, (hsize, hsize), sigma, sigma) #Apply gaussian blur

    #Initialize sobel kernals
    xSobel = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    ySobel = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

    #Apply sobel kernals
    xSobImg = cv.filter2D(smoothImg, cv.CV_16S, xSobel)
    ySobImg = cv.filter2D(smoothImg, cv.CV_16S, ySobel)
   
    #Create magnitude gradient image and orientation gradient image by copying dimensions of x sobel
    magSobel = np.copy(xSobImg)
    oriSobel = np.copy(xSobImg)
    
    magSobel = np.hypot(xSobImg, ySobImg)
    oriSobel = np.abs(np.degrees(np.arctan2(ySobImg, xSobImg))/2)
    
    #Apply Sticks filter
    magSobelSticks = sticksFilter(magSobel)

    #Apply non maximum suppression
    edgesSticks = nonMaximumSuppression(magSobelSticks, oriSobel, 1)
    edgesOriginal = nonMaximumSuppression(magSobel, oriSobel, 1)
    #edges = cv.Canny(smoothImg, 10, 10)

    #All the displaying and saving of images
    cv.imshow("original", img0)

    cv.imshow("Gradient magnitude image", magSobel)

    cv.imshow("Gradient orientation image", oriSobel)

    cv.imshow("Edge detection", edgesOriginal)

    cv.imshow("Magnitute sticks", magSobelSticks)

    cv.imshow("Sticks with non maximum suppression", edgesSticks)
    #cv.imshow("edges", edges)
    cv.waitKey(0)
    cv.destroyAllWindows()  

image = cv.imread("cat2.jpg")

#this function does edge detection
edgeDetection(image, 1)