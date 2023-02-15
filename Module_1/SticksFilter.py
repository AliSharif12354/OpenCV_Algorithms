import cv2 as cv
import numpy as np

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
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
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
    cv.imshow("Original", image)
    cv.imshow("Sticks filtered", output)
    cv.waitKey(0)
    cv.destroyAllWindows()  
    return output;



#Get the image we want to run our sticks filter on
image = cv.imread("cat2.jpg")
sticksImage = sticksFilter(image)

cv.imshow("Original", image)
cv.imshow("Sticks filtered", sticksImage)