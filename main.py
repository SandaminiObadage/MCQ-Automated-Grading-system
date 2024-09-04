import cv2
import numpy as np
import utilis

######################################
__path__="C:/Mcq-DIP/images/mcqsheet.png"
widthImg = 700
heightImg = 700
######################################

img = cv2.imread(__path__)

#IMG PREPROCESSING
img = cv2.resize(img,(widthImg,heightImg))
imgContours = img.copy()
imgGray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
imgBlur= cv2.GaussianBlur(imgGray,(5,5),1)
imgCanny= cv2.Canny(imgBlur,10,50)

#TO FINDing all  CONTUOURS
countours,heirarchy= cv2.findContours(imgCanny,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
cv2.drawContours(imgContours,countours,-1,(0,255,0),10)

#FIND RECTANGLES
utilis.rectContours(countours)
imgBlank= np.zeros_like(img)
imgArray= ([img,imgGray,imgBlur,imgCanny],
           [imgContours,imgBlank,imgBlank,imgBlank])
imgStacked = utilis.stackImages(imgArray,0.5)

cv2.imshow("Stacked Images",imgStacked )
# cv2.imshow("Original Image", img)
# cv2.imshow("Gray Image", imgGray)
# cv2.imshow("Blur Image", imgBlur)
# cv2.imshow("Canny Image", imgCanny)
cv2.waitKey(0)
