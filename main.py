import cv2
import numpy as np
import utilis

######################################
__path__="D:\\DIP\\mcqsheet.png"
widthImg = 700
heightImg = 700
######################################

img = cv2.imread(__path__)

#IMG PREPROCESSING
img = cv2.resize(img,(widthImg,heightImg))
imgContours = img.copy()
imgBiggestContours = img.copy()
imgGray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
imgBlur= cv2.GaussianBlur(imgGray,(5,5),1)
imgCanny= cv2.Canny(imgBlur,10,50)

#TO FINDing all  CONTUOURS
countours,heirarchy= cv2.findContours(imgCanny,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
cv2.drawContours(imgContours,countours,-1,(0,255,0),10)

#FIND RECTANGLES
rectCon=utilis.rectContours(countours)
biggestContour = utilis.getCornerPoints(rectCon[0])
gradePoints=utilis.getCornerPoints(rectCon[1])
#print(biggestContour)
if biggestContour.size != 0 and gradePoints.size != 0:
    cv2.drawContours(imgBiggestContours,biggestContour,-1,(0,255,0),20)
    cv2.drawContours(imgBiggestContours,gradePoints,-1,(255,0,0),20)

    biggestContour= utilis.reorder(biggestContour)
    gradePoints= utilis.reorder(gradePoints)

    pt1 = np.float32(biggestContour)
    pt2 = np.float32([[0,0],[widthImg,0],[0,heightImg],[widthImg,heightImg]])
    matrix = cv2.getPerspectiveTransform(pt1,pt2)
    imgWarpColored= cv2.warpPerspective(img,matrix,(widthImg,heightImg))

    ptG1 = np.float32(gradePoints)
    ptG2 = np.float32([[0,0],[325,0],[0,150],[325,150]])
    matrixG = cv2.getPerspectiveTransform(ptG1,ptG2)
    imgGradeDisplay= cv2.warpPerspective(img,matrixG,(325,150))
    cv2.imshow("Grade",imgGradeDisplay)


imgBlank= np.zeros_like(img)
imgArray= ([img,imgGray,imgBlur,imgCanny],
           [imgContours,imgBiggestContours,imgWarpColored,imgBlank])
imgStacked = utilis.stackImages(imgArray,0.5)

cv2.imshow("Stacked Images",imgStacked )
# cv2.imshow("Original Image", img)
# cv2.imshow("Gray Image", imgGray)
# cv2.imshow("Blur Image", imgBlur)
# cv2.imshow("Canny Image", imgCanny)
cv2.waitKey(0)
