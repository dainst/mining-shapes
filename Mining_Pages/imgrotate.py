import os
import imutils
import cv2

 

# Create an Image object from an Image
inputdir='X:/Projekte/iDAI.shapes/Mining_Shapes/INCOMING/testfpdf/ZenonID_000013934'
for image in os.listdir(inputdir):
    if image.endswith('.jpg'):
        print(image)
        colorImage  = cv2.imread(inputdir + '/' + image)
        rotated = imutils.rotate(colorImage, 180)
        cv2.imwrite(inputdir + '/' + image, rotated)

 

