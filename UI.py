import numpy as np
import cv2 as cv 

#x1 = int(input("X Coordinate"))
#y1 = int(input("Y Coordinate"))
#Generating space:
img = np.zeros((900,900,3), dtype = 'uint8')
#Adding logo:
#logo = cv.imread('images.png')
x_off = 765
y_off = 0
#logo_resize = cv.resize(logo, (100,100))
#img[y_off:y_off+logo_resize.shape[0],x_off:x_off+logo_resize.shape[1]] = logo_resize

#horizontal & vertical gradient
#change 'i' to something else to use directly in detect.py
for i in range(0,255):
    cv.circle(img,(50+i,600),2,(0,0+i,255-i),1)
    cv.circle(img,(600,50+i),2,(0,0+i,255-i),1)
    cv.circle(img,(50+255+i,600),2,(0,255-i,0+i),1)
    cv.circle(img,(600,255+50+i),2,(0,255-i,0+i),1)
#position indicators
cv.circle(img,(500,600),5,(255,255,255),-3)
cv.circle(img,(600,500),5,(255,255,255),-3)
#Other elements
cv.putText(img,'Crane Running', (700,500), cv.FONT_HERSHEY_DUPLEX, 0.5,(255,255,255),1)
cv.circle(img, (690,495),10,(0,0,255),-1,)
cv.putText(img,'Crane Stationary', (700,525), cv.FONT_HERSHEY_DUPLEX, 0.5,(255,255,255),1)
cv.circle(img, (690,520),10,(0,255,0),-1)
#Coordinates display:
cv.putText(img,'Vertical Movement:',(700,400),cv.FONT_HERSHEY_DUPLEX,0.5,(255,255,255),1)
cv.putText(img,'Horizontal Movement:',(700,425),cv.FONT_HERSHEY_DUPLEX,0.5,(255,255,255),1)
#Speed Warning:
cv.putText(img,'Speed Warning!!!',(700,300),cv.FONT_HERSHEY_DUPLEX,0.5,(0,0,255),1)

cv.imshow('Interface',img)

cv.waitKey(0)
cv.destroyAllWindows()