
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 14:55:23 2019

@author: Namitha
"""

import cv2
import numpy as np

fcascade=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
facedata=[]
facecount=0
cap=cv2.VideoCapture(0)
while True:
    ret,image=cap.read()
    faceGray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    faces=fcascade.detectMultiScale(faceGray,1.3,5)
    for (x,y,w,h) in faces:
       cropedface=image[y:y+h,x:x+w,:]
       resizedface=cv2.resize(cropedface,(50,50))
       if facecount%10==0 and len(facedata)<=20:
           facedata.append(resizedface)
    facecount+=1 
    cv2.putText(image,str(len(facedata)),(50,50),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,(255,0,0),2)      
    cv2.imshow("original",image)
    if cv2.waitKey(1)==27 or len(facedata)>20:
        break
    
facedata=np.array(facedata)
cap.release()
np.save("virat",facedata)
cv2.destroyAllWindows()