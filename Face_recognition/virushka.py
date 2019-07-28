# -*- coding: utf-8 -*-
"""
Created on Sun Jul 28 15:37:20 2019

@author: User
"""

import cv2
import numpy as np

fcascade=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

names={0:"virat",1:"anushka"}
person1=np.load("virat.npy").reshape(21,50*50*3)
person2=np.load("anushka.npy").reshape(21,50*50*3)

data=np.concatenate([person1,person2])
labels=np.zeros((42,1))
labels[21:,:]=1

from sklearn.neighbors import KNeighborsClassifier
clf= KNeighborsClassifier(n_neighbors=5)
clf.fit(data,labels)
cap=cv2.VideoCapture(0)
while True:
    ret,image=cap.read()
    imgray=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    faces=fcascade.detectMultiScale(imgray,1.3,5)
    for (x,y,w,h) in  faces:
        cropedface=image[y:y+h,x:x+w,:]
        resizedface=cv2.resize(cropedface,(50,50))
        reshapedface=resizedface.reshape(1,50*50*3)
        pred=clf.predict(reshapedface)
        name=names[int(pred)]
        cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)
        cv2.putText(image,name,(x,y),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,255),2)
    cv2.imshow("output",image)
    if cv2.waitKey(1)==27:
        cap.release()
        cv2.destroyAllWindows()
        break
    