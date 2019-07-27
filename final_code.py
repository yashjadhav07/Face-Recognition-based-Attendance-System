import cv2
import numpy as np
import pickle
import datetime
from PIL import Image


face_cascade=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
#recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer = cv2.face.EigenFaceRecognizer_create()
recognizer.read("trainner.yml")
labels={"person_name":1}
with open("labels.pickle",'rb')as f:
    og_labels=pickle.load(f)
    labels={v:k for k,v in og_labels.items()}
data={}
currentDt1=datetime.datetime.now()
cap=cv2.VideoCapture(0)
while True:
    ret,frame=cap.read()
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces=face_cascade.detectMultiScale(gray,1.5, 5)
    for (x,y,w,h) in faces:
        end_cord_x= x+w
        end_cord_y = y + h

        cv2.rectangle(frame,(x,y),(end_cord_x,end_cord_y),(255,0,0),2)#(stroke=2)
        #print(x,y,w,h)
        roi_gray=gray[y:y+h, x:x+w]
        roi_color = frame[y:y + h, x:x + w]
        size = (220,220)
        roi_gray = cv2.resize(roi_gray,size)
        id_, conf=recognizer.predict(roi_gray)
        #print(conf)
        if conf>=5500:
            #print(conf)
            #print(id_)
           # print(labels[id_])
            font= cv2.FONT_HERSHEY_SIMPLEX
            name=labels[id_]
            color=(0,255,255)
            stroke=2
            cv2.putText(frame,name,(x,y),font,1,color,stroke,cv2.LINE_AA)
            currentDt = datetime.datetime.now()
            # print(name)
            if name in data:
                difference = currentDt - data[name]
                difference_in_seconds = difference.total_seconds()
                difference_in_minutes = divmod(difference_in_seconds, 60)[0]
                if (difference_in_minutes >= 5):
                    data[name] = currentDt


            else:
                data[name] = currentDt

        else:
            color = (0, 0, 255)
            stroke = 2
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(frame, "Unknown", (x, y), font, 1, color, stroke, cv2.LINE_AA)

        img_item="my_image.png"
        cv2.imwrite(img_item,roi_color)
    cv2.imshow('img',frame)
    k=cv2.waitKey(1)
    if k==32:
        break
file = open("database.txt", 'a')
for k in data:
    file.write(k)
    file.write(":")
    file.write(str(data[k]))
    file.write("\n")
file.close()
cap.release()
cv2.destroyAllWindows()