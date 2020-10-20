#-*- coding: utf-8 -*-
import cv2
import sys
import gc
import json
import numpy as np
from keras.models import Sequential
from keras.models import model_from_json
root_path=r'C:\Users\user\Desktop\emotionDetect'
model_path=root_path+r'\model\\'
img_size=48
# emo_labels = ['angry','fear','happy','sad','surprise','neutral']
#load json and create model arch
emo_labels = ['angry', 'disgust:', 'fear', 'happy', 'sad', 'surprise', 'neutral']
num_class = len(emo_labels)
json_file=open(model_path+'model_json.json')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
#load weight
model.load_weights(model_path+'model_weight.h5')

       
#框住人脸的矩形边框颜色       
color = (0, 0, 2555)

#人脸识别分类器本地存储路径
cascade_path = r'C:\Users\user\Desktop\emotionDetect\haarcascade_frontalface_alt.xml'  

frame=cv2.imread(root_path+"\\ang.png")
frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#使用人脸识别分类器，读入分类器
cascade = cv2.CascadeClassifier(cascade_path)                

#利用分类器识别出哪个区域为人脸
faceRects = cascade.detectMultiScale(frame_gray, scaleFactor = 1.1,
                            minNeighbors = 1, minSize = (120, 120))        
if len(faceRects) > 0:                 
    for faceRect in faceRects: 
        x, y, w, h = faceRect
        images=[]
        rs_sum=np.array([0.0]*num_class)
        #截取脸部图像提交给模型识别这是谁
        image = frame_gray[y: y + h, x: x + w ]
        image=cv2.resize(image,(img_size,img_size))
        image=image*(1./255)
        images.append(image)
        images.append(cv2.flip(image,1))
        images.append(cv2.resize(image[2:45,:],(img_size,img_size)))
        for img in images:
            image=img.reshape(1,img_size,img_size,1)
            print("456465465")
            list_of_list = model.predict_proba(image,batch_size=32,verbose=1)#predict
            result = [prob for lst in list_of_list for prob in lst]
            rs_sum+=np.array(result)
        print(rs_sum)
        label=np.argmax(rs_sum)
        emo = emo_labels[label]
        print ('Emotion : ',emo)
        cv2.rectangle(frame, (x - 10, y - 10), (x + w + 10, y + h + 10), color, thickness = 2)
        font = cv2.FONT_HERSHEY_SIMPLEX
        #文字提示是谁
        cv2.putText(frame,'%s' % emo,(x + 30, y + 30), font, 1, (255,0,255),4)
cv2.imshow("OK!", frame)

cv2.waitKey(0)
cv2.destroyAllWindows()
