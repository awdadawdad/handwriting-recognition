
from PIL import Image
from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json
import numpy as np
import os
import matplotlib.pyplot as plt

from PIL import Image
import numpy as np
json_file=open('MNISTmodel.json','r')
loaded_model_json=json_file.read()
json_file.close()
model=model_from_json(loaded_model_json)
model.load_weights('MNISTmodel.h5')
print("loaded model done")

import serial
serialPort = "COM3"  # 串口
baudRate = 9600  # 波特率
ser = serial.Serial(serialPort, baudRate, timeout=0.5)
print("参数设置：串口=%s ，波特率=%d" % (serialPort, baudRate))


def plot_image(image):
  fig = plt.gcf()
  fig.set_size_inches(2,2)
  plt.imshow(image,cmap='binary')
  plt.show()

import cv2
 
cap=cv2.VideoCapture(0)
i=0
demo1=b"0"#将0转换为ASCII码方便发送
demo2=b"1"#同理
while(1):
    ret ,frame = cap.read()
    k=cv2.waitKey(1)
    if k==27:
        break
    elif k==ord('s'):
        ret,frame = cv2.threshold(frame,50,255,cv2.THRESH_BINARY)
        cv2.imwrite('C:/Users/gaiya/Desktop/test/'+str(i)+'.jpg',frame)
        print('shot')
        imm1_1= np.array(Image.open('C:/Users/gaiya/Desktop/test/'+str(i)+'.jpg').convert('L').resize([28,28]))
        imm1_2=255-imm1_1
        imm1_4=imm1_2/255
        plot_image(imm1_2)
        imm1_4exp=np.expand_dims(imm1_4,axis=2)
        imm1_4exp=np.expand_dims(imm1_4exp,axis=0)
        print("imm1_4xep shape:",imm1_4exp.shape)

        prediction_test=model.predict(imm1_4exp)
        prediction_test = np.argmax(prediction_test,1)
        print("Prediction is :",prediction_test[0])
        
        if prediction_test[0]==3:
            ser.write(demo2)
        else:
            ser.write(demo1)
            
        i+=1
    cv2.imshow("capture", frame)
cap.release()
cv2.destroyAllWindows()

