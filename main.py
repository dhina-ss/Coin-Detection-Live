from tensorflow.keras.preprocessing import image
from keras.models import load_model
from PIL import Image
import numpy as np
import time
import cv2

model = load_model('coin.h5')
classes = ['1 Rupee', '10 Rupee', '2 Rupee', '5 Rupee']

def predict_image(filename):
    img_ = image.load_img(filename, target_size=(227, 227))
    img_array = image.img_to_array(img_)
    img_processed = np.expand_dims(img_array, axis=0)
    img_processed /= 255.
    prediction = model.predict(img_processed)
    # print(prediction)
    index = np.argmax(prediction)
    return str(classes[index])

capture_duration = 60
cam = cv2.VideoCapture(0)
start_time = time.time()
font = cv2.FONT_HERSHEY_SIMPLEX
org = (50, 50)
fontScale = 1
color = (225, 0, 0)
thickness = 2
while(int(time.time() - start_time) < capture_duration):

    success,img=cam.read()
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    if success:
        cv2.imwrite('image.jpg',img)
        result = predict_image('image.jpg')
    image1 = cv2.putText(img, result, org, font, fontScale, color, thickness, cv2.LINE_AA)
    cv2.imshow("Coin",image1)
    if(cv2.waitKey(1)==ord('q')):
        break

cam.release()
cv2.destroyAllWindows()