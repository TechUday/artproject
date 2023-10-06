from flask import Flask,render_template,request
import numpy as np
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from werkzeug.utils import secure_filename
from keras.models import load_model
from keras.preprocessing.image import img_to_array
from keras.preprocessing import image
from keras.preprocessing import image
from keras.models import load_model
import matplotlib.pyplot as plt
from skimage.io import imread
import cv2
import numpy as np
import pickle
from pathlib import Path
import tensorflow as tf
import time

# load the model
model=load_model(r'C:\Users\udays\Desktop\Data Science\projects\artproject\artproject\art.h5')


image_dir = Path(r'E:\Art Images dataset\testing images')

feature_list = np.array(pickle.load(open('embeddings.pkl','rb')))
filenames = pickle.load(open('filenames.pkl','rb'))

# Get filepaths and labels
filepaths = list(image_dir.glob(r'*.jpg'))

def model_predict(img_path,model):
    test_image=image.load_img(img_path,target_size=(224,224))
    test_image=image.img_to_array(test_image)
    test_image=np.expand_dims(test_image,axis=0)
    test_image = tf.keras.applications.mobilenet_v2.preprocess_input(test_image)
    result=model.predict(test_image)
    return result


for file in filepaths:

    frame =  imread(file) #cap.read()   #cap.retrieve() #
    # test_image = image.img_to_array(frame,target_size=(224,224))
    # test_image = test_image / 255
    # test_image=image.img_to_array(test_image)
    # test_image = np.expand_dims(test_image, axis=0)
    # result = classifier.predict(test_image)
    print(frame)

    # Make prediction
    result = model_predict(file, model)
    categories = ['Foreign', 'Indian']
    # process your result for human
    pred_class = result.argmax()
    output = categories[pred_class]




    # frame = cv2.rectangle(frame, (80, 80), (460, 424), (0, 255, 0), 2)
    frame = cv2.putText(frame,output + ' Art' , (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
    frame = cv2.resize(frame, (500, 500))






    # if face_extractor(frame) is not None:
    #     count +=1
    #     face=cv2.resize(face_extractor(frame),(400,400))
    #
    #     file_name_path = r'C:\Users\Dell\Desktop\Data Science\projects\TransferLearning facerecog\images\train\krishna/'+str(count)+'.jpg'
    #     cv2.imwrite(file_name_path,face)
    #
    #     #put count on the images and display live count
    #     cv2.putText(face,str(count),(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
    #     cv2.imshow('Face Cropper',face)
    #
    # else:
    #     print("Face Not Found")
    #     pass

    cv2.imshow('Art Detection', frame)

    if cv2.waitKey(1) & 0xFF== ord('q'):
        break

    time.sleep(1)

# cap.release()
cv2.destroyAllWindows()