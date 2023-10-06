from flask import Flask, flash, request, redirect, url_for, render_template
import urllib.request
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os
from werkzeug.utils import secure_filename
import tensorflow
from tensorflow.keras.preprocessing import image
import pickle
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50,preprocess_input
from sklearn.neighbors import NearestNeighbors
from numpy.linalg import norm
import tensorflow as tf

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads/'

# Define the folder where you want to save the image
output_folder = 'static/images/'

app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

# load the model
model=load_model(r'C:\Users\udays\Desktop\Data Science\projects\artproject\artproject\art.h5')

feature_list = np.array(pickle.load(open('embeddings.pkl','rb')))
filenames = pickle.load(open('filenames.pkl','rb'))

modelt = ResNet50(weights='imagenet',include_top=False,input_shape=(224,224,3))
modelt.trainable = False

modelt = tensorflow.keras.Sequential([
    modelt,
    GlobalMaxPooling2D()
])


def recommend(features,feature_list):
    neighbors = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='euclidean')
    neighbors.fit(feature_list)

    distances, indices = neighbors.kneighbors([features])

    return indices



def model_predict(img_path,model):
    test_image=image.load_img(img_path,target_size=(224,224))
    test_image=image.img_to_array(test_image)
    test_image=np.expand_dims(test_image,axis=0)
    test_image = tf.keras.applications.mobilenet_v2.preprocess_input(test_image)
    result=model.predict(test_image)
    return result


def feature_extraction(img_path,model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    result = model.predict(preprocessed_img).flatten()
    normalized_result = result / norm(result)

    return normalized_result



def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        flash('No image selected for uploading')
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        fpath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

        # Make prediction
        result = model_predict(fpath, model)
        print(result)

        categories = ['Foreign','Indian']
        # process your result for human
        pred_class = result.argmax()
        output = categories[pred_class]

        features = feature_extraction(fpath, modelt)
        # st.text(features)
        # recommendention
        indices = recommend(features, feature_list)

        output_img_path_list = []

        for i in range(1,5):
            image_path1 = filenames[indices[0][i]].replace("\\", "/")
            image = Image.open(image_path1)
            image_name = image_path1.split('/')[-1]
            output_image_path = os.path.join(output_folder, image_name)
            image.save(output_image_path)
            image.close()
            output_img_path_list.append(output_image_path)


        # print(indices)
        # image_path1 = filenames[indices[0][1]].replace("\\","/")
        # print(image_path1)
        # # Open the image using Pillow
        # image = Image.open(image_path1)
        #
        # image_name = image_path1.split('/')[-1]
        # # Define the output file path (including the desired filename)
        # output_image_path = os.path.join(output_folder, image_name)
        # # Save the image to the output folder
        # image.save(output_image_path)
        # image.close()
        # print(output_image_path)



        # print('upload_image filename: ' + filename)
        # E:\Art Images dataset\Art_Dataset_Clear\Indian
        flash('The Art you selected')
        return render_template('result.html', filename=filename, pred = output, image_path1=output_img_path_list[0], image_path2=output_img_path_list[1], image_path3=output_img_path_list[2], image_path4=output_img_path_list[3] )
    else:
        flash('Allowed image types are - png, jpg, jpeg, gif')
        return redirect(request.url)


@app.route('/display/<filename>')
def display_image(filename):
    # print('display_image filename: ' + filename)
    return redirect(url_for('static', filename='uploads/' + filename), code=301)


if __name__ == "__main__":
    app.run(debug=False,port=5590)