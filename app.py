import os
from flask import Flask, request, redirect, url_for, send_from_directory, render_template
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, load_model
from werkzeug.utils import secure_filename
import numpy as np
import pickle
import cv2
from skimage import feature as ft



ALLOWED_EXTENSIONS = set(['jpg', 'jpeg', 'png'])
IMAGE_SIZE = (224, 224)
UPLOAD_FOLDER = 'uploads'
dic = {0 : 'NON-UPS', 1 : 'UPS'}
vgg16 = pickle.load(open('img_model.p', 'rb'))



def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


def hog_feature(img_array, resize=(256, 192)):
    img = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    img = cv2.resize(img, resize)
    bins = 9
    cell_size = (8, 8)
    cpb = (2, 2)
    norm = "L2"
    features = ft.hog(img, orientations=bins, pixels_per_cell=cell_size,
                      cells_per_block=cpb, block_norm=norm, transform_sqrt=True)
    return features


def predict(file):
    positive_img = cv2.imread(file)
    feature = hog_feature(positive_img)
    probs = vgg16.predict(feature.reshape(1, -1))
    return dic[probs[0]]


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route("/")
def template_test():
    return render_template('index.html', label='', imagesource='file://null')


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            output = predict(file_path)
    return render_template("index.html", label=output, imagesource=file_path)


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'],
                               filename)

if __name__ == "__main__":
    app.run(debug=False, threaded=False)
