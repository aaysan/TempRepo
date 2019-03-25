#use FLASK_APP=recognize_faces.py flask run

from flask import Flask
from flask import request
import requests
import shutil
import os
import subprocess
import time
import tensorflow as tf
import facenet
import classify_faces
import cv2 as cv
from PIL import Image
import numpy as np
from mtcnn.mtcnn import MTCNN


app = Flask(__name__)
sess = None
face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')


@app.route("/")
def hello():
    return "Hello World!"


@app.route('/get_name', methods=['POST'])
def get_name():


    # print(sess)
    t_start = time.time()
    os.system("rm -f aligned/somename/*")

    file = request.files['file']


    file.save("aligned/somename/" + file.filename)

    t_downloaded = time.time()
    t_init = time.time()
    img = cv.imread(file.filename)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    ## TODO: add try catch statements
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    print(faces)

    x = faces[0][0]
    y = faces[0][1]
    w = faces[0][2]
    h = faces[0][3]

    roi_color = img[y:y + h, x:x + w]

    res = cv.resize(roi_color, dsize=(160, 160), interpolation=cv.INTER_CUBIC)
    # cv.imwrite("res.jpg", res)

    # detector = MTCNN()
    #
    # result = detector.detect_faces(im)
    # try:
    #     bounding_box = result[0]['box']  # this is the bounding box around the face
    #
    # except:
    #     return "Image had no face detected."
    #     return ""
    #
    # x1 = bounding_box[0]
    # x2 = bounding_box[0] + bounding_box[2]
    # y1 = bounding_box[1]
    # y2 = bounding_box[1] + bounding_box[3]
    # crop_img = im[y1:y2, x1:x2]
    # print(crop_img.shape)
    # # final_img = np.reshape(crop_img,(160,160,3))
    # res = cv2.resize(crop_img, dsize=(160, 160), interpolation=cv2.INTER_CUBIC)

    cv.imwrite("aligned/somename/tmp.png", res)
    # t1 = time.time()

    # print(t1 - t0)

    t_fin = time.time()
    t0 = time.time()
    a, b = classify_faces.classify_face(sess,graph,images_placeholder,embeddings,phase_train_placeholder,embedding_size)

    t1 = time.time()

    print("Total time: ", str(t1-t0))

    res = "Name: " + str(a) + "\n" + \
        "probability of it being correct: " + str(b[0]) + "\n" + \
        "time for classification: %0.3f seconds\n" % (t1-t0) + \
        "time for face alignment: %0.3f seconds\n" % (t_fin-t_init) + \
        "time for acquiring the image: %0.3f seconds\n" % (t_downloaded - t_start)

    return res

if __name__ == '__main__':
    currentdir = os.getcwd()
    pythonpath = currentdir ##+ "/facenet/src"
    os.environ["PYTHONPATH"] = pythonpath
    model = "../../pretrained_model/"

    graph = tf.Graph()

    # with graph.as_default():
    with tf.Session() as sess:

        facenet.load_model(model)
        images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
        embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
        phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
        embedding_size = embeddings.get_shape()[1]
        print("loaded")
        app.run(debug=False)