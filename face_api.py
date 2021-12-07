import io
import os
import cv2
import json                    
import base64
import logging             
import numpy as np
from PIL import Image
import tensorflow as tf
from flask_cors import CORS
from tensorflow.keras import datasets, layers, models
from flask import Flask, request, jsonify, abort

os.environ["CUDA_VISIBLE_DEVICES"]="0"
app = Flask(__name__)          
CORS(app)
app.logger.setLevel(logging.DEBUG)
model = tf.keras.models.load_model('/tmp/model_train_8.h5')
classes = ["Real", "Fake","Altered"]
@app.route("/image", methods=['POST'])
def test_method():         
    # print(request.json)      
    if not request.json or 'image' not in request.json: 
        abort(400)
             
    # get the base64 encoded string
    im_b64 = request.json['image']


    img_bytes = base64.b64decode(im_b64.encode('utf-8'))
   
    nparr = np.fromstring(img_bytes, np.uint8)
    
    img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    shaped_img_np=np.expand_dims(img_np, axis=0) 
    print('img shape', shaped_img_np.shape)
    #print('img shape', shaped_img_np.shape[1], shaped_img_np.shape[2])
    
    if((shaped_img_np.shape[1] != 200) or (shaped_img_np.shape[2] != 200)):
        im_rgb = cv2.cvtColor(shaped_img_np[0], cv2.COLOR_BGR2RGB)
        im_temp = cv2.resize(im_rgb, (200, 200))
        print("Writing tmp-img.jpg to local current")
        cv2.imwrite("./tmp-img.jpg", im_temp)
        img = cv2.imread("./tmp-img.jpg")
        shaped_img_np = np.array([im_temp])
        os.remove("./tmp-img.jpg")
    print('img shape', shaped_img_np.shape[1], shaped_img_np.shape[2])
    #process img   
    shaped_img_np = shaped_img_np / 255.0
    prediction = model.predict(shaped_img_np)
    response = np.argmax(prediction)
    print("--------------------------------------------------------------")
    print("--------------------------------------------------------------")
    print(prediction)
    print("--------------------------------------------------------------")
    print(response)
    print("--------------------------------------------------------------")
    #print("The face is " + response)
    return '{"response":"'+classes[response]+'"}'
    # access other keys of json
    # print(request.json['other_key'])

    #result_dict = {'output': 'output_key'}
    r#eturn result_dict
  
  
def run_server_api():
    app.run(host='0.0.0.0', port=8893)
    
  
if __name__ == "__main__":     
    run_server_api()
