import flask
from flask import Flask, request, make_response
import torch
import base64
import numpy as np
import cv2
import io
from PIL import Image
import json

model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
app = Flask(__name__)

@app.route('/camera', methods=['POST'])
def home():
    imgbase64 = request.data

    #img = base64.b64decode(imgbase64)
    img_arr = np.frombuffer(imgbase64, dtype=np.uint8)
 
    img = cv2.imdecode(img_arr, cv2.IMREAD_UNCHANGED) 
    #img = Image.open(io.BytesIO(img))
    #img = np.array(img)

    res = model(img)

    img = cv2.cvtColor(res.render()[0], cv2.COLOR_BGR2RGB)
    ret, buffer = cv2.imencode('.jpg', img)

    response = make_response(buffer.tobytes())
    response.status_code = 200
    response.mimetype = 'image/png'

    return response

@app.route('/route2', methods=['GET'])
def road2():
    return 'test2'

@app.route('/route1', methods=['GET'])
def route1():
    return 'test1'

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=4244, debug=True)
