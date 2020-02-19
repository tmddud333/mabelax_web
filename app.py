import flask
from flask import Flask, request, render_template
import numpy as np
#from scipy import misc
from matplotlib.pyplot import imread
from flask_restful import Resource, Api
import torch
import cv2
import torch.nn.functional as F
from model import network
import os
from werkzeug.utils import secure_filename


app = Flask(__name__)
api = Api(app)
app.debug = True;

@app.route("/")
@app.route("/index")
def index():
    return flask.render_template('index.html')

@app.rout("/upload", method=['POST', 'GET'])
def upload():
	if request.method == 'POST':
		f = request.files['uploadBtn']
		f.save(os.path.join('uploads', secure_filename(f.filename)))

def load_model(path):
    
    myModel = torch.load(path, map_location=lambda storage, loc: storage)
    myModel.eval()
    myModel.to(device)
    
    return myModel

def seg_process(image, net):

    origin_h, origin_w, c = image.shape
    image_resize = cv2.resize(image, (500, 500), interpolation=cv2.INTER_CUBIC)
    image_resize = (image_resize - (104., 112., 121.,)) / 255.0

    tensor_4D = torch.FloatTensor(1, 3, 500, 500)
    
    tensor_4D[0,:,:,:] = torch.FloatTensor(image_resize.transpose(2,0,1))
    inputs = tensor_4D.to(device)

    trimap, alpha = net(inputs)
  

    if args.without_gpu:
        alpha_np = alpha[0,0,:,:].data.numpy()
    else:
        alpha_np = alpha[0,0,:,:].cpu().data.numpy()


    alpha_np = cv2.resize(alpha_np, (origin_w, origin_h), interpolation=cv2.INTER_CUBIC)

    bg_gray = np.multiply(1-alpha_np[..., np.newaxis], image)
    bg_gray = cv2.cvtColor(bg_gray, cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(bg_gray, 50, 255, cv2.THRESH_BINARY)
    mask = mask.astype(np.uint8)
    mask_inv = cv2.bitwise_not(mask)

    b,g,r = cv2.split(image)
    rgba = [b,g,r,mask_inv]
    out2 = cv2.merge(rgba,4)
    cv2.imwrite("./result/out1.png", out2)

    return out2

@app.route('/predict', methods=['POST'])
def make_prediction():
    if request.method == 'POST':
        file = request.files['image']
        if not file: return render_template('index.html', ml_label="No Files")

        img = imread(file)
        print(img.shape)
        
        frame_seg = seg_process(img, myModel)

        label = '0'
        return render_template('index.html', ml_label=label)


if __name__ == '__main__':
    torch.set_grad_enabled(False)
    device = torch.device('cpu')
    print('Using cpu')

    myModel = load_model('./model/model_obj.pth')
  
    app.run(host="0.0.0.0", port=5000)
