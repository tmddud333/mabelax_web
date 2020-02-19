import flask
from flask import Flask, request, render_template, send_file
import numpy as np
#from scipy import misc
from matplotlib.pyplot import imread
from flask_restful import Resource, Api
import torch
import cv2
import torch.nn.functional as F
from model import network



app = Flask(__name__)
api = Api(app)


def load_model(path):
    
    #myModel = torch.load(path)
    myModel = torch.load(path, map_location=lambda storage, loc: storage)
    myModel.eval()
    myModel.to(device)
    
    return myModel

def seg_process(image, net):

    # opencv
    origin_h, origin_w, c = image.shape
    image_resize = cv2.resize(image, (256, 256), interpolation=cv2.INTER_CUBIC)
    image_resize = (image_resize - (104., 112., 121.,)) / 255.0



    tensor_4D = torch.FloatTensor(1, 3, 256, 256)
    
    tensor_4D[0,:,:,:] = torch.FloatTensor(image_resize.transpose(2,0,1))
    inputs = tensor_4D.to(device)

    trimap, alpha = net(inputs)
  

    alpha_np = alpha[0,0,:,:].cpu().data.numpy()


    alpha_np = cv2.resize(alpha_np, (origin_w, origin_h), interpolation=cv2.INTER_CUBIC)

    #fg = np.multiply(alpha_np[..., np.newaxis], image)
    
    #bg = background

    # mask 추출 
    bg_gray = np.multiply(1-alpha_np[..., np.newaxis], image)
    bg_gray = cv2.cvtColor(bg_gray, cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(bg_gray, 50, 255, cv2.THRESH_BINARY)
    mask = mask.astype(np.uint8)
    mask_inv = cv2.bitwise_not(mask)

    # background 합성된 iamge
    #bg = cv2.bitwise_and(bg, bg, mask=mask)
    #fg = cv2.bitwise_and(image, image, mask=mask_inv)
    #out1 = cv2.add(bg,fg)
    #cv2.imwrite("./result/out1.png", out1)

    # backround 제거된 image (with alpha channel)
    b,g,r = cv2.split(image)
    rgba = [b,g,r,mask_inv]
    out2 = cv2.merge(rgba,4)
    cv2.imwrite("./result/final_out.png", out2)

    
   # bg[:,:,0] = bg_gray
   # bg[:,:,1] = bg_gray
   # bg[:,:,2] = bg_gray

    #fg[fg<=0] = 0
    #fg[fg>255] = 255
    #fg = fg.astype(np.uint8)
    #out = cv2.addWeighted(fg, 0.7, bg, 0.3, 0)
    
    #out = fg + bg
    #out[out<0] = 0
    #out[out>255] = 255
    #out = out.astype(np.uint8)

    return out2


# 메인 페이지 라우팅
@app.route("/")
@app.route("/index")
def index():
    return flask.render_template('index.html')




# 데이터 예측 처리
@app.route('/predict', methods=['POST'])
def make_prediction():
    if request.method == 'POST':
       
        # 업로드 파일 처리 분기
        file = request.files['image']
        if not file: return render_template('index.html')

        # # 이미지 픽셀 정보 읽기
        img = imread(file)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
       # print(img1)
       # print('-----------------------------------')
       # img2 = cv2.imread('./test.jpg',cv2.IMREAD_COLOR)
       # print(img2)
        
        #img = img[:, :, :3]
        

        # 입력 받은 이미지 예측
        #prediction = model.predict(img)
        frame_seg = seg_process(img, myModel)

        # 결과 리턴
        return render_template('index.html')
       # return send_file("./result/final_out.png", attachment_filename='output.png')


if __name__ == '__main__':
    torch.set_grad_enabled(False)
    device = torch.device('cpu')
    print('Using cpu')
    # 모델 로드
    # ml/model.py 선 실행 후 생성
    myModel = load_model('./model/model_obj.pth')
    #model = joblib.load('./model/model.pkl')
    # Flask 서비스 스타트
    app.run(port=8000, debug=True)
