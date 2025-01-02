from flask import Flask,render_template,request
from PIL import Image 
app = Flask(__name__)
import cv2
from keras.preprocessing import image
from keras.models import load_model
import numpy as np

@app.route('/')
def hello_world():
    return render_template('here.html')

@app.route('/model_say',methods=['POST'])
def model_say():
    f = request.files['you']
    f.save("tmp.png")   
    
    img = image.load_img("tmp.png",target_size = (48,48),color_mode = "grayscale")
    img = np.array(img)

    img = np.expand_dims(img,axis = 0) #makes image shape (1,48,48)
    input_array = img.reshape(1,48,48,1)


    model=load_model("../model.h5")
    
    result=model.predict([input_array])
    result = list(result[0])

    label_dict = {0:'Angry',1:'Disgust',2:'Fear',3:'Happy',4:'Neutral',5:'Sad',6:'Surprise'}
    
    img_index = result.index(max(result))
    

    return {"result":label_dict[img_index]}


if __name__ == '__main__':
   app.run(host='0.0.0.0', port=3000)
