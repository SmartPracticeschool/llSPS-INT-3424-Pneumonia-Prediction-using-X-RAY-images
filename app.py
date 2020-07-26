import numpy as np
import os
from keras.models import load_model
from keras.preprocessing import image
import tensorflow as tf
global graph
graph = tf.get_default_graph()
from flask import Flask , request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

app = Flask(__name__)
model = load_model("cn.h5")

@app.route('/', methods=['GET'])
def index():
    return render_template('base.html')

@app.route('/predict',methods = ['GET','POST'])
def upload():
    if request.method == 'POST':
        f = request.files['image']
        print("current path")
        basepath = os.path.dirname(__file__)
        print("current path", basepath)
        file_path = os.path.join(basepath,'uploads',secure_filename(f.filename))
        print("upload folder is ", file_path)
        f.save(file_path)
        
        img = image.load_img(file_path,target_size = (64,64))
        x = image.img_to_array(img)
        x = np.expand_dims(x,axis=0)
        
        with graph.as_default():
            preds = model.predict_classes(x)
            
            print("prediction",preds)
            
        index = ['normal','pneumonia']
        
        text = "the predicted Report is " + index[preds[0][0]]
                                                
        
    return text
if __name__ == '__main__':
    app.run(debug = True, threaded = False)
        
        
        
    
    
    