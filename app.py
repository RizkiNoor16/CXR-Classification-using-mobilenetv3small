from flask import Flask, render_template,request
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'images'

 # load model
model = load_model('model_mobilenet.h5')
classes = ['Normal', 'Covid-19', 'Pneumonia']

@app.route('/', methods=["GET"])
def hello():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def predict():
    # load image
    imagefile = request.files['imagefile']
    image_path = "./images/" + imagefile.filename
    imagefile.save(image_path)

    img = image.load_img(image_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = x/255.0
        
    # predict
    preds = model.predict(x)
        
    # get predicted class
    class_idx = np.argmax(preds[0])
    #get class label
    class_label = classes[class_idx]
        
    # get confidence level
    confidence = preds[0][class_idx]
        
    # render result template with predicted class and confidence level
    return render_template('index.html', class_name=class_label, confidence=confidence)

if __name__ == '__main__':
    app.run(debug=True)