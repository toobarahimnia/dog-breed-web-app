from flask import Flask, render_template, request, jsonify, url_for
# import fastbook
from fastbook import *
from fastai import *
from fastai.vision import *
from werkzeug.utils import secure_filename
import os
from waitress import serve


# Define a flask app
app = Flask(__name__)
root = os.getcwd()
filename = 'model.pkl'
learn = load_learner(os.path.join(root, filename))


@app.route('/')
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    # if request.method == 'POST':
    #     prediction = single_image_predict(request.files['image'])
    #     final_pred = str(prediction[0])
    # return render_template('result.html', prediction=final_pred,
    #                                 comment='asd')

    prediction = single_image_predict(request.files['uploaded-file'])
    prediction_dict = json.loads(prediction.data.decode('utf-8'))  # Convert bytes to string and then to dict

    return render_template('result.html', percentage=prediction_dict['probability'], prediction=prediction_dict['dog_type'])
 
    # return single_image_predict(request.files['image'])
    
#function to predict image
def single_image_predict(image):
    img_PIL = Image.open(image)
    img = tensor(img_PIL) #converting PIL image to tensor
    learn_inf = learn.predict(img)
    return jsonify({'dog_type': learn_inf[0][0:],
                    'probability': str(round(max(learn_inf[2].numpy()*100), 2))})


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)

