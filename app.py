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
filename = 'models/model.pkl'
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
    app.run(host="0.0.0.0", port=8000, debug=True)
    # serve(app, host="0.0.0.0", port=8000,debug=True)

# for uploading the model to git need to install git LFS since it's a big file (steps in notion)
'''
https://github.com/shankarj67/Water-classifier-fastai/blob/master/app.py
https://github.com/danielchang2002/DogBreedClassification/blob/main/app.py
https://medium.com/@bijil.subhash/image-classifier-deployment-on-heroku-using-fastai-flask-and-node-js-70ad7057efc2
https://github.com/ryanmark1867/fastai_deployment/blob/main/web_flask_deploy.py
https://github.com/jakerieger/FlaskIntroduction/blob/master/app.py
https://github.com/bigyankarki/experiments_by_bigyan/blob/master/templates/index.html
https://forums.fast.ai/t/how-to-deploy-a-deep-learning-model-to-google-app-engine-using-flask-api-free-step-by-step-guide-for-beginners/74159


postman: https://web.postman.co/workspace/My-Workspace~a6f8252d-9139-4f35-89c2-41d412515550/request/create?requestId=80ebe32d-b912-4e41-a62e-917783b5d3fd

'''