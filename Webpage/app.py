import os
from flask import Flask, request, render_template, redirect, url_for
from werkzeug.utils import secure_filename
import numpy as np
import cv2
import os

app = Flask(__name__)

# Define the directory where uploaded images will be stored
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'gif'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER





# Load the saved weights and biases from the binary file
loaded_network = np.load('neural_network_weights.npy', allow_pickle=True).item()

# Access the loaded weights and biases for the first and second sets
loaded_weights_1 = loaded_network['weights_1']
loaded_biases_1 = loaded_network['biases_1']

loaded_weights_2 = loaded_network['weights_2']
loaded_biases_2 = loaded_network['biases_2']

class_number_dict = loaded_network['class_number_dict']

def ReLU(Z):
    return np.maximum(Z, 0)

def softmax(Z):
    A = np.exp(Z) / sum(np.exp(Z))
    return A

def predict(w1, b1, w2, b2, X):
    Z1 = w1.dot(X) + b1
    A1 = ReLU(Z1)
    Z2 = w2.dot(A1) + b2
    A2 = softmax(Z2)
    return np.argmax(A2, 0)

def get_prediction(filename):
        image = cv2.imread(filename, 0)
        image = cv2.resize(image, (32, 32), interpolation= cv2.INTER_LINEAR)
        image = np.array(image)
        image = image.reshape(-1, 1024)

        X_test = image.T
        X_test = X_test / 255.

        prediction = predict(loaded_weights_1, loaded_biases_1, loaded_weights_2, loaded_biases_2, X_test)

        for x in class_number_dict.items():
            if (int(x[1]) == int(prediction)):
                return x[0]





# Function to check if a file has an allowed extension
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():

    if 'file' not in request.files:
        return redirect(request.url)

    file = request.files['file']

    if file.filename == '':
        return redirect(request.url)

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        
        pred = get_prediction(os.path.join(app.config['UPLOAD_FOLDER'], filename))

        os.remove(os.path.join(app.config['UPLOAD_FOLDER'], filename))

        return f'The Predicted ASL Sign is: {pred}'
    else:
        return 'Invalid file format. Allowed formats are jpg, jpeg, png, and gif.'

if __name__ == '__main__':
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    app.run(debug=True)
