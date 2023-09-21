import os
from flask import Flask, request, render_template, redirect
from werkzeug.utils import secure_filename
import os
import predict

# Get the current directory
current_dir = os.path.dirname(os.path.realpath(__file__))

# Create a Flask app with the template folder set to the current directory
app = Flask(__name__, template_folder=current_dir)

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
        file.save(filename)
        
        pred = predict.get_prediction(filename)
        #os.remove(filename)

        return render_template('index.html', prediction=pred, uploaded_image=filename)
    else:
        return render_template('index.html', error='Invalid file format. Allowed formats are jpg, jpeg, png, and gif.')

# Function to check if a file has an allowed extension
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'jpg', 'jpeg', 'png', 'gif'}

if __name__ == '__main__':
    app.run(debug=True)
