import os
import uuid
from flask import Flask, flash, render_template, request, redirect, url_for
from onnx_utils import classify_car
from werkzeug.utils import secure_filename

app = Flask(__name__)

UPLOAD_FOLDER = os.path.join(app.root_path, 'uploads')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/classify',methods=['GET', 'POST'])
def classify():
    if request.method == 'GET':
        print("I am here")
        return render_template('classify_form.html') 
    elif request.method == 'POST':
        # Get the results from the post, and pass it to classify_car, and return the image to the template.
        if 'inputCarImg' not in request.files:
            # flash('No file part')
            return redirect(request.url)
        file = request.files['inputCarImg']
        if file.filename == '':
            # flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            uuid_filename = uuid.uuid4().hex + '.' + file.filename.rsplit('.', 1)[1]
            filename = os.path.join(app.config['UPLOAD_FOLDER'], uuid_filename)
            file.save(filename)
        return render_template('classify_results.html', output_img=classify_car(filename,uuid_filename),
                                   input_img=filename)

@app.route('/results',methods=['GET', 'POST'])
def results():

    img_file = request.form["inputCarImg"]
    return render_template('classify_results.html', output_img=classify_car(img_file),
                                   input_img=img_file)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS