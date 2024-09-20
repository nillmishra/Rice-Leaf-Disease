from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

model = load_model('rice_leaf_disease_model.h5')

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/result', methods=['POST'])
def result():
    try:
        if 'file' not in request.files:
            return redirect(url_for('error', error_message="No file part"))
        
        file = request.files['file']

        if file.filename == '':
            return redirect(url_for('error', error_message="No selected file"))

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            img = image.load_img(filepath, target_size=(150, 150))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array /= 255.0  

            prediction = model.predict(img_array)
            disease_classes = ['Bacterial Blight','Brown Spot','Healthy', 'Blast','Scald','Narrow Brown Spot']
            predicted_class = disease_classes[np.argmax(prediction)]

            return render_template('result.html', filename=filename, prediction=predicted_class)
        else:
            return redirect(url_for('error', error_message="Invalid file format"))
    
    except Exception as e:
        return redirect(url_for('error', error_message=str(e)))

@app.route('/error')
def error():
    error_message = request.args.get('error_message', 'An error occurred')
    return render_template('error.html', error=error_message)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return redirect(url_for('static', filename='uploads/' + filename))

if __name__ == '__main__':

    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    app.run(debug=True)
