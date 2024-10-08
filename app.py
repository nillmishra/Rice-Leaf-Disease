from flask import Flask, render_template, request, redirect, url_for, flash, session
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os

app = Flask(__name__)
app.secret_key = 'supersecretkey'
model = tf.keras.models.load_model('advanced_rice_leaf_disease_model.h5')

UPLOAD_FOLDER = 'static/uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Function to check allowed file types
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Function to predict the class of the uploaded image
def predict_image(file_path):
    image_size = (150, 150)
    img = load_img(file_path, target_size=image_size)
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction, axis=1)
    return predicted_class

# Home route
@app.route('/')
def index():
    return render_template('index.html')

# Upload page route
@app.route('/upload')
def upload():
    return render_template('upload.html')

# About page route
@app.route('/about')
def about():
    return render_template('about.html')

# Contact page route
@app.route('/contact', methods=['GET', 'POST'])
def contact():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        message = request.form['message']
        flash('Message sent successfully!')
        return redirect(url_for('index'))
    
    return render_template('contact.html')

# Route to handle image prediction
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(url_for('upload'))
    
    file = request.files['file']
    
    if file.filename == '':
        flash('No selected file')
        return redirect(url_for('upload'))
    
    if file and allowed_file(file.filename):
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)
        
        try:
            predicted_class = predict_image(file_path)
            class_names = ['Bacterial Blight', 'Blast', 'Brown Spot', 'Tungro']
            result = class_names[int(predicted_class[0])]
            
            return render_template('result.html', result=result, file_path=file_path)
        
        except Exception as e:
            flash(f'Error occurred: {str(e)}')
            return render_template('error.html', error_message=str(e))

    else:
        flash('Allowed file types are png, jpg, jpeg')
        return redirect(url_for('upload'))

# Error page route
@app.route('/error')
def error():
    return render_template('error.html')

# Login page route
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        # Implement authentication logic here
        flash('Login successful!')
        return redirect(url_for('index'))
    
    return render_template('login.html')

# Signup page route
@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        # Implement user registration logic here
        flash('Signup successful! Please log in.')
        return redirect(url_for('login'))
    
    return render_template('signup.html')

if __name__ == '__main__':
    app.run(debug=True)
