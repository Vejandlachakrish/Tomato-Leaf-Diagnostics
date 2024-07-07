from flask import Flask, render_template, request, redirect, url_for
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os
import uuid
import logging

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# Ensure the upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load TensorFlow model
model = tf.keras.models.load_model("trained_plant_disease_model.keras")


# Helper function for model prediction
def model_prediction(image):
    image = image.resize((128, 128))  # Resize image to match model input
    input_arr = np.array(image)  # Convert image to array
    input_arr = np.expand_dims(input_arr, axis=0)  
    # Convert single image to batch
    predictions = model.predict(input_arr)
    return np.argmax(predictions)  # Return index of max element


logging.basicConfig(level=logging.DEBUG)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/contact')
def contact():
    return render_template('contact.html')


@app.route('/handle_contact', methods=['POST'])
def handle_contact():
    name = request.form['name']
    email = request.form['email']
    message = request.form['message']
    a = 'Received contact form submission:'
    app.logger.info(f'{a} Name={name},Email={email},Message={message}')
    return redirect(url_for('contact'))


@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return redirect(request.url)    
    file = request.files['image']
    if file.filename == '':
        return redirect(request.url)
    if file:
        # Save the image to the upload folder
        filename = f"{uuid.uuid4().hex}.png"
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        image = Image.open(io.BytesIO(file.read()))
        image.save(file_path)
        result_index = model_prediction(image)
        # Define class names
        class_names = ['Tomato Bacterial spot',
                       'Tomato Early blight', 'Tomato Late blight',
                       'Tomato Leaf Mold', 'Tomato Septoria leaf spot',
                       'Tomato Spider mites Two spotted spider mite', 
                       'Tomato Target Spot',
                       'Tomato Yellow Leaf Curl Virus',
                       'Tomato mosaic virus',
                       'Tomato healthy']
        prediction = class_names[result_index]
        return render_template('result.html',
                               prediction=prediction, image_url=url_for
                               ('static', filename=f'uploads/{filename}'))    
    return redirect(url_for('home'))


if __name__ == '__main__':
    app.run(debug=True)
