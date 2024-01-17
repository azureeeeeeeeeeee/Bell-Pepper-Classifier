from flask import Flask, url_for, request, render_template
import tensorflow as tf
import numpy as np
import os

app = Flask(__name__)
model = tf.keras.models.load_model('Model/model.h5')

@app.route('/')
def main():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the uploaded file from the request
    file = request.files['file']

    # Save the file to a temporary location
    file_path = 'temp.jpg'
    file.save(file_path)

    # Load the image using Keras preprocessing
    img = tf.keras.preprocessing.image.load_img(file_path, target_size=(100, 100))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalize the image

    # Make predictions using your TensorFlow model
    predictions = model.predict(img_array)

    # Assuming your model outputs a binary prediction
    result = 'Infected' if predictions[0][0] > 0.5 else 'Healthy'

    # Clean up: remove the temporary file
    os.remove(file_path)

    return str(predictions[0][0])

if __name__=='__main__':
    app.run(debug=True)