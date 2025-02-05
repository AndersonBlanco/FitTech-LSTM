from flask import Flask, request, jsonify
import mediapipe as mp
import tensorflow as tf
# ... import your ML model
from vision import runPoseEstimation

def preprocess_image(image):
    # Resize the image to a fixed size (e.g., 256x256)
    image = tf.image.resize(image, (40, 8))

    # Normalize the pixel values to the range [0, 1]
    image = image / 255.0

    return image

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.data
    image = tf.image.decode_image(data, channels=3)
    image = preprocess_image(image)
    # ... process data with your ML model
    result = runPoseEstimation(image)

    image_bytes = tf.io.encode_jpeg(image).numpy()
    image_base64 = image_bytes.decode('utf-8')

    # Return the image as a JSON response
    return jsonify(image_base64)
 

if __name__ == '__main__':
    app.run(debug=True)