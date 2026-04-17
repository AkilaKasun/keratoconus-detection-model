from flask import Flask, request, jsonify
from flask_cors import CORS  #
import tensorflow as tf
from PIL import Image
import numpy as np
import io

app = Flask(__name__)
CORS(app) # This prevents "CORS Errors" 


model = tf.keras.models.load_model('keratoconus_model.h5')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
    
    file = request.files['image']
    image = Image.open(io.BytesIO(file.read())).convert('RGB')
    
    # Pre-process image (same as your Streamlit/Colab code)
    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    # Predict
    prediction = model.predict(img_array)
    classes = ['Keratoconus', 'Normal', 'Suspect']
    result = classes[np.argmax(prediction)]
    confidence = float(np.max(prediction)) * 100
    
    return jsonify({
        'status': result,
        'confidence': f"{confidence:.2f}%"
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)