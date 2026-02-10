from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import io

app = Flask(__name__)

print("Loading model...")
model = tf.keras.models.load_model('blood_cell_model.h5')
print("Model loaded!")

CLASS_NAMES = ['EOSINOPHIL', 'LYMPHOCYTE', 'MONOCYTE', 'NEUTROPHIL']

def prepare_image(image, target_size=(128, 128)):
    """
    Preprocesses the image to match the training format:
    1. Resize to 128x128
    2. Convert to RGB
    3. Normalize pixel values (0-1)
    4. Expand dimensions (1, 128, 128, 3)
    """
    if image.mode != "RGB":
        image = image.convert("RGB")
    
    image = image.resize(target_size)
    image = tf.keras.preprocessing.image.img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = image / 255.0
    return image

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    try:
        image = Image.open(io.BytesIO(file.read()))
        processed_image = prepare_image(image)

        prediction = model.predict(processed_image)
        
        class_idx = np.argmax(prediction[0])
        confidence = float(np.max(prediction[0]))
        class_name = CLASS_NAMES[class_idx]
        
        return jsonify({
            'class': class_name,
            'confidence': f"{confidence:.2%}",
            'all_probabilities': {k: float(v) for k, v in zip(CLASS_NAMES, prediction[0])}
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)