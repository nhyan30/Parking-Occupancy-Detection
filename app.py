from flask import Flask, render_template, request, jsonify
import tensorflow as tf
from PIL import Image
import numpy as np
import io
from joblib import load
import cv2
from tensorflow.keras.applications.resnet50 import preprocess_input

app = Flask(__name__)

# Load all models
first_model = tf.keras.models.load_model('first_model.h5')
optimized_model = tf.keras.models.load_model('OPti.h5')
random_forest_model = load('random_forest_model.pkl')
decision_tree_model = load('dec.joblib')  # Load Decision Tree model

def preprocess_image_cnn(image):
    # Convert PIL Image to cv2 format
    img_array = np.array(image)
    img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    
    # Resize and preprocess for CNN
    img_array = cv2.resize(img_array, (37, 49))
    img_array = preprocess_input(img_array)
    return np.expand_dims(img_array, axis=0)

def preprocess_image_tree(image):
    try:
        # Convert PIL Image to cv2 format
        img_array = np.array(image)
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        # Resize image
        img_array = cv2.resize(img_array, (64, 64))
        
        # Extract color histogram features
        hsv = cv2.cvtColor(img_array, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 8, 8], 
                           [0, 180, 0, 256, 0, 256])
        cv2.normalize(hist, hist)
        features = hist.flatten().reshape(1, -1)
        
        return features
    except Exception as e:
        print(f"Error in preprocessing: {str(e)}")
        raise

def preprocess_image_rf(image):
    try:
        # Convert PIL Image to numpy array
        img_array = np.array(image)
        
        # Resize to match model's expected input
        img_array = cv2.resize(img_array, (37, 49))
        
        # Preprocess input
        img_array = preprocess_input(img_array)
        
        # Flatten the image for Random Forest
        return img_array.flatten().reshape(1, -1)
    except Exception as e:
        print(f"Error in preprocessing: {str(e)}")
        raise

@app.route('/')
def home():
    return render_template('page 1.html')

@app.route('/page1')
def page1():
    return render_template('page 1.html')

@app.route('/page2')
def page2():
    return render_template('page 2.html')

@app.route('/div')
def div():
    return render_template('div.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the image and model type from the POST request
        file = request.files['image']
        model_type = request.form.get('model_type', 'optimized')  # Default to optimized if not specified
        
        image = Image.open(io.BytesIO(file.read()))
        
        # Select the appropriate preprocessing and model based on model_type
        if model_type == 'decision_tree':
            processed_image = preprocess_image_tree(image)
            model = decision_tree_model
            prediction = model.predict_proba(processed_image)
            result = "Occupied" if prediction[0][1] > 0.5 else "Empty"
            confidence = float(prediction[0][1])
        elif model_type == 'random_forest':
            processed_image = preprocess_image_rf(image)
            model = random_forest_model
            prediction = model.predict_proba(processed_image)
            result = "Occupied" if prediction[0][1] > 0.5 else "Empty"
            confidence = float(prediction[0][1])
        else:  # CNN or optimized model
            processed_image = preprocess_image_cnn(image)
            if model_type == 'cnn':
                model = first_model
            else:  # optimized model
                model = optimized_model
            prediction = model.predict(processed_image)
            result = "Occupied" if prediction[0][0] > 0.5 else "Empty"
            confidence = float(prediction[0][0])
        
        return jsonify({
            'success': True,
            'prediction': result,
            'confidence': f"{confidence:.2%}",
            'model_used': model_type
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

if __name__ == '__main__':
    app.run(debug=True) 