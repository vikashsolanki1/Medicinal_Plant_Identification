import os
import numpy as np
from flask import Flask, render_template, request, jsonify, url_for
from keras.models import load_model
from PIL import Image
from werkzeug.utils import secure_filename
import google.generativeai as genai
from tensorflow.keras.preprocessing.image import load_img, img_to_array 
from tensorflow.keras.applications.xception import preprocess_input  # 
app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


model = load_model(r'C:\Programming\FinalMed\model\RandomVSCode.keras')


class_names = ['Aloevera', 'Amruthaballi', 'Bhrami', 'Castor', 'Coriender', 'Hibiscus', 'Kasambruga', 'Neem', 'Papaya', 'Tulsi']
# Gemini API setup
genai.configure(api_key="AIzaSyCHWQjXQhPk2SRtypTfTiR3-arysC2aowY")  
chat_model = genai.GenerativeModel("models/gemini-1.5-flash-001")
chat = chat_model.start_chat()


def preprocess_image(file_path):
    
    image = Image.open(file_path).convert("RGB")
    image = image.resize((299, 299))  

    
    image_array = img_to_array(image)
    image_array = np.expand_dims(image_array, axis=0)

    #  Apply Xception-specific preprocessing
    image_array = preprocess_input(image_array)
    return image_array



@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file uploaded", 400

    file = request.files['file']
    if file.filename == '':
        return "Empty filename", 400

    if file:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        
        img_array = preprocess_image(filepath)
        prediction = model.predict(img_array)

        
        print("Raw prediction vector:", prediction)
        predicted_index = np.argmax(prediction)
        print("Predicted index:", predicted_index)

        predicted_class = class_names[predicted_index]
        print("Predicted class label:", predicted_class)

        return render_template('result.html', prediction=predicted_class,
                                image_path=file.filename)

    return "File processing error", 500


@app.route('/chat', methods=['POST'])
def chat_with_bot():
    data = request.get_json()
    user_message = data.get("message")

    if not user_message:
        return jsonify({"reply": "Please enter a question."})

    try:
        response = chat.send_message(user_message)
        return jsonify({"reply": response.text})
    except Exception as e:
        return jsonify({"reply": f"Error: {str(e)}"})

if __name__ == '__main__':
    app.run(debug=True)
