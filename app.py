from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
from PIL import Image
from smtp import gmail_transfer
from flask_mail import Mail, Message
from dotenv import load_dotenv
import os

load_dotenv()

gmail = os.getenv('EMAIL')
passwo = os.getenv('PASSWORD')

app = Flask(__name__)
CORS(app)

app.config['MAIL_SERVER']= 'smtp.gmail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USERNAME'] = gmail
app.config['MAIL_PASSWORD'] = passwo
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USE_SSL'] = False
mail = Mail(app)

model = load_model("model.h5") 

def preprocess_image(image, target_size):
    image = Image.open(image)
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image = image.resize(target_size)
    image = img_to_array(image)  
    image = image / 255.0      
    image = np.expand_dims(image, axis=0) 
    return image

@app.route('/predict', methods=['POST'])
def predict():
    image = request.files['file']
    name = request.form.get('name')
    email = request.form.get('email')
    processed_image = preprocess_image(image, target_size=(224, 224))
    prediction = model.predict(processed_image)
    label = np.argmax(prediction, axis=1)[0]
    index = ['Mild Demented', 'Moderate Demented', 'Non Demented', 'Very Mild Demented']
    # gmail_transfer(request.form.get('email'),request.form.get('name'),index[label])
    msg = Message(subject=f"Hi {name}, Here is your report from Alzheimer's Disease Website", sender='tejaswinimaddirala@gmail.com', recipients=[email])
    msg.body = f"Dear {name},\n\nThe predicted stage of your Alzheimer's disease is {index[label]}\n"
    mail.send(msg)
    return jsonify({'prediction': index[label]})

if __name__ == '__main__':
    app.run(debug=True)