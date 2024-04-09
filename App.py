from flask import Flask, jsonify, request
from pdf2image import convert_from_path
import fitz
from PIL import Image
from pathlib import Path
import torch
from flask_cors import CORS
import pytesseract
import re
import os
import numpy as np

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array


# class 0 :Aadhar number ; class 1 :Emblem logo ; class 2 :Goi symbol ; class 3 :Text
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes


# Load the YOLOv5 model
model = torch.hub.load('ultralytics/yolov5:v7.0', 'custom', path=r"C:\Users\moses\OneDrive\Desktop\DocumentVerification\best.pt")
print("YOLOv5 model loaded successfully!")
# Set the model to inference mode
model.eval()


# Function to validate Aadhaar number.
def is_valid_aadhaar_number(text):
    # Regex to check valid Aadhaar number.
    regex = ("^[2-9]{1}[0-9]{3}\\" +
             "s[0-9]{4}\\s[0-9]{4}$")

    # Compile the ReGex
    p = re.compile(regex)

    # If the string is empty return false
    if text is None:
        return False

    # Return if the string matched the ReGex
    return bool(re.search(p, text))

def is_valid_logo(image_path, model, class_labels, threshold=0.5):
    try:
        # Load and preprocess the image
        img = Image.open(image_path)
        img = img.resize((224, 224))  # Resize to match the input size of the model
        img = img_to_array(img)
        img = np.expand_dims(img, axis=0)  # Add batch dimension
        img = img / 255.0  # Normalize pixel values to [0, 1]

        # Use the model to make a prediction
        predictions = model.predict(img)
        print("Raw Predictions:", predictions)

        # Get the predicted class label based on the threshold
        if predictions[0, 0] >= threshold:
            predicted_class = class_labels[0]  
        else:
            predicted_class = class_labels[1]  

        return predicted_class
    except Exception as e:
        return str(e)

def is_valid_symbol(image_path, model, class_labels, threshold=0.5):
    try:
        # Load and preprocess the image
        img = Image.open(image_path)
        img = img.resize((224, 224))  # Resize to match the input size of the model
        img = img_to_array(img)
        img = np.expand_dims(img, axis=0)  # Add batch dimension
        img = img / 255.0  # Normalize pixel values to [0, 1]

        # Use the model to make a prediction
        predictions = model.predict(img)
        print("Raw Predictions:", predictions)

        # Get the predicted class label based on the threshold
        if predictions[0, 0] >= threshold:
            predicted_class = class_labels[0]  
        else:
            predicted_class = class_labels[1]  

        return predicted_class
    except Exception as e:
        return str(e)

@app.route('/upload', methods=['POST'])
def upload():
    try:
        if 'pdf' not in request.files:
            return jsonify({'error': 'No file part'}), 400

        pdf_file = request.files['pdf']

        if pdf_file.filename == '':
            return jsonify({'error': 'No selected file'}), 400

        # Convert PDF to image using PyMuPDF (fitz)
        print('PDF received by backend')
        # Save the PDF content to a temporary file
        temp_pdf_path = 'temp.pdf'
        pdf_file.save(temp_pdf_path)

        pdf_document = fitz.open(temp_pdf_path)
        first_page = pdf_document[0]
        pix = first_page.get_pixmap()

        # Save the image as temp.jpg
        temp_image_path = 'temp.jpg'
        with open(temp_image_path, 'wb') as img_file:
            img_file.write(pix.tobytes())

        # Open the image using PIL
        pdf_image = Image.open(temp_image_path)
        # Perform inference on the image
        results = model(temp_image_path)
        results.show()
        # Extract bounding box coordinates (xmin, ymin, xmax, ymax) from results.xyxy tensor
        boxes = results.xyxy[0].cpu().numpy()
        
        labels_from_boxes = set(boxes[:, -1])
        if(len(labels_from_boxes)<4):
            is_valid_aadhaar=False
            return jsonify({'message': "Not all labels present in boxes.",'is_valid_aadhaar': False})
        # # Save the original image
        # pdf_image.save(Path('C:/Users/Sanjay Yadav/OneDrive/Desktop/BackendModels/images/temp.jpg'))

        # Iterate over each bounding box and save the cropped region
        for i, box in enumerate(boxes):
            xmin, ymin, xmax, ymax = box[:4]
            cropped_image = pdf_image.crop((xmin, ymin, xmax, ymax))
            cropped_image.save(Path(f'C:\\Users\\moses\\OneDrive\\Desktop\\DocumentVerification\\Backend\\Images\\Cropped\\cropped_{box[5]}.jpg'))
        result_list=[]
        # Display the results
        # results.show()
        
        # Extract text from the cropped image
        number_path=r'C:\Users\moses\OneDrive\Desktop\DocumentVerification\Backend\Images\Cropped\cropped_0.0.jpg'
        if(os.path.exists(number_path)):
            with open(number_path,'rb') as number_file:
                content=number_file.read()
            number=Image.open(number_path)
            text = pytesseract.image_to_string(number)
            print("Hi2")
            print("Number is :",text)
            # Validate Aadhaar number
            is_valid_aadhaar = is_valid_aadhaar_number(text)
            print("is valid aadhar is ",is_valid_aadhaar)
            if(is_valid_aadhaar):
                result_list.append(True)
            else:
                result_list.append(False)

        else:
            return jsonify({'error': 'Image file not found', 'is_valid_aadhaar': False}), 404


        model_logo = load_model(r"C:\Users\moses\Downloads\logo_classification_model (1).h5")
        single_image_path = r"C:\Users\moses\OneDrive\Desktop\DocumentVerification\Backend\Images\Cropped\cropped_1.0.jpg" 
        # Define class labels for your 2 different logos
        
        class_labels = ["emblem_logo", "No_emblem_logo"]
        # Set the class threshold (adjust as needed)
        threshold_value = 0.93
        # Use the function to classify the single image
        result_logo = is_valid_logo(single_image_path, model_logo, class_labels, threshold=threshold_value)
        if(result_logo==class_labels[0]):
            result_list.append(True)
        else:
            result_list.append(False)


        model_symbol = load_model(r"C:\Users\moses\Downloads\title_classification_model (2).h5")
        single_image_path = r"C:\Users\moses\OneDrive\Desktop\DocumentVerification\Backend\Images\Cropped\cropped_2.0.jpg" 
        class_labels = ["title", "No_title"]

        threshold_value = 0.70
        result_symbol = is_valid_symbol(single_image_path, model_symbol, class_labels, threshold=threshold_value)
        if(result_symbol==class_labels[0]):
            result_list.append(True)
        else:
            result_list.append(False)
        print("Hi")
        print("Result list is : ",result_list)
        if(False not in result_list):
            return jsonify({'message': "Document is verified.",'is_valid_aadhaar': True})
        return jsonify({'message': "Document is irrelevant.",'is_valid_aadhaar': False})
    except Exception as e:
        # Print or log the exception details
        print(f"An error occurred: {str(e)}")
        return jsonify({'error': 'Internal Server Error'}), 500

if __name__ == '__main__':
    app.run(debug=True)
