from flask import Flask, jsonify, request , send_from_directory
from pdf2image import convert_from_path
import fitz
from PIL import Image
from pathlib import Path
import torch
from flask_cors import CORS
import re
import os
import numpy as np
from collections import defaultdict
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import easyocr
from datetime import datetime
from difflib import SequenceMatcher



#  0=aadhar no ; 1=dob ; 2=emblem logo ; 3=goi symbol ; 4=name
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# # Load the YOLOv5 model
model=None
model_logo=None
model_symbol=None
def loadAllModels():
    global model
    global model_logo
    global model_symbol
    model = torch.hub.load('ultralytics/yolov5:v7.0', 'custom', path=r"C:\Users\DELL\OneDrive\Desktop\documentverification\Backend\best (2).pt")
    print("YOLOv5 model loaded successfully!")
    # Set the model to inference mode
    model.eval()
    model_logo = load_model(r"C:\Users\DELL\Downloads\logo_classification_model.h5")
    model_symbol = load_model(r"C:\Users\DELL\Downloads\title_classification_model.h5")



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

def is_similar_name(name1, name2, threshold=0.7):
    # Get the similarity ratio between the two names
    similarity_ratio = SequenceMatcher(None, name1, name2).ratio()
    return similarity_ratio >= threshold

def clear_existing_images(folder_path):
    existing_images = os.listdir(folder_path)
    for existing_image in existing_images:
        image_path = os.path.join(folder_path, existing_image)
        os.remove(image_path)

@app.route('/Images/cropped/<path:filename>')
def serve_image(filename):
    return send_from_directory(r'C:\Users\DELL\OneDrive\Desktop\documentverification\Backend/Images/cropped', filename)


@app.route('/upload', methods=['POST'])
def upload():
    global model
    global model_logo
    global model_symbol
    try:
        name = request.form['name']
        date_of_birth = request.form['dateOfBirth']
        aadhar_number = request.form['aadharNumber']
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
        print("\n")
        print(labels_from_boxes)
        if 0.0 not in labels_from_boxes or 2.0 not in labels_from_boxes or 3.0 not in labels_from_boxes:
            is_valid_aadhaar=False
            return jsonify({'message': "Not all labels present in boxes.",'is_valid_aadhaar': False})
        # # Save the original image
        pdf_image.save(Path(r'C:\Users\DELL\OneDrive\Desktop\documentverification\Backend\Images\Cropped\photo.jpg'))

        
        # Dictionary to store the highest confidence for each class
        highest_confidence = defaultdict(float)
        # Dictionary to store the corresponding bounding box for each class
        best_boxes = defaultdict(list)


        for box in boxes:
            xmin, ymin, xmax, ymax = box[:4]
            confidence = box[4]
            class_name = box[5]

            # Check if the current box has higher confidence than the stored confidence for the class
            if confidence > highest_confidence[class_name]:
                highest_confidence[class_name] = confidence
                best_boxes[class_name] = [xmin, ymin, xmax, ymax]
        print(highest_confidence)
        # Save the cropped images using the best bounding boxes
        for class_name, best_box in best_boxes.items():
            xmin, ymin, xmax, ymax = best_box
            cropped_image = pdf_image.crop((xmin, ymin, xmax, ymax))
            cropped_image.save(Path(f'C:\\Users\\DELL\\OneDrive\\Desktop\\documentverification\\Backend\\Images\\Cropped\\cropped_{class_name}.jpg'))

        result_dict={}
        result_dict['aadhar no']=False
        result_dict['emblem logo']=False
        result_dict['goi symbol']=False
        result_dict['dob']=True
        result_dict['name']=False
        

        # Extract text from the cropped image
        number_path=r'C:\\Users\\DELL\\OneDrive\\Desktop\\documentverification\\Backend\\Images\\Cropped\\cropped_0.0.jpg'
        if(os.path.exists(number_path)):
            with open(number_path,'rb') as number_file:
                content=number_file.read()
            
            reader = easyocr.Reader(['en'])
            result = reader.readtext(number_path)
            for detection in result:
                text = detection[1]
                print(f"Extracted Text: {text}")
                if(len(text)>10 and (text.replace(" ","")).isdigit()):
                    # Validate Aadhaar number
                    is_valid_aadhaar = is_valid_aadhaar_number(text)
                    print("is valid aadhar is ",is_valid_aadhaar)
                    if(is_valid_aadhaar):
                        if ''.join(aadhar_number.split()) == ''.join(text.split()):
                            result_dict['aadhar no']=True
                    else:
                        result_dict['aadhar no']=False

        else:
            return jsonify({'error': 'Image file not found', 'is_valid_aadhaar': False}), 404


        # model_logo = load_model(r"C:\Users\moses\Downloads\logo_classification_model (1).h5")
        single_image_path = r"C:\\Users\\DELL\\OneDrive\\Desktop\\documentverification\\Backend\\Images\\Cropped\\cropped_2.0.jpg" 
        # Define class labels for your 2 different logos
        
        class_labels = ["emblem_logo", "No_emblem_logo"]
        # Set the class threshold (adjust as needed)
        threshold_value = 0.93
        # Use the function to classify the single image
        result_logo = is_valid_logo(single_image_path, model_logo, class_labels, threshold=threshold_value)
        if(result_logo==class_labels[0]):
            result_dict['emblem logo']=True
        else:
            if(highest_confidence[2.0]>0.7):
                result_dict['emblem logo']=True
            else:
                result_dict['emblem logo']=False


        # model_symbol = load_model(r"C:\Users\moses\Downloads\title_classification_model (2).h5")
        single_image_path = r"C:\\Users\\DELL\\OneDrive\\Desktop\\documentverification\\Backend\\Images\\Cropped\\cropped_3.0.jpg" 
        class_labels = ["title", "No_title"]

        threshold_value = 0.70
        result_symbol = is_valid_symbol(single_image_path, model_symbol, class_labels, threshold=threshold_value)
        if(result_symbol==class_labels[0]):
            result_dict['goi symbol']=True
        else:
            if(highest_confidence[3.0]>0.7):
                result_dict['goi symbol']=True
            else: 
                result_dict['goi symbol']=False

        number_path=r'C:\\Users\\DELL\\OneDrive\\Desktop\\documentverification\\Backend\\Images\\Cropped\\cropped_1.0.jpg'
        if(os.path.exists(number_path)):
            with open(number_path,'rb') as number_file:
                content=number_file.read()
            reader = easyocr.Reader(['en'])
            result = reader.readtext(number_path)
            for detection in result:
                text = detection[1]
                print(f"Extracted Text: {text}")
                digits_only = re.sub(r'\D', '', text)
                if(len(digits_only)==8):
                    parsed_date = datetime.strptime(date_of_birth, "%Y-%m-%d")
                    # Format the date as DDMMYYYY
                    formatted_date = parsed_date.strftime("%d%m%Y")
                    if(formatted_date==digits_only):
                        result_dict['dob']=True
                    else:
                        result_dict['dob']=False
                elif(len(digits_only)==4):
                    parsed_date_form = datetime.strptime(date_of_birth, "%Y-%m-%d")
                    year_aadhar = int(digits_only)
                    year_form = parsed_date_form.year
                    if year_aadhar==year_form:
                        result_dict['dob']=True
                    else:
                        result_dict['dob']=False


        number_path=r'C:\\Users\\DELL\\OneDrive\\Desktop\\documentverification\\Backend\\Images\\Cropped\\cropped_4.0.jpg'
        if(os.path.exists(number_path)):
            with open(number_path,'rb') as number_file:
                content=number_file.read()
            reader = easyocr.Reader(['en'])
            result = reader.readtext(number_path)
            for detection in result:
                text = detection[1]
                print(f"Extracted Text: {text}")
                if is_similar_name(name, text, threshold=0.7):
                    result_dict['name'] = True
                else:
                    result_dict['name'] = False

        print("Result :",result_dict)

        if any(value is False for value in result_dict.values()):
            response_data = {
                'message': "Document is irrelevant.",
                'is_valid_aadhaar': False,
                'verification_results': result_dict,    
            }
            return jsonify(response_data)
        else:
            response_data = {
                'message': "Document is relevant.",
                'is_valid_aadhaar': True,
                'verification_results': result_dict,
            }
            return jsonify(response_data)
    except Exception as e:
        # Print or log the exception details
        print(f"An error occurred: {str(e)}")
        return jsonify({'error': 'Internal Server Error'}), 500

if __name__ == '__main__':
    loadAllModels()
    app.run(debug=True)