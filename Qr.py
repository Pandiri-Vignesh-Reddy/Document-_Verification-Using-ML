import cv2
from pyzbar.pyzbar import decode

def extract_qr_code_data(image_path):
    # Read the image
    image = cv2.imread(image_path)

    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Use PyZbar to decode the QR code
    decoded_objects = decode(gray_image)

    # List to store QR code data
    qr_code_data_list = []

    # Process each decoded QR code
    for decoded_object in decoded_objects:
        data = decoded_object.data.decode('utf-8')
        qr_code_data_list.append(data)

    return qr_code_data_list

if __name__ == "__main__":
    # Specify the path to the QR image
    qr_image_path = r'C:\Users\DELL\Downloads\WhatsApp Image 2024-01-03 at 12.42.36 PM.jpeg'

    # Call the function to extract QR code data
    extracted_data = extract_qr_code_data(qr_image_path)

    # Print the extracted QR code data
    print("Extracted QR Code Data:")
    for data in extracted_data:
        print(data)

