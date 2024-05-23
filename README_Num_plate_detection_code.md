from google.colab import drive
drive.mount('/content/drive')
# Clone YOLOv5 repository
!git clone https://github.com/ultralytics/yolov5
%cd yolov5

# Install dependencies
!pip install -r requirements.txt
# Mount Google Drive to save dataset and results
from google.colab import drive
drive.mount('/content/drive')

# Replace with your Roboflow dataset URL
roboflow_url = "https://app.roboflow.com/ds/NkYQ0U4gth?key=NniAlI45oe"

# Download and unzip the dataset
!curl -L "{roboflow_url}" > roboflow.zip
!unzip roboflow.zip -d /content/roboflow

import os

# Check if directories exist
train_images_dir = '/content/roboflow/train/images'
val_images_dir = '/content/roboflow/valid/images'

print("Train images directory exists:", os.path.exists(train_images_dir))
print("Validation images directory exists:", os.path.exists(val_images_dir))

# List a few files in the directories
print("Train images:", os.listdir(train_images_dir)[:5])
print("Validation images:", os.listdir(val_images_dir)[:5])

data_yaml = """
train: /content/roboflow/train/images
val: /content/roboflow/valid/images

nc: 1
names: ['number_plate']
"""

# Write the data.yaml file
with open('/content/roboflow/data.yaml', 'w') as file:
    file.write(data_yaml)

# Train the model using a larger YOLOv5 variant (yolov5m.pt or yolov5l.pt)
!python train.py --img 640 --batch 16 --epochs 100 --data /content/roboflow/data.yaml --weights yolov5m.pt

# Evaluate the model
!python val.py --weights runs/train/exp/weights/best.pt --data /content/roboflow/data.yaml --img 640 --conf 0.25

import os

# Check the contents of the weights directory
weights_dir = 'runs/train/exp/weights'
print("Contents of 'runs/train/exp/weights':", os.listdir(weights_dir))

import torch
import cv2
import matplotlib.pyplot as plt

# Load the trained model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='runs/train/exp/weights/best.pt', force_reload=True)

def detect_and_display(image_path):
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Unable to load the image from path:", image_path)
        return

    # Print image dimensions
    print("Image dimensions (HxW):", image.shape[:2])

    # Perform inference
    results = model(image_path)

    # Inspect model output
    print("Model Output:", results.xyxy)

    # Extract confidence scores
    confidence_scores = results.xyxy[0][:, 4].cpu().numpy()
    print("Confidence Scores:", confidence_scores)

    # Draw bounding boxes on the image
    for detection, conf in zip(results.xyxy[0], confidence_scores):
        # Convert tensor to CPU first
        x1, y1, x2, y2, _, cls = detection.cpu().int().numpy()

        if conf > 0.5:  # Apply confidence threshold
            # Ensure bounding box coordinates are within image dimensions
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(image.shape[1], x2), min(image.shape[0], y2)

            # Draw bounding box
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, f'{model.names[int(cls)]} {conf:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
            print("Bounding Box Drawn:", (x1, y1), (x2, y2))
        else:
            print("Confidence below threshold, not drawing bounding box")

    # Display the image with detections
    plt.imshow(image[:, :, ::-1])  # Convert BGR to RGB
    plt.axis('off')  # Hide axes for better visualization
    plt.show()

# Test the function with a new image
image_path = "/content/DSC_1074.JPG"  # Replace with your image path
detect_and_display(image_path)

import torch
import cv2
from google.colab.patches import cv2_imshow

# Load the trained model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='runs/train/exp/weights/best.pt', force_reload=True)

# Function to perform inference and visualize results
def detect_and_display(image_path):
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Unable to load the image from path:", image_path)
        return

    # Perform inference
    results = model(image_path)

    # Extract bounding boxes, confidence scores, and class predictions
    boxes = results.xyxy[0].cpu().numpy()[:, :4]  # Extract bounding boxes
    confidences = results.xyxy[0].cpu().numpy()[:, 4]  # Extract confidence scores
    classes = results.xyxy[0].cpu().numpy()[:, 5].astype(int)  # Extract class predictions

    # Draw bounding boxes on the image
    for bbox, confidence, cls in zip(boxes, confidences, classes):
        if confidence > 0.5:  # Apply confidence threshold
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, f'{model.names[cls]} {confidence:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

    # Display the image with detections using cv2_imshow()
    cv2_imshow(image)

# Test the function with a new image
image_path = "/content/DSC_1074.JPG"  # Replace with your image path
detect_and_display(image_path)

!sudo apt-get install tesseract-ocr
!pip install pytesseract

import torch
import cv2
from google.colab.patches import cv2_imshow
import pytesseract
from PIL import Image
import numpy as np

# Ensure Tesseract is installed
!apt-get install -y tesseract-ocr
!pip install pytesseract

# Load the trained model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='runs/train/exp/weights/best.pt', force_reload=True)

# Function to preprocess the image for better OCR accuracy
def preprocess_image(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Apply bilateral filter to remove noise
    denoised = cv2.bilateralFilter(gray, 11, 17, 17)
    # Apply adaptive thresholding
    thresh = cv2.adaptiveThreshold(denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    # Apply morphological operations to emphasize the text
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    return morph

# Function to perform inference, visualize results, and extract text
def detect_and_extract_number_plate(image_path):
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Unable to load the image from path:", image_path)
        return

    # Perform inference
    results = model(image_path)

    # Extract bounding boxes, confidence scores, and class predictions
    boxes = results.xyxy[0].cpu().numpy()[:, :4]  # Extract bounding boxes
    confidences = results.xyxy[0].cpu().numpy()[:, 4]  # Extract confidence scores
    classes = results.xyxy[0].cpu().numpy()[:, 5].astype(int)  # Extract class predictions

    # Process each detected number plate
    for bbox, confidence, cls in zip(boxes, confidences, classes):
        if confidence > 0.5:  # Apply confidence threshold
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, f'{model.names[cls]} {confidence:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

            # Crop the detected number plate from the image
            plate_image = image[y1:y2, x1:x2]

            # Preprocess the cropped image for better OCR accuracy
            preprocessed_plate = preprocess_image(plate_image)

            # Convert the preprocessed image to PIL format for OCR
            plate_pil = Image.fromarray(preprocessed_plate)

            # Use Tesseract to extract text
            plate_text = pytesseract.image_to_string(plate_pil, config='--psm 8')  # PSM 8 is for single word/line
            print("Detected Number Plate:", plate_text.strip())

    # Display the image with detections using cv2_imshow()
    cv2_imshow(image)

# Test the function with a new image
image_path = "/content/test_plate.jpg"  # Replace with your image path
detect_and_extract_number_plate(image_path)

import torch
import cv2
import pytesseract
from PIL import Image
import numpy as np

# Ensure Tesseract is installed
!apt-get install -y tesseract-ocr
!pip install pytesseract

# Load the trained model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='runs/train/exp/weights/best.pt', force_reload=True)

# Function to preprocess the image for better OCR accuracy
def preprocess_image(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Apply bilateral filter to remove noise
    denoised = cv2.bilateralFilter(gray, 11, 17, 17)
    # Apply adaptive thresholding
    thresh = cv2.adaptiveThreshold(denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    # Apply morphological operations to emphasize the text
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    return morph

# Function to perform inference, visualize results, and extract text
def detect_and_extract_number_plate(frame):
    # Perform inference
    results = model(frame)

    # Extract bounding boxes, confidence scores, and class predictions
    boxes = results.xyxy[0].cpu().numpy()[:, :4]  # Extract bounding boxes
    confidences = results.xyxy[0].cpu().numpy()[:, 4]  # Extract confidence scores
    classes = results.xyxy[0].cpu().numpy()[:, 5].astype(int)  # Extract class predictions

    # Process each detected number plate
    for bbox, confidence, cls in zip(boxes, confidences, classes):
        if confidence > 0.5:  # Apply confidence threshold
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f'{model.names[cls]} {confidence:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

            # Crop the detected number plate from the frame
            plate_image = frame[y1:y2, x1:x2]

            # Preprocess the cropped image for better OCR accuracy
            preprocessed_plate = preprocess_image(plate_image)

            # Convert the preprocessed image to PIL format for OCR
            plate_pil = Image.fromarray(preprocessed_plate)

            # Use Tesseract to extract text
            plate_text = pytesseract.image_to_string(plate_pil, config='--psm 8')  # PSM 8 is for single word/line
            print("Detected Number Plate:", plate_text.strip())

    return frame

# Replace 'your_video_file.mp4' with the path to your uploaded video file
video_path = '/content/WhatsApp Video 2024-05-20 at 00.59.05_0623a890.mp4'
cap = cv2.VideoCapture(video_path)

# Get the video writer initialized to save the output video
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output_video.mp4', fourcc, cap.get(cv2.CAP_PROP_FPS),
                      (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Detect and extract number plate
    processed_frame = detect_and_extract_number_plate(frame)

    # Write the processed frame to the output video
    out.write(processed_frame)

# Release the capture and writer
cap.release()
out.release()

# Save the final video to your local file system
from google.colab import files
files.download('output_video.mp4')


import torch
from PIL import Image
import pytesseract
import cv2
import numpy as np
from google.colab.patches import cv2_imshow

# Load the YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='runs/train/exp/weights/best.pt', force_reload=True)

# Function to detect number plate and extract numbers
def detect_and_extract_numbers(image_path):
    # Load image
    img = Image.open(image_path)

    # Perform inference
    results = model(img)

    # Extract bounding boxes and confidence scores
    boxes = results.xyxy[0][:, :4].cpu().numpy()  # (x_min, y_min, x_max, y_max) format
    confidences = results.xyxy[0][:, 4].cpu().numpy()

    # Filter boxes with confidence threshold
    threshold = 0.5
    detected_boxes = boxes[confidences > threshold]

    # Iterate over detected boxes
    for box in detected_boxes:
        x_min, y_min, x_max, y_max = box.astype(int)

        # Crop the detected number plate
        plate_img = np.array(img)[y_min:y_max, x_min:x_max]

        # Convert to grayscale
        gray_plate = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)

        # Apply thresholding
        _, thresh_plate = cv2.threshold(gray_plate, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        # Apply morphological operations to enhance the plate
        kernel = np.ones((3, 3), np.uint8)
        plate_enhanced = cv2.morphologyEx(thresh_plate, cv2.MORPH_CLOSE, kernel)

        # Apply OCR to extract numbers
        plate_text = pytesseract.image_to_string(plate_enhanced, config='--psm 6')

        # Display the cropped plate and extracted numbers
        cv2_imshow(plate_enhanced)
        print("Detected Number Plate:", plate_text.strip())

    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Test the function with a new image
image_path = "/content/test_plate.jpg"  # Replace with your image path
detect_and_extract_numbers(image_path)

!pip install easyocr


import cv2
import easyocr

# Load the image
image_path = '/content/post_processed_image3.png'  # Replace with your actual image path
image = cv2.imread(image_path)

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Increase image resolution
scale_percent = 300  # Increase the size by 300%
width = int(gray.shape[1] * scale_percent / 100)
height = int(gray.shape[0] * scale_percent / 100)
dim = (width, height)
resized = cv2.resize(gray, dim, interpolation=cv2.INTER_LINEAR)

# Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
clahe_applied = clahe.apply(resized)

# Apply bilateral filter to reduce noise while keeping edges sharp
bilateral_filtered = cv2.bilateralFilter(clahe_applied, 9, 75, 75)

# Apply adaptive thresholding
adaptive_thresh = cv2.adaptiveThreshold(
    bilateral_filtered, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

# Erode and dilate to remove noise and better isolate characters
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
eroded = cv2.erode(adaptive_thresh, kernel, iterations=1)
dilated = cv2.dilate(eroded, kernel, iterations=1)

# Save the preprocessed image for debugging purposes
cv2.imwrite('preprocessed_image.png', dilated)

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'])

# Use EasyOCR to extract text
result = reader.readtext(dilated)

# Extract and print the recognized text
plate_text = ''.join([res[1] for res in result])
print("Extracted Number Plate Text:", plate_text)
