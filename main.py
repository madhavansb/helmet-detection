!pip install ultralytics opencv-python easyocr pandas
#install necessary dependencies
import cv2
import numpy as np
from ultralytics import YOLO
import easyocr
import pandas as pd
from datetime import datetime
import os
from google.colab import files

# Initialize YOLO model
model = YOLO('yolov8n.pt')


reader = easyocr.Reader(['en'], gpu=False)  # Set gpu=True if using Colab with GPU

# Initialize Excel file
excel_file = 'non_helmet_violations.xlsx'
if os.path.exists(excel_file):
    df = pd.read_excel(excel_file)
else:
    df = pd.DataFrame(columns=['Timestamp', 'License_Plate'])

# Function to upload video file in Colab
def select_video_file():
    print("Please upload a video file from your desktop.")
    uploaded = files.upload()
    if not uploaded:
        print("No file uploaded. Exiting...")
        return None
    # Get the first uploaded file
    file_name = list(uploaded.keys())[0]
    # Save the uploaded file to the Colab environment
    with open(file_name, 'wb') as f:
        f.write(uploaded[file_name])
    return file_name

# Function to process frame and detect helmet/vehicle
def process_frame(frame):
    results = model(frame)[0]
    
    helmet_detected = False
    license_plate_text = None
    license_plate_box = None
    
    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, conf, cls = result
        label = model.names[int(cls)]
        
        if label == 'helmet':  # Adjust based on your model's class names
            helmet_detected = True
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(frame, f'Helmet: {conf:.2f}', (int(x1), int(y1) - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        if label == 'vehicle' or label == 'license_plate':
            license_plate_box = (int(x1), int(y1), int(x2), int(y2))
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
    
    if not helmet_detected and license_plate_box:
        x1, y1, x2, y2 = license_plate_box
        plate_img = frame[y1:y2, x1:x2]
        plate_img_gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
        ocr_result = reader.readtext(plate_img_gray)
        if ocr_result:
            license_plate_text = ocr_result[0][1]
            cv2.putText(frame, f'Plate: {license_plate_text}', (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            global df
            df = pd.concat([df, pd.DataFrame([{'Timestamp': timestamp, 'License_Plate': license_plate_text}])], ignore_index=True)
            df.to_excel(excel_file, index=False)
    
    return frame, helmet_detected, license_plate_text

# Main function to process video
def main():
    # Upload video file
    video_path = select_video_file()
    if not video_path:
        return
    
    # Open video feed
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process the frame
        processed_frame, helmet_detected, plate_text = process_frame(frame)
        
        # Note: cv2.imshow is not supported in Colab; skip display
        # Optionally save frames or process silently
        status = 'Helmet Detected' if helmet_detected else 'No Helmet'
        print(f"Frame processed: {status}, License Plate: {plate_text if plate_text else 'None'}")
    
    # Release resources
    cap.release()
    
    # Download the Excel file
    if os.path.exists(excel_file):
        files.download(excel_file)

if __name__ == '__main__':
    main()
