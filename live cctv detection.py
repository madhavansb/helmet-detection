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

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'], gpu=False)  # Set gpu=True if using Colab with GPU

# Initialize Excel file
excel_file = 'non_helmet_violations.xlsx'
if os.path.exists(excel_file):
    df = pd.read_excel(excel_file)
else:
    df = pd.DataFrame(columns=['Timestamp', 'License_Plate', 'Image_Path'])

# Function to preprocess image for better OCR
def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)
    return blurred

# Function to process frame and detect helmet/vehicle
def process_frame(frame, frame_count, output_dir='violation_images'):
    results = model(frame)[0]
    
    helmet_detected = False
    license_plate_text = None
    license_plate_box = None
    image_path = None
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, conf, cls = result
        label = model.names[int(cls)]
        
        if label == 'helmet':
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
        else:
            image_path = os.path.join(output_dir, f'violation_frame_{frame_count}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.jpg')
            cv2.imwrite(image_path, frame)
            
            saved_img = cv2.imread(image_path)
            if license_plate_box:
                plate_img = saved_img[y1:y2, x1:x2]
                preprocessed_img = preprocess_image(plate_img)
                ocr_result = reader.readtext(preprocessed_img)
                if ocr_result:
                    license_plate_text = ocr_result[0][1]
                    cv2.putText(frame, f'Plate: {license_plate_text}', (x1, y1 - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        global df
        df = pd.concat([df, pd.DataFrame([{
            'Timestamp': timestamp,
            'License_Plate': license_plate_text if license_plate_text else 'Not Detected',
            'Image_Path': image_path if image_path else 'N/A'
        }])], ignore_index=True)
        df.to_excel(excel_file, index=False)
    
    return frame, helmet_detected, license_plate_text, image_path

# Main function to process CCTV live feed
def main():
    # Replace with your CCTV's RTSP URL
    rtsp_url = input("Enter your CCTV RTSP URL (e.g., rtsp://admin:password@192.168.1.100:554/stream1): ")
    cap = cv2.VideoCapture(rtsp_url)
    
    if not cap.isOpened():
        print(f"Error: Could not connect to CCTV feed at {rtsp_url}")
        return
    
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to retrieve frame from CCTV feed")
            break
        
        processed_frame, helmet_detected, plate_text, image_path = process_frame(frame, frame_count)
        frame_count += 1
        
        status = 'Helmet Detected' if helmet_detected else 'No Helmet'
        print(f"Frame {frame_count}: {status}, License Plate: {plate_text if plate_text else 'Not Detected'}, Image: {image_path if image_path else 'N/A'}")
        
        # Exit on 'q' key (Colab doesn't support cv2.imshow, so we use a time-based exit or manual stop)
        if frame_count % 100 == 0:  # Check every 100 frames to avoid excessive input checks
            user_input = input("Enter 'q' to stop processing (or wait for next prompt): ")
            if user_input.lower() == 'q':
                break
    
    cap.release()
    
    if os.path.exists(excel_file):
        files.download(excel_file)
    for file in os.listdir('violation_images'):
        if file.endswith('.jpg'):
            files.download(os.path.join('violation_images', file))

if __name__ == '__main__':
    main()
