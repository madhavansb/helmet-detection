!pip install ultralytics
!wget https://huggingface.co/aneesarom/Helmet-Violation-Detection/resolve/main/best.pt -O best.pt
from ultralytics import YOLO

# Load the model
model = YOLO('best.pt')
from google.colab import files
uploaded = files.upload()

import os

# Get the uploaded image filename
image_path = next(iter(uploaded))
# Perform inference
results = model(image_path)

# Display the results
results[0].show()

# Upload a video
uploaded_video = files.upload()
video_path = next(iter(uploaded_video))

# Perform inference on the video
results = model(video_path, save=True)

# The annotated video will be saved in the 'runs/detect/predict' directory
