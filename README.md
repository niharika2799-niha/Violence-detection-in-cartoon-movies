Cartoon Violence Detection – Deep Learning & Flask Application
Overview

Cartoon Violence Detection is a Python 3 + Flask web application that detects violent scenes in cartoon videos or images using a trained CNN-BiGRU deep learning model.
The system analyzes frames and classifies content into:

Explosion

Fighting

Gunshot

Normal

The project is modular, lightweight, and designed for academic research, machine learning deployment, and multimedia content analysis.

Features
Violence Classification:
Detects violent events in cartoon scenes using a trained deep learning model.
Video & Image Support:
Uploads can be images (.jpg/.png) or videos (.mp4/.mov/.avi).
Frame Sampling & Processing:
Automatically extracts 16 frames from videos for prediction.

Prediction Dashboard:
Displays:
Predicted class
Confidence score
Full probability distribution
Sample frames from the video

📓 Notebook Integration:
Includes model.ipynb for training pipeline and model explanation.

Table of Contents
Installation
Usage
Project Structure
Contributing

Installation
1. Clone the Repository
git clone https://github.com/yourusername/cartoon_violence_detection.git
cd cartoon_violence_detection

2. Install Dependencies
pip install -r requirements.txt

3. Prepare Model File
Ensure the trained model file exists in the root directory:
cnn_bigru_model.h5



Default settings inside app.py:
IMG_SIZE = 224
SEQ_LEN = 16
CLASS_NAMES = ["Explosion", "Fighting", "Gunshot", "Normal"]
UPLOAD_FOLDER = "static/uploads"

Usage
Run the Application
python app.py

Access in Browser
http://127.0.0.1:5000/

Available Interface Pages
Page	Description
/	Home screen
/prediction	Upload image/video for violence detection
/analysis	View analysis plots or model images
/notebook	View exported Jupyter notebook
Running Predictions
Upload a Video

The system:
samples 16 frames
preprocesses each frame
runs CNN-BiGRU model inference
displays class & confidence
Upload an Image

The model:
resizes image to 224×224
predicts using same classification head

Project Structure
cartoon_violence_detection/
├── app.py                    # Flask application
├── cnn_bigru_model.h5        # Trained deep learning model
├── requirements.txt          # Python dependencies
├── model.ipynb               # Model training notebook
├── static/
│   ├── uploads/              # Uploaded files
│   ├── frames/               # Sampled video frames
│   ├── analysis/             # Model plots and analysis visuals
├── templates/
│   ├── home.html
│   ├── prediction.html
│   ├── result.html
│   ├── analysis.html
│   ├── notebook.html
└── README.md                 # Documentation

Contributing

Contributions are welcome!
You may:
Improve the UI
Add model explainability (Grad-CAM, visualization)
Enhance training pipeline
Expand dataset

To contribute:

Fork the repo
Create a new branch
Commit your changes
Open a pull request
