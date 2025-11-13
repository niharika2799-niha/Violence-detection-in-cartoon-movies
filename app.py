from flask import Flask, render_template, request, redirect, url_for
import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from werkzeug.utils import secure_filename

# ==========================
# CONFIGURATION
# ==========================
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['ALLOWED_EXTENSIONS'] = {'mp4', 'avi', 'mov', 'jpg', 'jpeg', 'png'}


MODEL_PATH = "cnn_bigru_model.h5"
CLASS_NAMES = ["Explosion", "Fighting", "Gunshot", "Normal"]
IMG_SIZE = 224
SEQ_LEN = 16

# Load model once at startup
model = load_model(MODEL_PATH)
print("✅ Model loaded successfully.")

# ==========================
# HELPER FUNCTIONS
# ==========================
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def load_video_frames(path, seq_len=SEQ_LEN, img_size=IMG_SIZE):
    cap = cv2.VideoCapture(path)
    frames = []
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idxs = np.linspace(0, total - 1, seq_len).astype(int)

    for i in range(total):
        ret, frame = cap.read()
        if not ret:
            break
        if i in idxs:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_resized = cv2.resize(frame_rgb, (img_size, img_size))
            frames.append(frame_resized)
    cap.release()

    if len(frames) < seq_len:
        while len(frames) < seq_len:
            frames.append(frames[-1])

    clip = np.expand_dims(np.array(frames[:seq_len]), axis=0)
    return clip, frames

def predict_clip(file_path):
    clip, frames = load_video_frames(file_path)
    probs = model.predict(clip)[0]
    pred_idx = np.argmax(probs)
    return {
        "pred_class": CLASS_NAMES[pred_idx],
        "confidence": float(probs[pred_idx]),
        "probs": {CLASS_NAMES[i]: float(probs[i]) for i in range(len(CLASS_NAMES))},
        "frames": frames
    }

# ==========================
# ROUTES
# ==========================
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/prediction', methods=['GET', 'POST'])
def prediction():
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                # Treat as single image
                img = cv2.imread(file_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                img_batch = np.expand_dims(img_resized, axis=0)
                probs = model.predict(img_batch)[0]
                pred_idx = np.argmax(probs)
                result = {
                    "pred_class": CLASS_NAMES[pred_idx],
                    "confidence": float(probs[pred_idx]),
                    "probs": {CLASS_NAMES[i]: float(probs[i]) for i in range(len(CLASS_NAMES))},
                    "frames": [img]
                }
            else:
                # Treat as video
                result = predict_clip(file_path)

            # Save sampled frames
            frames_dir = os.path.join('static', 'frames')
            os.makedirs(frames_dir, exist_ok=True)
            frame_files = []
            for i, f in enumerate(result['frames'][:6]):
                frame_path = os.path.join(frames_dir, f'frame_{i}.jpg')
                cv2.imwrite(frame_path, cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
                frame_files.append(frame_path)

            return render_template('result.html',
                                   file_path=file_path,
                                   pred_class=result['pred_class'],
                                   confidence=round(result['confidence'] * 100, 2),
                                   probs=result['probs'],
                                   frame_files=frame_files)
        else:
            return "Invalid file format. Please upload a video or image."

    return render_template('prediction.html')

@app.route('/analysis')
def analysis():
    # assumes you have static/analysis/*.png
    analysis_images = [f for f in os.listdir('static/analysis') if f.endswith(('.png', '.jpg'))]
    return render_template('analysis.html', analysis_images=analysis_images)

@app.route('/notebook')
def notebook():
    # convert your notebook to HTML and save as static/notebook.html
    return render_template('notebook.html')

# ==========================
# MAIN
# ==========================
if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True)
