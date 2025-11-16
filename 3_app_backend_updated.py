import tensorflow as tf
from tensorflow import keras
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import base64
import io
from PIL import Image
import os
import cv2

# --- 1. Inisialisasi Flask ---
app = Flask(__name__)
CORS(app)

@app.route('/')
def halaman_utama():
    return "Server klasifikasi aksara aktif. Kirim POST ke /predict untuk prediksi."

# --- 2. Variabel Global ---
prediction_model = None
class_names = None
IMG_HEIGHT = 64
IMG_WIDTH = 256

# --- 3. Preprocessing sesuai training ---
def preprocess_image(img_pil):
    try:
        # Convert ke grayscale
        img = np.array(img_pil.convert('L'))

        # Resize
        img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))

        # CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        img = clahe.apply(img)

        # Normalisasi dan invert
        img = img.astype(np.float32) / 255.0
        img = 1.0 - img

        # Expand dims: (H, W, 1) → (1, H, W, 1)
        img = np.expand_dims(img, axis=-1)
        img = np.expand_dims(img, axis=0)
        return img
    except Exception as e:
        print(f"Error preprocessing: {e}")
        return None

# --- 4. Load model dan kelas ---
def load_app_model():
    global prediction_model, class_names
    model_path = "model_cnn_terbaik1.keras"
    classes_path = "classes.txt"

    if not os.path.exists(model_path):
        print(f"Error: File model '{model_path}' tidak ditemukan.")
        return
    if not os.path.exists(classes_path):
        print(f"Error: File '{classes_path}' tidak ditemukan.")
        return

    # Load kelas
    with open(classes_path, "r", encoding="utf-8") as f:
        class_names = [line.strip() for line in f if line.strip()]

    # Load model
    prediction_model = keras.models.load_model(model_path)
    print("Model berhasil dimuat.")

# --- 5. Endpoint prediksi ---
@app.route('/predict', methods=['POST'])
def predict():
    if prediction_model is None or class_names is None:
        return jsonify({'error': 'Model belum dimuat'}), 500

    data = request.json
    if 'image' not in data:
        return jsonify({'error': 'Key \"image\" tidak ditemukan'}), 400

    try:
        # Decode base64
        img_data_b64 = data['image'].split(',')[1]
        img_bytes = base64.b64decode(img_data_b64)
        img_pil = Image.open(io.BytesIO(img_bytes))

        # Preprocess
        processed_img = preprocess_image(img_pil)
        if processed_img is None:
            return jsonify({'error': 'Gagal preprocessing gambar'}), 400

        # Prediksi
        raw_pred = prediction_model.predict(processed_img)
        pred_idx = int(np.argmax(raw_pred))
        pred_class = class_names[pred_idx]
        confidence = float(np.max(raw_pred))

        return jsonify({
            'prediction': pred_class,
            'confidence': round(confidence, 4)
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 400

# --- 6. Jalankan server ---
if __name__ == '__main__':
    load_app_model()
    app.run(host='0.0.0.0', port=5000, debug=False)