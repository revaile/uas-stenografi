from flask import Flask, render_template, request, send_file
import numpy as np
import cv2
import os
import random
from werkzeug.utils import secure_filename

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads/'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Generate pseudo noise
def generate_noise(shape, key):
    np.random.seed(key)
    return np.random.choice([1, -1], size=shape)

# Embed message using Spread Spectrum
def spread_spectrum_embed(image, message, key):
    message_bits = ''.join(format(ord(char), '08b') for char in message)
    img = image.copy().astype(np.float32)
    flat = img.flatten()
    for i, bit in enumerate(message_bits):
        noise = generate_noise(flat.shape, key + i)
        flat += noise * (1 if bit == '1' else -1) * 1.5  # scale factor
    img = np.clip(flat.reshape(img.shape), 0, 255).astype(np.uint8)
    return img

# Extract message
def spread_spectrum_extract(image, length, key):
    img = image.copy().astype(np.float32).flatten()
    bits = ''
    for i in range(length * 8):
        noise = generate_noise(img.shape, key + i)
        dot = np.dot(img, noise)
        bits += '1' if dot > 0 else '0'
    chars = [chr(int(bits[i:i+8], 2)) for i in range(0, len(bits), 8)]
    return ''.join(chars)

@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('index.html')

@app.route('/embed', methods=['POST'])
def embed():
    image_file = request.files['cover']
    message = request.form['message']
    key = int(request.form['key'])

    filename = secure_filename(image_file.filename)
    path = os.path.join(UPLOAD_FOLDER, filename)
    image_file.save(path)

    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    encoded_img = spread_spectrum_embed(img, message, key)

    result_path = os.path.join(UPLOAD_FOLDER, 'stego_' + filename)
    cv2.imwrite(result_path, encoded_img)

    return render_template('result.html', result_image=result_path.split('/')[-1])

@app.route('/extract', methods=['POST'])
def extract():
    image_file = request.files['stego']
    length = int(request.form['length'])
    key = int(request.form['key'])

    filename = secure_filename(image_file.filename)
    path = os.path.join(UPLOAD_FOLDER, filename)
    image_file.save(path)

    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    message = spread_spectrum_extract(img, length, key)

    return render_template('extracted.html', message=message)

if __name__ == '__main__':
    app.run(debug=True)
