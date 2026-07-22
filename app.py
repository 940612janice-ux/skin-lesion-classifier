import os, json, io, base64, zipfile
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import torchvision.models as models
from flask import Flask, request, jsonify, render_template, send_from_directory
from huggingface_hub import hf_hub_download

CLASS_NAMES = ['Nevus', 'Melanoma', 'BasalCellCarcinoma', 'SquamousCellCarcinoma']

MODEL_REPO = "Iris2005/skin-lesion-model"
MODEL_FILE = "pytorch_model.pth"
IMAGES_ZIP_REPO = "Iris2005/skin-deploy-files"
IMAGES_ZIP_FILE = "test_images_full.zip"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_IMG_DIR = os.path.join(BASE_DIR, 'test_images_full')
LABELS_PATH = os.path.join(TEST_IMG_DIR, 'labels.json')

# Download and extract test images if not present
if not os.path.exists(TEST_IMG_DIR):
    print("Downloading test images...")
    zip_path = hf_hub_download(repo_id=IMAGES_ZIP_REPO, filename=IMAGES_ZIP_FILE, repo_type='dataset')
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(TEST_IMG_DIR)
    print(f"Extracted to {TEST_IMG_DIR}")

with open(LABELS_PATH, 'r') as f:
    GROUND_TRUTH = json.load(f)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model_path = hf_hub_download(repo_id=MODEL_REPO, filename=MODEL_FILE)
model = models.resnet50(weights=None)
model.fc = nn.Linear(model.fc.in_features, 4)
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

app = Flask(__name__)

def predict_image(img, filename=None):
    img = img.convert('RGB').resize((224, 224))
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_array = np.transpose(img_array, (2, 0, 1))
    mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
    std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
    img_array = (img_array - mean) / std
    img_tensor = torch.tensor(img_array, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.softmax(outputs, dim=1)[0]

    pred_idx = torch.argmax(probs).item()
    pred_class = CLASS_NAMES[pred_idx]
    confidence = round(probs[pred_idx].item(), 4)
    all_probs = {cls: round(probs[i].item(), 4) for i, cls in enumerate(CLASS_NAMES)}
    ground_truth = GROUND_TRUTH.get(filename, 'Unknown') if filename else None

    return pred_class, confidence, all_probs, ground_truth

@app.route('/')
def index():
    return render_template('index.html')

# 處理的是使用者從網頁表單或 API 選擇的本地圖片檔案
@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']
    img = Image.open(file.stream)
    pred_class, confidence, all_probs, _ = predict_image(img)

    return jsonify({
        'predicted': pred_class,
        'confidence': confidence,
        'all_probs': all_probs
    })

# 解碼成二進位
@app.route('/predict_base64', methods=['POST'])
def predict_base64():
    data = request.get_json()
    if not data or 'image' not in data:
        return jsonify({'error': 'No image data'}), 400

    img_data = base64.b64decode(data['image'])
    img = Image.open(io.BytesIO(img_data))
    pred_class, confidence, all_probs, _ = predict_image(img)

    return jsonify({
        'predicted': pred_class,
        'confidence': confidence,
        'all_probs': all_probs
    })

# 沒有經過模型推論，只是純粹的靜態檔案傳送
@app.route('/test_images/<path:filename>')    
def test_image(filename):
    return send_from_directory(TEST_IMG_DIR, filename)

# 處理系統預設測試集資料夾內的圖片（已有名確標準答案）
@app.route('/test_images')
def list_test_images():
    limit = request.args.get('limit', 50, type=int)
    offset = request.args.get('offset', 0, type=int)
    try:
        all_files = sorted([f for f in os.listdir(TEST_IMG_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    except FileNotFoundError:
        return jsonify({'error': 'Test images not found. Delete test_images/ and restart to re-download.'}), 500

    page = all_files[offset:offset + limit]
    result = []
    for f in page:
        result.append({
            'filename': f,
            'ground_truth': GROUND_TRUTH.get(f, 'Unknown')
        })
    return jsonify({'total': len(all_files), 'images': result})

# 送入模型並同步查詢標準答案
@app.route('/predict_test', methods=['POST'])
def predict_test():
    data = request.get_json()
    if not data or 'filename' not in data:
        return jsonify({'error': 'No filename provided'}), 400

    filename = data['filename']
    img_path = os.path.join(TEST_IMG_DIR, filename)
    if not os.path.exists(img_path):
        return jsonify({'error': f'File not found: {filename}'}), 404

    img = Image.open(img_path)
    pred_class, confidence, all_probs, ground_truth = predict_image(img, filename=filename)

    return jsonify({
        'filename': filename,
        'predicted': pred_class,
        'confidence': confidence,
        'all_probs': all_probs,
        'ground_truth': ground_truth,
        'correct': pred_class == ground_truth
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=7860)
