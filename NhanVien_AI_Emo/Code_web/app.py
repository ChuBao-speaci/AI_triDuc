import glob
import os
import cv2
import torch
import numpy as np
from flask import Flask, render_template, request, jsonify, session, redirect, url_for
from PIL import Image
from torchvision import transforms
from ultralytics import YOLO

try:
    from model import PAttLite
except ImportError:
    print("Không tìm thấy file model.py!")

# --- CONFIG ---
# Get absolute paths based on file location
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FACE_MODEL = YOLO(os.path.join(BASE_DIR, 'runs/detect/yolov26_e20/weights/best.pt'))
EMO_MODEL_PATH = os.path.join(BASE_DIR, 'runs', 'PAtt_Lite_final.pth')
if not os.path.exists(EMO_MODEL_PATH):
    alt_path = os.path.join(BASE_DIR, 'runs', 'runs', 'PAtt_Lite_final.pth')
    if os.path.exists(alt_path):
        EMO_MODEL_PATH = alt_path
    else:
        matches = glob.glob(os.path.join(BASE_DIR, '**', 'PAtt_Lite_final.pth'), recursive=True)
        if matches:
            EMO_MODEL_PATH = matches[0]
        else:
            raise FileNotFoundError(
                f"Emotion model not found. Checked: {os.path.join(BASE_DIR, 'runs', 'PAtt_Lite_final.pth')} "
                f"and {alt_path}"
            )
CLASSES = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Load emotion model
emo_net = PAttLite(num_classes=7)
emo_net.load_state_dict(torch.load(EMO_MODEL_PATH, map_location=DEVICE, weights_only=True))
emo_net.to(DEVICE).eval()

data_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

base_dir = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__,
            template_folder=os.path.join(base_dir, 'templates'),
            static_folder=os.path.join(base_dir, 'static'))

# Simple session secret (override in environment for production)
app.secret_key = os.environ.get('SECRET_KEY', 'dev_secret_key')

# Simple in-memory employee credentials (replace with DB in production)
EMPLOYEES = {
    'alice': 'password1',
    'bob': 'password2'
}


@app.route("/")
def index():
    # require login
    if not session.get('user'):
        return redirect(url_for('login'))
    return render_template("index.html", user=session.get('user'))


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        if username in EMPLOYEES and EMPLOYEES[username] == password:
            session['user'] = username
            return redirect(url_for('index'))
        else:
            return render_template('login.html', error='Sai tên đăng nhập hoặc mật khẩu')
    return render_template('login.html')


@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect(url_for('login'))


@app.route("/predict", methods=["POST"])
def predict_emotion():
    # require login for predictions
    if not session.get('user'):
        return jsonify({"error": "Unauthorized"}), 401
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']
    img_bytes = file.read()
    nparr = np.frombuffer(img_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if frame is None:
        return jsonify({"error": "Invalid image"}), 400

    results = FACE_MODEL(frame, verbose=False, conf=0.5)[0]
    output = []

    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        face = frame[max(0, y1):y2, max(0, x1):x2]

        if face.size > 0:
            pil_img = Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
            tensor = data_transform(pil_img).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                logits = emo_net(tensor)
                probs = torch.nn.functional.softmax(logits, dim=1).cpu().numpy()[0]
                idx = int(np.argmax(probs))

            # Build all_scores dict for the frontend emotion bars
            all_scores = {cls: float(probs[i] * 100) for i, cls in enumerate(CLASSES)}

            output.append({
                "bbox":       [x1, y1, x2, y2],
                "label":      CLASSES[idx],
                "score":      float(probs[idx] * 100),
                "all_scores": all_scores,
            })

    return jsonify({"results": output})


if __name__ == "__main__":
   app.run(host="0.0.0.0", port=9000, debug=True)