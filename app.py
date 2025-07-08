
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Disable CUDA GPU usage
os.environ["TF_DIRECTML_VISIBLE_DEVICES"] = ""  # Disable DirectML GPU usage
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
import numpy as np
import cv2
import tensorflow as tf
import requests

app = Flask(__name__)

# Set UPLOAD_FOLDER to relative path with fallback for deployment
UPLOAD_FOLDER = os.path.join(os.getcwd(), 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
 # Create folder if it doesn't exist
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024  # 5MB upload limit

# Log the path for debugging
print(f"📂 UPLOAD_FOLDER set to: {UPLOAD_FOLDER}")

# Labels
labels = [
    '12 Eczema - Finger Tips', '13 Eczema - Foot', '15 Eczema - Hand',
    '16 Eczema - Nummular', '18 Eczema - Sub Acute', '19 Lichen - Planus',
    '2 Acne - Cystic', '20 Melanoma - Malignant', '21 Melanoma - Nevi',
    '24 Nail Infection - Others', '27 Pityriasis - Rosea', '29 Psoriasis - Chronic Plaque',
    '31 Psoriasis - Guttate', '33 Psoriasis - Nail', '34 Psoriasis - Others',
    '35 Psoriasis - Palms and Soles', '37 Psoriasis - Scalp', '38 Rosacea',
    '7 Atopic - Adult', '9 Benign Keratosis'
]

image_size = 224

# Function to download large Google Drive files
def download_large_gdrive_file(file_id, destination, max_retries=3):
    def get_confirm_token(response):
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                return value
        return None

    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    session = requests.Session()
    
    for attempt in range(max_retries):
        print(f"📡 Download attempt {attempt + 1}/{max_retries}")
        response = session.get(url, stream=True)
        
        # Log response headers for debugging
        print(f"Response headers: {response.headers}")
        
        # Check for virus scan warning
        token = get_confirm_token(response)
        if token:
            params = {'id': file_id, 'confirm': token}
            response = session.get(url, params=params, stream=True)
            print(f"Using confirmation token: {token}")
        
        # Verify response status and content type
        if response.status_code != 200:
            print(f"Download failed with status code {response.status_code}")
            continue
        
        content_type = response.headers.get('Content-Type', '')
        if 'text/html' in content_type:
            print("Downloaded content is HTML, not the model file. Retrying...")
            continue

        # Save file in chunks
        with open(destination, 'wb') as f:
            for chunk in response.iter_content(32768):
                if chunk:
                    f.write(chunk)
        print(f"📥 Downloaded model to {destination}")

        # Verify file size (expected ~211 MB)
        file_size = os.path.getsize(destination) / (1024 * 1024)  # Size in MB
        if file_size < 10:  # Threshold for invalid file
            print(f"Downloaded file is too small ({file_size:.2f} MB), likely corrupted. Retrying...")
            continue
        print(f"✅ Downloaded file size: {file_size:.2f} MB")
        return  # Success, exit function
    
    raise Exception(f"Failed to download valid model file after {max_retries} attempts")

# Load the model
def load_model():
    try:
        model_path = os.path.join('models', 'Skin_Resnet50_FT.h5')
        if not os.path.exists(model_path):
            os.makedirs('models', exist_ok=True)
            print(f"📥 Downloading model to {model_path}")
            # New Google Drive file ID
            file_id = '1dpFrrqee65E3xOv-S_Fw_UnXFr6Kb_q8'
            download_large_gdrive_file(file_id, model_path)
        model = tf.keras.models.load_model(model_path)
        print(f"✅ Model loaded successfully from {model_path}")
        return model
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        raise

model = load_model()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        print("❌ No file part in request")
        return render_template('index.html', error='No file uploaded')
    
    file = request.files['file']
    if file.filename == '':
        print("❌ No file selected")
        return render_template('index.html', error='No file selected')

    allowed_extensions = {'png', 'jpg', 'jpeg'}
    file_ext = file.filename.rsplit('.', 1)[-1].lower()
    if file_ext not in allowed_extensions:
        print(f"❌ Invalid file format: {file_ext}")
        return render_template('index.html', error='Invalid file format. Please upload PNG or JPEG.')

    try:
        # Save file to uploads folder
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        print(f"📂 Attempting to save file to: {file_path}")
        file.save(file_path)
        if os.path.exists(file_path):
            print(f"✅ File saved successfully at: {file_path}")
        else:
            print(f"❌ File not saved at: {file_path}")
            return render_template('index.html', error='Failed to save file')

        # Load and preprocess the saved image
        img = cv2.imread(file_path)
        if img is None:
            print(f"❌ Failed to load image: {file_path}")
            return render_template('index.html', error='Failed to load image')
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (image_size, image_size))
        img = tf.keras.applications.resnet50.preprocess_input(img)
        img = np.expand_dims(img, axis=0)

        # Predict
        prediction = model.predict(img)
        predicted_class = labels[np.argmax(prediction)]
        confidence = np.max(prediction) * 100

        # Browser-friendly image path
        uploaded_image_path = f"/static/uploads/{filename}"
        print(f"🌐 Browser image path: {uploaded_image_path}")

        return render_template(
            'index.html',
            prediction=predicted_class,
            confidence=f"{confidence:.2f}%",
            uploaded_image=uploaded_image_path
        )

    except Exception as e:
        print(f"❌ Error processing image: {str(e)}")
        return render_template('index.html', error=f'Error processing image: {str(e)}')

@app.route('/Uploads-list')
def uploads_list():
    try:
        files = os.listdir(app.config['UPLOAD_FOLDER'])
        if not files:
            print("📂 No files found")
            return render_template('index.html', error='No uploaded images')
        print(f"📂 Found files: {files}")
        return render_template('Uploads.html', files=files)
    except Exception as e:
        print(f"⚠️ Error listing uploads: {str(e)}")
        return render_template('index.html', error=f'Error listing uploads: {str(e)}')

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
