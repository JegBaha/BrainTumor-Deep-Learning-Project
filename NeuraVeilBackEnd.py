from flask import Flask, request, jsonify
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import logging
import os
from flask_cors import CORS
import io
import tensorflow as tf
import json
import glob
from tensorflow.keras.applications import efficientnet, mobilenet, densenet, resnet, xception, vgg16

# oneDNN uyarılarını kapat
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

app = Flask(__name__)

# Enable CORS for all routes, allowing requests from the client's origin
CORS(app, resources={r"/*": {"origins": "*"}})

# Loglamayı yapılandır
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# GPU bellek yönetimini optimize et
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

# Mixed precision inference'ı etkinleştir
from tensorflow.keras.mixed_precision import set_global_policy
set_global_policy('mixed_float16')

# Focal Loss implementasyonu
def focal_loss_fixed(y_true, y_pred, gamma=2.0, alpha=0.25):
    y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1. - tf.keras.backend.epsilon())
    cross_entropy = -y_true * tf.math.log(y_pred)
    weight = alpha * y_true * tf.pow(1 - y_pred, gamma)
    loss = weight * cross_entropy
    return tf.reduce_mean(loss, axis=-1)

# Özel fonksiyonu kaydet
tf.keras.utils.get_custom_objects()['focal_loss_fixed'] = focal_loss_fixed

# Sınıf sıralamasını JSON dosyasından yükle (isteğe bağlı)
try:
    if os.path.exists('class_labels.json'):
        with open('class_labels.json', 'r') as f:
            classes = json.load(f)
        logger.info(f"Sınıf sıralaması JSON dosyasından yüklendi: {classes}")
    else:
        classes = ['Glioma', 'Pituitary', 'Meningioma', 'No Tumor']
        logger.info(f"Varsayılan sınıf sıralaması kullanıldı: {classes}")
except Exception as e:
    logger.error(f"Sınıf sıralaması yüklenirken hata: {str(e)}")
    classes = ['Glioma', 'Pituitary', 'Meningioma', 'No Tumor']
    logger.info(f"Hata nedeniyle varsayılan sınıf sıralaması kullanıldı: {classes}")

# Models folder path
MODELS_DIR = 'models'
os.makedirs(MODELS_DIR, exist_ok=True)

# Global variable to store the current model, its name, and hyperparameters
current_model = None
current_model_name = None
current_hyperparams = None

# Function to load hyperparameters if available
def load_hyperparameters(model_name):
    hyperparam_path = os.path.join(MODELS_DIR, f"{model_name}_hyperparams.json")
    if os.path.exists(hyperparam_path):
        with open(hyperparam_path, 'r') as f:
            hyperparams = json.load(f)
        return hyperparams
    return None

# Function to get preprocessing based on model architecture and return both function and its name
def get_preprocess_function(model_name):
    model_name_lower = model_name.lower()
    if 'efficientnet' in model_name_lower:
        return efficientnet.preprocess_input, 'efficientnet.preprocess_input'
    elif 'mobilenet' in model_name_lower:
        return mobilenet.preprocess_input, 'mobilenet.preprocess_input'
    elif 'densenet' in model_name_lower:
        return densenet.preprocess_input, 'densenet.preprocess_input'
    elif 'resnet' in model_name_lower:
        return resnet.preprocess_input, 'resnet.preprocess_input'
    elif 'xception' in model_name_lower:
        return xception.preprocess_input, 'xception.preprocess_input'
    elif 'vgg16' in model_name_lower:
        return vgg16.preprocess_input, 'vgg16.preprocess_input'
    if 'efficientnetB0' in model_name_lower:
        return efficientnet.preprocess_input, 'efficientnet.preprocess_input'
    if 'efficientnetbo' in model_name_lower:
        return efficientnet.preprocess_input, 'efficientnet.preprocess_input'
    else:
        raise ValueError(f"Unknown model architecture for {model_name}")

def load_model_by_name(model_name):
    global current_model, current_model_name, current_hyperparams
    try:
        model_path = os.path.join(MODELS_DIR, model_name)
        logger.info(f"Yükleniyor: {model_path}")
        model = load_model(model_path, custom_objects={'focal_loss_fixed': focal_loss_fixed})
        logger.info("Model başarıyla yüklendi.")
        logger.info(f"Model giriş şekli: {model.input_shape}")
        logger.info(f"Model çıkış şekli: {model.output_shape}")
        current_model = model
        current_model_name = model_name
        # Load hyperparameters if available
        current_hyperparams = load_hyperparameters(model_name)
        if current_hyperparams:
            logger.info(f"Hiperparametreler yüklendi: {current_hyperparams}")
        return True
    except Exception as e:
        logger.error(f"Model yükleme hatası ({model_name}): {str(e)}")
        return False

# Load the first available model on startup
model_files = glob.glob(os.path.join(MODELS_DIR, '*.h5'))
if model_files:
    first_model = os.path.basename(model_files[0])
    if load_model_by_name(first_model):
        logger.info(f"Başlangıç modeli yüklendi: {first_model}")
    else:
        logger.error("Başlangıç modeli yüklenemedi.")
else:
    logger.warning("Models klasöründe .h5 dosyası bulunamadı.")
    current_model_name = "No Model Available"

# Endpoint: List available models
@app.route('/model-list', methods=['GET'])
def model_list():
    try:
        logger.debug("GET /model-list isteği alındı.")
        model_files = glob.glob(os.path.join(MODELS_DIR, '*.h5'))
        model_names = [os.path.basename(f) for f in model_files]
        return jsonify({'models': model_names})
    except Exception as e:
        logger.error(f"Model listesi alınırken hata: {str(e)}")
        return jsonify({'error': str(e)}), 500

# Endpoint: Set active model
@app.route('/set-model', methods=['POST'])
def set_model():
    try:
        logger.debug("POST /set-model isteği alındı.")
        data = request.get_json()
        if 'model_name' not in data:
            logger.error("Hata: 'model_name' anahtarı bulunamadı.")
            return jsonify({'error': "'model_name' gerekli."}), 400
        model_name = data['model_name']
        model_path = os.path.join(MODELS_DIR, model_name)
        if not os.path.exists(model_path):
            logger.error(f"Hata: Model bulunamadı: {model_path}")
            return jsonify({'error': f"Model bulunamadı: {model_name}"}), 400
        if load_model_by_name(model_name):
            return jsonify({'model_name': current_model_name})
        else:
            return jsonify({'error': f"Model yüklenemedi: {model_name}"}), 500
    except Exception as e:
        logger.error(f"Model ayarlanırken hata: {str(e)}")
        return jsonify({'error': str(e)}), 500

# Endpoint: Model ismini döndür
@app.route('/model-info', methods=['GET'])
def model_info():
    try:
        logger.debug("GET /model-info isteği alındı.")
        return jsonify({'model_name': current_model_name or "No Model Loaded"})
    except Exception as e:
        logger.error(f"Model bilgisi alınırken hata: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/predict', methods=['POST'])
def predict():
    try:
        logger.debug("POST /predict isteği alındı.")
        if not current_model:
            logger.error("Hata: Model yüklü değil.")
            return jsonify({'error': "Model yüklü değil."}), 500

        if 'file' not in request.files:
            logger.error("Hata: 'file' anahtarı bulunamadı.")
            return jsonify({'error': 'Dosya gönderilmedi.'}), 400

        img_file = request.files['file']
        logger.debug(f"Dosya alındı: {img_file.filename}, Boyut: {img_file.content_length}")

        if img_file.filename == '':
            logger.error("Hata: Boş dosya gönderildi.")
            return jsonify({'error': 'Boş dosya gönderildi.'}), 400

        # İstemciden gelen isGrayscale ve resolution değerlerini kontrol et
        is_grayscale = request.form.get('isGrayscale', 'false').lower() == 'true'
        resolution = request.form.get('resolution', '150x150')  # Default to 150x150 if not provided
        logger.debug(f"Grayscale aktif mi: {is_grayscale}, Seçilen çözünürlük: {resolution}")

        # Parse the resolution (e.g., "224x224" -> (224, 224))
        try:
            target_size = tuple(map(int, resolution.split('x')))
            if len(target_size) != 2 or target_size[0] <= 0 or target_size[1] <= 0:
                raise ValueError("Invalid resolution format")
        except Exception as e:
            logger.error(f"Hata: Geçersiz çözünürlük formatı: {resolution}, Hata: {str(e)}")
            return jsonify({'error': f"Geçersiz çözünürlük: {resolution}"}), 400

        img_stream = io.BytesIO(img_file.read())
        logger.debug(f"Dosya içeriği boyutu: {img_stream.getbuffer().nbytes} bayt")
        img_stream.seek(0)

        # Görüntüyü isGrayscale ve seçilen çözünürlüğe göre yükle
        color_mode = 'grayscale' if is_grayscale else 'rgb'
        img = image.load_img(img_stream, target_size=target_size, color_mode=color_mode)
        logger.debug(f"Görüntü {color_mode} modunda yüklendi ve {target_size} boyutuna yeniden boyutlandırıldı.")

        img_array = image.img_to_array(img)
        logger.debug(f"Görüntü şekli: {img_array.shape}")

        # Eğer gri tonlamalıysa, 3 kanallı hale getir
        if img_array.shape[-1] == 1:
            img_array = np.repeat(img_array, 3, axis=-1)
            logger.debug("Görüntü gri tonlamadan RGB'ye çevrildi.")

        img_array = np.expand_dims(img_array, axis=0)
        logger.debug("Görüntü batch boyutuna genişletildi.")

        # Model bazlı ön işleme fonksiyonunu ve adını al
        preprocess_input, preprocess_name = get_preprocess_function(current_model_name)
        img_array = preprocess_input(img_array)
        logger.debug(f"Model bazlı ön işleme uygulandı: {preprocess_name}")

        prediction = current_model(img_array)
        prediction_np = np.array(prediction)
        logger.debug(f"Ham tahmin çıktısı: {prediction_np[0]}")
        logger.debug(f"Tahmin edilen indeks: {np.argmax(prediction_np[0])}")
        predicted_class = classes[np.argmax(prediction_np[0])]
        logger.debug(f"Tahmin edilen sınıf: {predicted_class}")

        return jsonify({
            'class': predicted_class,
            'probability': prediction_np[0].tolist(),
            'preprocess_function': preprocess_name  # Hangi preprocess_input kullanıldı
        })
    except Exception as e:
        logger.error(f"Tahmin sırasında hata: {str(e)}")
        return jsonify({'error': str(e)}), 400

@app.route('/predict-from-path', methods=['POST'])
def predict_from_path():
    try:
        logger.debug("POST /predict-from-path isteği alındı.")
        if not current_model:
            logger.error("Hata: Model yüklü değil.")
            return jsonify({'error': "Model yüklü değil."}), 500

        data = request.get_json()
        if 'path' not in data:
            logger.error("Hata: 'path' anahtarı bulunamadı.")
            return jsonify({'error': 'Dosya yolu gönderilmedi.'}), 400

        file_path = data['path']
        logger.debug(f"Dosya yolu alındı: {file_path}")

        if not os.path.exists(file_path):
            logger.error(f"Hata: Dosya bulunamadı: {file_path}")
            return jsonify({'error': f"Dosya bulunamadı: {file_path}"}), 400

        # İstemciden gelen isGrayscale ve resolution değerlerini kontrol et
        is_grayscale = data.get('isGrayscale', False)
        resolution = data.get('resolution', '150x150')  # Default to 150x150 if not provided
        logger.debug(f"Grayscale aktif mi: {is_grayscale}, Seçilen çözünürlük: {resolution}")

        # Parse the resolution (e.g., "224x224" -> (224, 224))
        try:
            target_size = tuple(map(int, resolution.split('x')))
            if len(target_size) != 2 or target_size[0] <= 0 or target_size[1] <= 0:
                raise ValueError("Invalid resolution format")
        except Exception as e:
            logger.error(f"Hata: Geçersiz çözünürlük formatı: {resolution}, Hata: {str(e)}")
            return jsonify({'error': f"Geçersiz çözünürlük: {resolution}"}), 400

        # Görüntüyü isGrayscale ve seçilen çözünürlüğe göre yükle
        color_mode = 'grayscale' if is_grayscale else 'rgb'
        img = image.load_img(file_path, target_size=target_size, color_mode=color_mode)
        logger.debug(f"Görüntü {color_mode} modunda yüklendi ve {target_size} boyutuna yeniden boyutlandırıldı.")

        img_array = image.img_to_array(img)
        logger.debug(f"Görüntü şekli: {img_array.shape}")

        # Eğer gri tonlamalıysa, 3 kanallı hale getir
        if img_array.shape[-1] == 1:
            img_array = np.repeat(img_array, 3, axis=-1)
            logger.debug("Görüntü gri tonlamadan RGB'ye çevrildi.")

        img_array = np.expand_dims(img_array, axis=0)
        logger.debug("Görüntü batch boyutuna genişletildi.")

        # Model bazlı ön işleme fonksiyonunu ve adını al
        preprocess_input, preprocess_name = get_preprocess_function(current_model_name)
        img_array = preprocess_input(img_array)
        logger.debug(f"Model bazlı ön işleme uygulandı: {preprocess_name}")

        prediction = current_model(img_array)
        prediction_np = np.array(prediction)
        logger.debug(f"Ham tahmin çıktısı: {prediction_np[0]}")
        logger.debug(f"Tahmin edilen indeks: {np.argmax(prediction_np[0])}")
        predicted_class = classes[np.argmax(prediction_np[0])]
        logger.debug(f"Tahmin edilen sınıf: {predicted_class}")

        return jsonify({
            'class': predicted_class,
            'probability': prediction_np[0].tolist(),
            'preprocess_function': preprocess_name  # Hangi preprocess_input kullanıldı
        })
    except Exception as e:
        logger.error(f"Tahmin sırasında hata: {str(e)}")
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)