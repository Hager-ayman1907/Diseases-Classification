import tensorflow as tf
from tensorflow.keras.models import load_model
import streamlit as st
import numpy as np
from PIL import Image
import os
from tensorflow.keras.preprocessing import image as tf_image  # تحديد الاستيراد بشكل صريح
import matplotlib.pyplot as plt

# Header
st.header('Disease Classification Model')

# Load the pre-trained model
model = load_model('my_model.h5')

# Categories of fruits and vegetables
data_cat = ['1. Eczema 1677', 
            '10. Warts Molluscum and other Viral Infections - 2103',
              '2. Melanoma 15.75k', 
              '3. Atopic Dermatitis - 1.25k', 
              '4. Basal Cell Carcinoma (BCC) 3323', 
              '5. Melanocytic Nevi (NV) - 7970',
              '6. Benign Keratosis-like Lesions (BKL) 2624', 
              '7. Psoriasis pictures Lichen Planus and related diseases - 2k', 
              '8. Seborrheic Keratoses and other Benign Tumors - 1.8k', 
              '9. Tinea Ringworm Candidiasis and other Fungal Infections - 1.7k']

class_names_dict = {
    '1. Eczema 1677': 'Eczema',
    '10. Warts Molluscum and other Viral Infections - 2103': 'Warts, Molluscum, and Viral Infections',
    '2. Melanoma 15.75k': 'Melanoma',
    '3. Atopic Dermatitis - 1.25k': 'Atopic Dermatitis',
    '4. Basal Cell Carcinoma (BCC) 3323': 'Basal Cell Carcinoma',
    '5. Melanocytic Nevi (NV) - 7970': 'Melanocytic Nevi',
    '6. Benign Keratosis-like Lesions (BKL) 2624': 'Benign Keratosis-like Lesions',
    '7. Psoriasis pictures Lichen Planus and related diseases - 2k': 'Psoriasis and related diseases',
    '8. Seborrheic Keratoses and other Benign Tumors - 1.8k': 'Seborrheic Keratoses',
    '9. Tinea Ringworm Candidiasis and other Fungal Infections - 1.7k': 'Fungal Infections'
}
data_cat_new = [
    'Eczema',
    'Warts, Molluscum, and Viral Infections',
    'Melanoma',
    'Atopic Dermatitis',
    'Basal Cell Carcinoma',
    'Melanocytic Nevi',
    'Benign Keratosis-like Lesions',
    'Psoriasis and related diseases',
    'Seborrheic Keratoses',
    'Fungal Infections'
]

# دالة لتحميل وتحضير الصورة
def load_and_prepare_image(img_path, target_size=(256, 256)):
    img = tf_image.load_img(img_path, target_size=target_size)  # استخدام دالة load_img من tensorflow
    img_array = tf_image.img_to_array(img)
    img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)  # Normalize
    return np.expand_dims(img_array, axis=0), img  # Add batch dimension

# إنشاء مجلد 'Images' إذا لم يكن موجودًا
if not os.path.exists('Images'):
    os.makedirs('Images')

# رفع الصورة باستخدام Streamlit
image_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if image_file is not None:
    # حفظ الصورة التي تم رفعها إلى مجلد 'Images'
    img_path = os.path.join('Images', image_file.name)
    with open(img_path, "wb") as f:
        f.write(image_file.getbuffer())

    # فتح الصورة التي تم رفعها
    image = Image.open(img_path)
    
    # عرض الصورة
    st.image(image, caption="Uploaded Image")

    # تحميل وتحضير الصورة للمعالجة
    processed_img, original_img = load_and_prepare_image(img_path)

    # التنبؤ باستخدام النموذج
    pred_probs = model.predict(processed_img)
    predicted_index = np.argmax(pred_probs)
    confidence = np.max(pred_probs)

    # عرض النتيجة
    st.write(f"Disease in image is: {data_cat_new[predicted_index]}")

