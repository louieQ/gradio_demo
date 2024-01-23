from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import (MobileNetV2,
                                                        decode_predictions,
                                                        preprocess_input)
from tensorflow.keras.preprocessing import image

# Load the MobileNetV2 model pre-trained on ImageNet
model = MobileNetV2(weights='imagenet')

# Function to classify whether an image contains a face or not
def classify_face(image_path):
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    predictions = model.predict(img_array)
    decoded_predictions = decode_predictions(predictions)

    # Check if any of the top predictions contain a face label
    for _, label, _ in decoded_predictions[0]:
        if 'face' in label.lower():
            return True

    return False

# Example usage
image_path = Path('test/resources/imgs/RandomFace.jpg')
result = classify_face(image_path)

if result:
    print("The image contains a face.")
else:
    print("No face detected in the image.")
