import numpy as np
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.applications.inception_v3 import (
    preprocess_input, decode_predictions
)
from tensorflow.keras.utils import load_img, img_to_array

# Load the pretrained model
model = InceptionV3(weights="imagenet")

# Load and preprocess the image
image = load_img("dog.jpg", target_size=(299, 299))
image_array = img_to_array(image)
image_array = np.expand_dims(image_array, axis=0)
image_array = preprocess_input(image_array)

# Predict the image category
predictions = model.predict(image_array, verbose=0)
results = decode_predictions(predictions, top=3)[0]

# Display the results
for _, label, confidence in results:
    print(f"{label}: {confidence * 100:.2f}%")