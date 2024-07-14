# image-recognisation-code-python

Example of an image recognition system using Python and the TensorFlow library with the pre-trained model MobileNetV2. This code will recognize the primary object in an image and provide a label for it.

First, ensure you have the necessary libraries installed:

```sh
pip install tensorflow pillow
```

Now, here's a simple Python script to perform image recognition:

```python
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import numpy as np
import sys

def load_and_preprocess_image(img_path, target_size=(224, 224)):
    """Load and preprocess an image."""
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

def predict_image(img_path):
    """Predict the object in the image."""
    model = MobileNetV2(weights='imagenet')
    img_array = load_and_preprocess_image(img_path)
    predictions = model.predict(img_array)
    decoded_predictions = decode_predictions(predictions, top=1)[0]
    return decoded_predictions

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Please provide the image file path as an argument.")
        sys.exit(1)

    img_path = sys.argv[1]
    predictions = predict_image(img_path)
    
    for i, (imagenet_id, label, score) in enumerate(predictions):
        print(f"{i + 1}. {label}: {score:.2f}")

# Save this as image_recognition.py
# Run with: python image_recognition.py path_to_your_image.jpg
```

### Explanation:

1. **Imports**: The necessary libraries are imported.
2. **Load and Preprocess Image**: The `load_and_preprocess_image` function loads an image from the given path, resizes it to the target size (224x224 for MobileNetV2), converts it to an array, expands dimensions to fit the model's input shape, and preprocesses the image.
3. **Predict Image**: The `predict_image` function loads the pre-trained MobileNetV2 model with ImageNet weights, preprocesses the input image, makes predictions, and decodes the top prediction.
4. **Main Block**: The script checks if an image path is provided as a command-line argument, calls the prediction function, and prints the result.

### Running the Script:

1. Save the script as `image_recognition.py`.
2. Run it from the command line, providing the path to an image as an argument:

```sh
python image_recognition.py path_to_your_image.jpg
```

This script will load the image, process it, and print the top prediction with its confidence score. You can modify the `target_size` parameter and the number of predictions to display (`top=1`) as needed for your application.
