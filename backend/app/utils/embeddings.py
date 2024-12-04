from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.efficientnet import preprocess_input
import numpy as np

# Cargar el modelo preentrenado directamente
model = EfficientNetB0(include_top=False, weights="imagenet", pooling="avg")

def extract_embedding(image):
    """
    Genera el embedding para una imagen usando EfficientNetB0.
    """
    img = image.resize((224, 224))
    img_array = img_to_array(img)
    img_array = preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)
    return model.predict(img_array)[0]
