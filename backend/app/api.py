from flask import Blueprint, request, jsonify, send_file
from app.utils.embeddings import extract_embedding
from app.utils.similarity import find_similar_images
import numpy as np
from PIL import Image
import os
from io import BytesIO

api_bp = Blueprint("api", __name__)

# Cargar embeddings y rutas
embeddings_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models/embeddings.npy")
image_paths_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models/image_paths.txt")
embeddings = np.load(embeddings_path)
with open(image_paths_path, "r") as f:
    image_paths = f.read().splitlines()

@api_bp.route("/recommend", methods=["POST"])
def recommend():
    try:
        # Leer imagen del request
        image_file = request.files["image"]
        img = Image.open(image_file.stream)

        # Generar embedding
        query_embedding = extract_embedding(img)

        # Encontrar la imagen más similar (solo una para este caso)
        recommendations = find_similar_images(query_embedding, embeddings, image_paths, top_n=1)

        # Obtener la ruta de la imagen más similar
        similar_image_path = recommendations[0]
        abs_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "images", *similar_image_path.split("/")[-2:])

        # Enviar la imagen como respuesta
        return send_file(abs_path, mimetype="image/jpeg")

    except Exception as e:
        return jsonify({"error": str(e)}), 500
