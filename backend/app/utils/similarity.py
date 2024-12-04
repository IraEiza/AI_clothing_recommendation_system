from sklearn.metrics.pairwise import cosine_similarity

def find_similar_images(query_embedding, embeddings, image_paths, top_n=5):
    """
    Encuentra las imágenes más similares basándose en la similitud coseno.
    
    Args:
        query_embedding (numpy.array): Embedding de la consulta.
        embeddings (numpy.array): Embeddings del catálogo.
        image_paths (list): Lista de rutas de las imágenes del catálogo.
        top_n (int): Número de imágenes similares a devolver.

    Returns:
        list: Lista de rutas de las imágenes más similares.
    """
    similarities = cosine_similarity([query_embedding], embeddings)[0]
    indices = similarities.argsort()[::-1][:top_n]
    return [image_paths[i] for i in indices]
