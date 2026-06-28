import numpy as np


def cosine_similarity(a, b, eps=1e-10):
    """
    Calcula la similitud coseno entre dos matrices de vectores.
    """
    a_norm = np.linalg.norm(a, axis=1, keepdims=True)
    b_norm = np.linalg.norm(b, axis=1, keepdims=True)
    return np.dot(a, b.T) / (a_norm * b_norm.T + eps)


def top_k_recommendations(
    user_embedding,
    item_embeddings,
    interactions: list[np.ndarray],
    k=5,
    block_size: int = 1000,
):
    """
    user_embedding:
        Matriz User X D.
    item_embeddings:
        Matriz Item X D.
    interactions:
        Lista de indices de items ya vistos por usuario.
    k:
        Cantidad de recomendaciones por usuario.
    block_size:
        Cantidad de usuarios procesados por bloque para limitar RAM.

    Returns:
        top_k_items, top_k_scores.
    """
    if k <= 0:
        raise ValueError("k debe ser mayor que 0")
    if k > item_embeddings.shape[0]:
        raise ValueError("k no puede ser mayor que la cantidad de items")
    if len(interactions) != user_embedding.shape[0]:
        raise ValueError("interactions debe tener una entrada por usuario")
    if block_size <= 0:
        raise ValueError("block_size debe ser mayor que 0")

    num_users = user_embedding.shape[0]
    top_k_items = np.empty((num_users, k), dtype=np.int32)
    top_k_scores = np.empty((num_users, k), dtype=np.float32)

    for start in range(0, num_users, block_size):
        end = min(start + block_size, num_users)
        scores = cosine_similarity(user_embedding[start:end], item_embeddings)

        for local_idx, interacted_items in enumerate(interactions[start:end]):
            if len(interacted_items) > 0:
                scores[local_idx, interacted_items] = -np.inf

        top_k_indices_unsorted = np.argpartition(-scores, k - 1, axis=1)[:, :k]
        top_k_scores_unsorted = np.take_along_axis(
            scores,
            top_k_indices_unsorted,
            axis=1,
        )
        top_k_indices_sorted = np.argsort(-top_k_scores_unsorted, axis=1)
        top_k_items[start:end] = np.take_along_axis(
            top_k_indices_unsorted,
            top_k_indices_sorted,
            axis=1,
        ).astype(np.int32)
        top_k_scores[start:end] = np.take_along_axis(
            top_k_scores_unsorted,
            top_k_indices_sorted,
            axis=1,
        ).astype(np.float32)

    return top_k_items, top_k_scores
