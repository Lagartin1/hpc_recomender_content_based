import numpy as np


def generate_items_embeddings(num_items, embedding_dim, dtype=np.float32,seed=42):
    """
    Genera embeddings de items aleatorios.

    Args:
        num_items (int): Número de items.
        embedding_dim (int): Dimensión de los embeddings.
        dtype (type): Tipo de dato del array.
        seed (int): Semilla para la generación de números aleatorios.
    Returns:
        np.ndarray: Matriz de embeddings de items de forma (num_items, embedding_dim).
    """
    
    
    gen = np.random.default_rng(seed=seed)
    enbeddings = gen.normal(0, 1, size=(num_items, embedding_dim)).astype(dtype)    
    
    
    return enbeddings
  
  
  
  
  
def generate_user_interactions(num_users, num_items, max_interactions_per_user, seed=42):
    """
    Genera interacciones aleatorias de usuarios con items.

    Args:
        num_users (int): Número de usuarios.
        num_items (int): Número de items.
        max_interactions_per_user (int): Número máximo de interacciones por usuario.
        seed (int): Semilla para la generación de números aleatorios.

    Returns:
        list[np.ndarray]: Lista de arrays que representan los índices de items con los que cada usuario ha interactuado.
    """
    
    gen = np.random.default_rng(seed=seed)
    interactions = []
    
    for _ in range(num_users):
        num_interactions = gen.integers(1, max_interactions_per_user + 1)
        interacted_items = gen.choice(num_items, size=num_interactions, replace=False)
        interactions.append(interacted_items)
    
    return interactions
  
  
def build_user_profiles(user_embeddings, item_embeddings, interactions):
    """
    Construye perfiles de usuario promediando los embeddings de los items con los que han interactuado.

    Args:
        user_embeddings (np.ndarray): Matriz de embeddings de usuarios de forma (num_users, embedding_dim).
        item_embeddings (np.ndarray): Matriz de embeddings de items de forma (num_items, embedding_dim).
        interactions (list[np.ndarray]): Lista de arrays que representan los índices de items con los que cada usuario ha interactuado.

    Returns:
        np.ndarray: Matriz de perfiles de usuario de forma (num_users, embedding_dim).
    """
    
    num_users = len(interactions)
    embedding_dim = item_embeddings.shape[1]
    
    user_profiles = np.zeros((num_users, embedding_dim), dtype=np.float32)
    
    for user_idx, interacted_items in enumerate(interactions):
        if len(interacted_items) > 0:
            user_profiles[user_idx] = np.mean(item_embeddings[interacted_items], axis=0)
    
    return user_profiles