import numpy as np
import torch


def _validate_inputs(
    user_embeddings: np.ndarray,
    item_embeddings: np.ndarray,
    interactions: list[np.ndarray],
    k: int,
) -> None:
    if k <= 0:
        raise ValueError("k debe ser mayor que 0")
    if k > item_embeddings.shape[0]:
        raise ValueError("k no puede ser mayor que la cantidad de items")
    if len(interactions) != user_embeddings.shape[0]:
        raise ValueError("interactions debe tener una entrada por usuario")


def _resolve_device(device: str) -> torch.device:
    if device not in {"auto", "cuda"}:
        raise ValueError("device debe ser 'auto' o 'cuda'")

    if torch.cuda.is_available():
        return torch.device("cuda")
    raise RuntimeError("La version Torch requiere GPU CUDA. Usa la version NumPy para CPU.")


def _recommend_block_torch(
    user_block: np.ndarray,
    item_gpu: torch.Tensor,
    item_norm_gpu: torch.Tensor,
    interactions_block: list[np.ndarray],
    k: int,
    device: torch.device,
    eps: float,
) -> tuple[np.ndarray, np.ndarray]:
    user_block = np.require(user_block, requirements=["C", "W"])
    user_gpu = torch.as_tensor(user_block, dtype=torch.float32, device=device)
    user_norm_gpu = torch.linalg.norm(user_gpu, dim=1, keepdim=True)
    scores = torch.mm(user_gpu, item_gpu.T) / (user_norm_gpu * item_norm_gpu.T + eps)

    for idx, interacted_items in enumerate(interactions_block):
        if len(interacted_items) > 0:
            item_idx = torch.as_tensor(interacted_items, dtype=torch.long, device=device)
            scores[idx, item_idx] = -torch.inf

    top_k_scores, top_k_items = torch.topk(scores, k=k, dim=1, largest=True)

    if device.type == "cuda":
        torch.cuda.synchronize(device)

    return (
        top_k_items.cpu().numpy().astype(np.int32),
        top_k_scores.cpu().numpy().astype(np.float32),
    )


def top_k_recommendations_torch(
    user_embeddings: np.ndarray,
    item_embeddings: np.ndarray,
    interactions: list[np.ndarray],
    k: int = 5,
    block_size: int | None = None,
    device: str = "auto",
    eps: float = 1e-10,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Recomendaciones top-k vectorizadas con PyTorch.

    device:
        "auto": usa CUDA si PyTorch la detecta; si no, falla.
        "cuda": exige CUDA.
    block_size:
        Cantidad de usuarios procesados por bloque. Si es None, procesa todos.
        En GPU conviene usar bloques para limitar VRAM.
    """
    _validate_inputs(user_embeddings, item_embeddings, interactions, k)
    torch_device = _resolve_device(device)

    num_users = user_embeddings.shape[0]
    if block_size is None:
        block_size = num_users
    if block_size <= 0:
        raise ValueError("block_size debe ser mayor que 0")

    with torch.no_grad():
        item_embeddings = np.require(item_embeddings, requirements=["C", "W"])
        item_gpu = torch.as_tensor(item_embeddings, dtype=torch.float32, device=torch_device)
        item_norm_gpu = torch.linalg.norm(item_gpu, dim=1, keepdim=True)

        item_blocks = []
        score_blocks = []
        for start in range(0, num_users, block_size):
            end = start + block_size
            top_items, top_scores = _recommend_block_torch(
                user_embeddings[start:end],
                item_gpu,
                item_norm_gpu,
                interactions[start:end],
                k,
                torch_device,
                eps,
            )
            item_blocks.append(top_items)
            score_blocks.append(top_scores)

    return np.vstack(item_blocks), np.vstack(score_blocks)
