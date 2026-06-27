import os

RAY_WORKER_ENV_VARS = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO": "0",
}

for env_name, env_value in RAY_WORKER_ENV_VARS.items():
    os.environ.setdefault(env_name, env_value)

import numpy as np
import ray


def init_ray():
    """
    Inicializa Ray si aun no esta inicializado.
    """
    if not ray.is_initialized():
        for name, value in RAY_WORKER_ENV_VARS.items():
            os.environ.setdefault(name, value)
        ray.init()

    return {
        "cpus": ray.available_resources().get("CPU", 1),
        "gpus": ray.available_resources().get("GPU", 0),
    }


def _cosine_similarity_np(a: np.ndarray, b: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    """
    Calcula la similitud coseno usando NumPy en CPU.
    """
    a_norm = np.linalg.norm(a, axis=1, keepdims=True)
    b_norm = np.linalg.norm(b, axis=1, keepdims=True)
    return np.dot(a, b.T) / (a_norm * b_norm.T + eps)


def _top_k_from_scores_np(
    scores: np.ndarray,
    interactions_block: list[np.ndarray],
    k: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Filtra items ya vistos y obtiene top-k usando NumPy.
    """
    for idx, interacted_items in enumerate(interactions_block):
        if len(interacted_items) > 0:
            scores[idx, interacted_items] = -np.inf

    top_k_indices_unsorted = np.argpartition(-scores, k - 1, axis=1)[:, :k]
    top_k_scores_unsorted = np.take_along_axis(scores, top_k_indices_unsorted, axis=1)
    top_k_indices_sorted = np.argsort(-top_k_scores_unsorted, axis=1)

    top_k_items = np.take_along_axis(
        top_k_indices_unsorted,
        top_k_indices_sorted,
        axis=1,
    )
    top_k_scores = np.take_along_axis(
        top_k_scores_unsorted,
        top_k_indices_sorted,
        axis=1,
    )

    return top_k_items.astype(np.int32), top_k_scores.astype(np.float32)


def _recommend_block_cpu(
    user_block: np.ndarray,
    item_embeddings: np.ndarray,
    interactions_block: list[np.ndarray],
    k: int,
) -> tuple[np.ndarray, np.ndarray]:
    scores = _cosine_similarity_np(user_block, item_embeddings)
    return _top_k_from_scores_np(scores, interactions_block, k)


def _recommend_block_cuda(
    user_block: np.ndarray,
    item_embeddings: np.ndarray,
    interactions_block: list[np.ndarray],
    k: int,
    eps: float = 1e-10,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Ejecuta el bloque en CUDA con PyTorch.
    """
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("Ray detecto GPU, pero PyTorch no detecta CUDA en el worker.")

    with torch.no_grad():
        torch_device = torch.device("cuda")
        user_block = np.require(user_block, requirements=["C", "W"])
        item_embeddings = np.require(item_embeddings, requirements=["C", "W"])
        user_gpu = torch.as_tensor(user_block, device=torch_device)
        item_gpu = torch.as_tensor(item_embeddings, device=torch_device)

        user_norm = torch.linalg.norm(user_gpu, dim=1, keepdim=True)
        item_norm = torch.linalg.norm(item_gpu, dim=1, keepdim=True)
        scores = torch.mm(user_gpu, item_gpu.T) / (user_norm * item_norm.T + eps)

        for idx, interacted_items in enumerate(interactions_block):
            if len(interacted_items) > 0:
                interacted_items = np.array(interacted_items, dtype=np.int64, copy=True)
                item_idx = torch.as_tensor(
                    interacted_items,
                    dtype=torch.long,
                    device=torch_device,
                )
                scores[idx, item_idx] = -torch.inf

        top_k_scores, top_k_items = torch.topk(scores, k=k, dim=1, largest=True)

        return (
            top_k_items.cpu().numpy().astype(np.int32),
            top_k_scores.cpu().numpy().astype(np.float32),
        )


@ray.remote
def recommend_block_ray_cpu(
    user_block: np.ndarray,
    item_embeddings: np.ndarray,
    interactions_block: list[np.ndarray],
    k: int,
) -> tuple[np.ndarray, np.ndarray]:
    return _recommend_block_cpu(user_block, item_embeddings, interactions_block, k)


@ray.remote(num_gpus=1)
def recommend_block_ray_cuda(
    user_block: np.ndarray,
    item_embeddings: np.ndarray,
    interactions_block: list[np.ndarray],
    k: int,
) -> tuple[np.ndarray, np.ndarray]:
    return _recommend_block_cuda(user_block, item_embeddings, interactions_block, k)


def _should_use_cuda(device: str) -> bool:
    if device not in {"auto", "cuda", "cpu"}:
        raise ValueError("device debe ser 'auto', 'cuda' o 'cpu'")

    if device == "cpu":
        return False

    resources = init_ray()
    has_ray_gpu = resources["gpus"] > 0
    has_torch_cuda = _torch_cuda_is_available()

    if device == "cuda" and not has_ray_gpu:
        raise RuntimeError("device='cuda' solicitado, pero Ray no detecta GPUs.")
    if device == "cuda" and not has_torch_cuda:
        raise RuntimeError("device='cuda' solicitado, pero PyTorch no detecta CUDA.")

    return has_ray_gpu and has_torch_cuda


def _torch_cuda_is_available() -> bool:
    try:
        import torch
    except ImportError:
        return False
    return torch.cuda.is_available()


def top_k_recommendations_ray(
    user_embeddings: np.ndarray,
    item_embeddings: np.ndarray,
    interactions: list[np.ndarray],
    k: int,
    block_size: int = 1000,
    device: str = "auto",
    max_in_flight: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Recomendaciones top-k con Ray.

    device:
        "auto": usa CUDA si Ray y PyTorch detectan GPU; si no, CPU.
        "cuda": exige GPU detectada por Ray y CUDA detectada por PyTorch.
        "cpu": fuerza CPU.
    """
    if k <= 0:
        raise ValueError("k debe ser mayor que 0")
    if k > item_embeddings.shape[0]:
        raise ValueError("k no puede ser mayor que la cantidad de items")
    if len(interactions) != user_embeddings.shape[0]:
        raise ValueError("interactions debe tener una entrada por usuario")
    if max_in_flight is not None and max_in_flight <= 0:
        raise ValueError("max_in_flight debe ser mayor que 0")

    use_cuda = _should_use_cuda(device)
    remote_task = recommend_block_ray_cuda if use_cuda else recommend_block_ray_cpu

    num_users = user_embeddings.shape[0]
    item_embeddings_ref = ray.put(item_embeddings)

    futures = []
    indexed_futures = {}
    results_by_block = []
    for start in range(0, num_users, block_size):
        end = start + block_size
        future = remote_task.remote(
            user_embeddings[start:end],
            item_embeddings_ref,
            interactions[start:end],
            k,
        )
        futures.append(future)
        indexed_futures[future] = len(results_by_block)
        results_by_block.append(None)

        if max_in_flight is not None and len(futures) >= max_in_flight:
            ready, futures = ray.wait(futures, num_returns=1)
            ready_future = ready[0]
            results_by_block[indexed_futures.pop(ready_future)] = ray.get(ready_future)

    while futures:
        ready, futures = ray.wait(futures, num_returns=1)
        ready_future = ready[0]
        results_by_block[indexed_futures.pop(ready_future)] = ray.get(ready_future)

    results = [result for result in results_by_block if result is not None]
    top_k_items = np.vstack([result[0] for result in results])
    top_k_scores = np.vstack([result[1] for result in results])

    return top_k_items, top_k_scores
