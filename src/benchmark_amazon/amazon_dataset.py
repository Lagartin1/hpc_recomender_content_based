import gzip
import hashlib
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np


TOKEN_RE = re.compile(r"[a-z0-9]+")


@dataclass
class AmazonDataset:
    user_profiles: np.ndarray
    item_embeddings: np.ndarray
    interactions: list[np.ndarray]
    user_ids: list[str]
    item_ids: list[str]


def load_amazon_electronics(
    reviews_path: str | Path,
    metadata_path: str | Path,
    max_users: int,
    max_items: int,
    max_interactions: int,
    dim: int,
    min_rating: float = 4.0,
    min_user_interactions: int = 2,
    seed: int = 42,
) -> AmazonDataset:
    """
    Carga un subconjunto real de Amazon Reviews 2023 Electronics.

    La lectura es por streaming sobre JSONL gzip para evitar descomprimir los
    archivos completos. Los embeddings se construyen desde metadata textual con
    hashing deterministico, sin depender de modelos externos.
    """
    if max_users <= 0 or max_items <= 0 or max_interactions <= 0:
        raise ValueError("max_users, max_items y max_interactions deben ser positivos")
    if dim <= 0:
        raise ValueError("dim debe ser positivo")

    user_items = _read_positive_interactions(
        reviews_path,
        max_interactions=max_interactions,
        min_rating=min_rating,
    )
    user_items = {
        user_id: items
        for user_id, items in user_items.items()
        if len(items) >= min_user_interactions
    }
    if not user_items:
        raise RuntimeError("No se encontraron usuarios suficientes con interacciones positivas.")

    item_counts = Counter(item for items in user_items.values() for item in items)
    selected_items = [item for item, _ in item_counts.most_common(max_items)]
    selected_item_set = set(selected_items)

    ranked_users = sorted(
        user_items,
        key=lambda user_id: len(user_items[user_id] & selected_item_set),
        reverse=True,
    )
    selected_users = []
    filtered_user_items: list[set[str]] = []
    for user_id in ranked_users:
        items = user_items[user_id] & selected_item_set
        if len(items) >= min_user_interactions:
            selected_users.append(user_id)
            filtered_user_items.append(items)
        if len(selected_users) >= max_users:
            break

    metadata_text = _read_metadata_text(metadata_path, set(selected_items))
    selected_items = [item for item in selected_items if item in metadata_text]
    if not selected_items:
        raise RuntimeError("Los items seleccionados no tienen metadata textual utilizable.")

    selected_item_set = set(selected_items)
    item_to_idx = {item_id: idx for idx, item_id in enumerate(selected_items)}
    item_embeddings = np.vstack(
        [
            text_to_hash_embedding(metadata_text[item_id], dim=dim, seed=seed)
            for item_id in selected_items
        ],
    ).astype(np.float32)

    user_ids = []
    interactions = []
    for user_id, items in zip(selected_users, filtered_user_items):
        indices = sorted(item_to_idx[item_id] for item_id in items if item_id in selected_item_set)
        if indices:
            user_ids.append(user_id)
            interactions.append(np.array(indices, dtype=np.int32))

    if not interactions:
        raise RuntimeError("No quedaron usuarios con items que tengan metadata.")

    user_profiles = build_user_profiles_from_items(item_embeddings, interactions)
    return AmazonDataset(
        user_profiles=user_profiles,
        item_embeddings=item_embeddings,
        interactions=interactions,
        user_ids=user_ids,
        item_ids=selected_items,
    )


def _read_positive_interactions(
    reviews_path: str | Path,
    max_interactions: int,
    min_rating: float,
) -> dict[str, set[str]]:
    user_items: dict[str, set[str]] = {}
    interactions = 0

    with gzip.open(reviews_path, "rt", encoding="utf-8") as file:
        for line in file:
            row = json.loads(line)
            if float(row.get("rating", 0.0)) < min_rating:
                continue

            user_id = row.get("user_id")
            item_id = row.get("parent_asin") or row.get("asin")
            if not user_id or not item_id:
                continue

            items = user_items.setdefault(user_id, set())
            previous_len = len(items)
            items.add(item_id)
            if len(items) > previous_len:
                interactions += 1

            if interactions >= max_interactions:
                break

    return user_items


def _read_metadata_text(
    metadata_path: str | Path,
    wanted_items: set[str],
) -> dict[str, str]:
    metadata_text = {}

    with gzip.open(metadata_path, "rt", encoding="utf-8") as file:
        for line in file:
            row = json.loads(line)
            item_id = row.get("parent_asin")
            if item_id not in wanted_items:
                continue

            text = metadata_to_text(row)
            if text:
                metadata_text[item_id] = text
                if len(metadata_text) >= len(wanted_items):
                    break

    return metadata_text


def metadata_to_text(row: dict) -> str:
    parts = [
        row.get("title") or "",
        row.get("main_category") or "",
        row.get("store") or "",
        " ".join(row.get("features") or []),
        " ".join(row.get("description") or []),
        " ".join(row.get("categories") or []),
    ]
    return " ".join(part for part in parts if part).strip()


def text_to_hash_embedding(text: str, dim: int, seed: int = 42) -> np.ndarray:
    vector = np.zeros(dim, dtype=np.float32)
    tokens = TOKEN_RE.findall(text.lower())
    if not tokens:
        return vector

    for token in tokens:
        digest = hashlib.blake2b(
            f"{seed}:{token}".encode("utf-8"),
            digest_size=8,
        ).digest()
        value = int.from_bytes(digest, byteorder="little", signed=False)
        index = value % dim
        sign = 1.0 if ((value >> 8) & 1) else -1.0
        vector[index] += sign

    norm = np.linalg.norm(vector)
    if norm > 0:
        vector /= norm
    return vector


def build_user_profiles_from_items(
    item_embeddings: np.ndarray,
    interactions: list[np.ndarray],
) -> np.ndarray:
    profiles = np.zeros((len(interactions), item_embeddings.shape[1]), dtype=np.float32)
    for user_idx, item_indices in enumerate(interactions):
        profiles[user_idx] = item_embeddings[item_indices].mean(axis=0)
        norm = np.linalg.norm(profiles[user_idx])
        if norm > 0:
            profiles[user_idx] /= norm
    return profiles
