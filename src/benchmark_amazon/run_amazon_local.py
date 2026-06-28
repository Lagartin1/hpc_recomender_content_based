import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.benchmark_amazon.amazon_dataset import load_amazon_electronics
from src.benchmark_utils.resource_metrics import empty_resource_metrics, measure_call
from src.recommenders.recommend_ray import top_k_recommendations_ray
from src.recommenders.recommend_torch import top_k_recommendations_torch
from src.recommenders.recommend_vectorized import top_k_recommendations


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pruebas locales crecientes con Amazon Reviews 2023 Electronics.",
    )
    parser.add_argument("--reviews", default="data/Electronics.jsonl.gz")
    parser.add_argument("--metadata", default="data/meta_Electronics.jsonl.gz")
    parser.add_argument(
        "--configs",
        default="5000:10000,10000:25000,20000:50000",
        help="Tamanos users:items separados por coma.",
    )
    parser.add_argument("--max-interactions", type=int, default=1000000)
    parser.add_argument("--min-rating", type=float, default=4.0)
    parser.add_argument("--min-user-interactions", type=int, default=2)
    parser.add_argument("--dim", type=int, default=384)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--block-size", type=int, default=1000)
    parser.add_argument(
        "--ray-max-in-flight",
        type=int,
        default=None,
        help="Maximo de bloques Ray simultaneos.",
    )
    parser.add_argument(
        "--backends",
        default="numpy,torch_gpu,ray_cpu,ray_cuda",
        help="Backends separados por coma: numpy,torch_gpu,ray_cpu,ray_cuda.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample-interval", type=float, default=0.05)
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--save-recommendations", action="store_true")
    parser.add_argument("--csv", default="results/amazon_local_benchmark.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = []
    backends = parse_backends(args.backends)

    for run_idx, (users, items) in enumerate(parse_configs(args.configs), start=1):
        rows.extend(run_one_config(args, run_idx, users, items, backends))

    csv_path = Path(args.csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_benchmark_csv(csv_path, rows)
    print(f"CSV benchmark guardado en: {csv_path}")


def run_one_config(
    args: argparse.Namespace,
    run_idx: int,
    users: int,
    items: int,
    backends: list[str],
) -> list[dict[str, object]]:
    print(f"\n[{run_idx}] Cargando Amazon Electronics: users={users}, items={items}...")
    load_start = time.perf_counter()
    dataset = load_amazon_electronics(
        reviews_path=Path(args.reviews),
        metadata_path=Path(args.metadata),
        max_users=users,
        max_items=items,
        max_interactions=args.max_interactions,
        dim=args.dim,
        min_rating=args.min_rating,
        min_user_interactions=args.min_user_interactions,
        seed=args.seed,
    )
    load_time = time.perf_counter() - load_start

    print(
        "Dataset listo: "
        f"{len(dataset.user_ids)} usuarios, "
        f"{len(dataset.item_ids)} items, "
        f"{sum(len(items) for items in dataset.interactions)} interacciones positivas."
    )

    rows = []
    for backend in backends:
        rows.append(run_one_backend(args, run_idx, users, items, dataset, load_time, backend))
    return rows


def run_one_backend(
    args: argparse.Namespace,
    run_idx: int,
    requested_users: int,
    requested_items: int,
    dataset,
    load_time: float,
    backend: str,
) -> dict[str, object]:
    print(f"Backend: {backend}")
    comparisons = len(dataset.user_ids) * len(dataset.item_ids)
    row = base_row(args, run_idx, requested_users, requested_items, dataset, load_time, backend)

    try:
        (top_items, top_scores), run_time, metrics = measure_call(
            lambda: run_backend(args, backend, dataset),
            interval_s=args.sample_interval,
        )

        output_path = ""
        if args.save_recommendations:
            output_path = str(
                Path(args.output_dir)
                / f"amazon_local_{backend}_u{len(dataset.user_ids)}_i{len(dataset.item_ids)}.npz"
            )
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                output_path,
                top_items=top_items,
                top_scores=top_scores,
                user_ids=np.array(dataset.user_ids),
                item_ids=np.array(dataset.item_ids),
            )

        throughput = comparisons / run_time
        row.update(
            {
                "status": "ok",
                "recommendation_time_s": run_time,
                "comparisons": comparisons,
                "throughput_comparisons_s": throughput,
                "output_npz": output_path,
                "error": "",
                **metrics,
            }
        )
        print(f"Tiempo recomendacion: {run_time:.3f} s")
        print(f"Throughput: {throughput:,.0f} comparaciones/s")
    except Exception as exc:
        row.update(
            {
                "status": "error",
                "recommendation_time_s": "",
                "comparisons": comparisons,
                "throughput_comparisons_s": "",
                "output_npz": "",
                "error": str(exc),
            }
        )
        print(f"Backend omitido por error: {exc}")

    return row


def run_backend(args: argparse.Namespace, backend: str, dataset) -> tuple[np.ndarray, np.ndarray]:
    if backend == "numpy":
        return top_k_recommendations(
            dataset.user_profiles,
            dataset.item_embeddings,
            dataset.interactions,
            k=args.k,
            block_size=args.block_size,
        )
    if backend == "torch_gpu":
        return top_k_recommendations_torch(
            dataset.user_profiles,
            dataset.item_embeddings,
            dataset.interactions,
            k=args.k,
            block_size=args.block_size,
            device="cuda",
        )
    if backend == "ray_cpu":
        return top_k_recommendations_ray(
            dataset.user_profiles,
            dataset.item_embeddings,
            dataset.interactions,
            k=args.k,
            block_size=args.block_size,
            max_in_flight=args.ray_max_in_flight,
            device="cpu",
        )
    if backend == "ray_cuda":
        return top_k_recommendations_ray(
            dataset.user_profiles,
            dataset.item_embeddings,
            dataset.interactions,
            k=args.k,
            block_size=args.block_size,
            max_in_flight=args.ray_max_in_flight,
            device="cuda",
        )
    raise ValueError(f"Backend no soportado: {backend}")


def base_row(
    args: argparse.Namespace,
    run_idx: int,
    requested_users: int,
    requested_items: int,
    dataset,
    load_time: float,
    backend: str,
) -> dict[str, object]:
    row = {
        "environment": "local",
        "backend": backend,
        "run": run_idx,
        "requested_users": requested_users,
        "requested_items": requested_items,
        "users": len(dataset.user_ids),
        "items": len(dataset.item_ids),
        "positive_interactions": sum(len(item_ids) for item_ids in dataset.interactions),
        "dim": args.dim,
        "k": args.k,
        "block_size": args.block_size,
        "ray_max_in_flight": args.ray_max_in_flight or "",
        "max_interactions": args.max_interactions,
        "min_rating": args.min_rating,
        "min_user_interactions": args.min_user_interactions,
        "load_time_s": load_time,
    }
    row.update(empty_resource_metrics())
    return row


def parse_configs(configs: str) -> list[tuple[int, int]]:
    parsed = []
    for raw_config in configs.split(","):
        users, items = raw_config.strip().split(":", maxsplit=1)
        parsed.append((int(users), int(items)))
    return parsed


def parse_backends(backends: str) -> list[str]:
    valid = {"numpy", "torch_gpu", "ray_cpu", "ray_cuda"}
    parsed = [backend.strip() for backend in backends.split(",") if backend.strip()]
    unknown = sorted(set(parsed) - valid)
    if unknown:
        raise ValueError(f"Backends no soportados: {', '.join(unknown)}")
    return parsed


def write_benchmark_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError("No hay resultados para escribir.")
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
