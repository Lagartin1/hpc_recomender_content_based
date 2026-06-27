import argparse
import csv
import glob
import os
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.benchmark_syntetic.create_syntetic import (
    build_user_profiles,
    generate_items_embeddings,
    generate_user_interactions,
)
from src.benchmark_utils.resource_metrics import start_codecarbon_tracker, stop_codecarbon_tracker
from src.recommenders.recommend_ray import top_k_recommendations_ray
from src.recommenders.recommend_torch import top_k_recommendations_torch
from src.recommenders.recommend_vectorized import top_k_recommendations

try:
    import psutil
except ImportError:  # pragma: no cover - dependencia opcional
    psutil = None


@dataclass
class BenchmarkResult:
    version: str
    run: int
    users: int
    items: int
    dim: int
    k: int
    block_size: int | None
    device: str
    time_s: float
    cpu_percent_avg: float | None
    cpu_percent_peak: float | None
    ram_rss_peak_mb: float | None
    vram_peak_mb: float | None
    cpu_power_w: float | None
    gpu_power_w: float | None
    total_power_w: float | None
    cpu_energy_wh: float | None
    gpu_energy_wh: float | None
    total_energy_wh: float | None
    codecarbon_energy_kWh: float | str
    codecarbon_emissions_kg: float | str
    codecarbon_cpu_energy_kWh: float | str
    codecarbon_gpu_energy_kWh: float | str
    codecarbon_ram_energy_kWh: float | str
    notes: str


class ResourceSampler:
    def __init__(self, interval_s: float = 0.05):
        self.interval_s = interval_s
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.cpu_samples: list[float] = []
        self.rss_samples_mb: list[float] = []
        self.vram_samples_mb: list[float] = []
        self.gpu_power_samples_w: list[tuple[float, float]] = []
        self._process = psutil.Process(os.getpid()) if psutil else None
        self._has_nvidia_smi = shutil.which("nvidia-smi") is not None

    def __enter__(self):
        if self._process:
            self._process.cpu_percent(interval=None)
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self._stop.set()
        if self._thread:
            self._thread.join()

    def _sample_loop(self):
        while not self._stop.is_set():
            timestamp = time.perf_counter()

            if self._process:
                try:
                    self.cpu_samples.append(self._process.cpu_percent(interval=None))
                    rss = self._process.memory_info().rss / (1024**2)
                    for child in self._process.children(recursive=True):
                        try:
                            rss += child.memory_info().rss / (1024**2)
                        except psutil.Error:
                            pass
                    self.rss_samples_mb.append(rss)
                except psutil.Error:
                    pass

            gpu = _query_nvidia_smi()
            if gpu:
                vram_mb, power_w = gpu
                self.vram_samples_mb.append(vram_mb)
                if power_w is not None:
                    self.gpu_power_samples_w.append((timestamp, power_w))

            self._stop.wait(self.interval_s)

    @property
    def cpu_percent_avg(self) -> float | None:
        return _mean(self.cpu_samples)

    @property
    def cpu_percent_peak(self) -> float | None:
        return max(self.cpu_samples) if self.cpu_samples else None

    @property
    def ram_rss_peak_mb(self) -> float | None:
        return max(self.rss_samples_mb) if self.rss_samples_mb else None

    @property
    def vram_peak_mb(self) -> float | None:
        return max(self.vram_samples_mb) if self.vram_samples_mb else None

    @property
    def gpu_energy_j(self) -> float | None:
        samples = self.gpu_power_samples_w
        if len(samples) < 2:
            return None

        energy = 0.0
        for (t0, p0), (t1, p1) in zip(samples, samples[1:]):
            energy += ((p0 + p1) / 2.0) * (t1 - t0)
        return energy

    @property
    def gpu_power_avg_w(self) -> float | None:
        return _mean([power_w for _, power_w in self.gpu_power_samples_w])


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _query_nvidia_smi() -> tuple[float, float | None] | None:
    if shutil.which("nvidia-smi") is None:
        return None

    query = "memory.used,power.draw"
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                f"--query-gpu={query}",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=1,
        )
    except (subprocess.SubprocessError, OSError):
        return None

    total_vram = 0.0
    total_power = 0.0
    power_count = 0
    for line in completed.stdout.strip().splitlines():
        parts = [part.strip() for part in line.split(",")]
        if not parts or not parts[0]:
            continue
        total_vram += float(parts[0])
        if len(parts) > 1 and parts[1] and parts[1] != "[N/A]":
            total_power += float(parts[1])
            power_count += 1

    if total_vram == 0.0 and power_count == 0:
        return None
    return total_vram, total_power if power_count else None


def _read_rapl_energy_uj() -> int | None:
    total = 0
    files = []
    for path in glob.glob("/sys/class/powercap/*-rapl:*/energy_uj"):
        zone_name = os.path.basename(os.path.dirname(path))
        if zone_name.count(":") == 1:
            files.append(path)

    for path in files:
        try:
            with open(path, encoding="utf-8") as file:
                total += int(file.read().strip())
        except OSError:
            continue
    return total if files else None


def _energy_delta_j(start_uj: int | None, end_uj: int | None) -> float | None:
    if start_uj is None or end_uj is None:
        return None
    if end_uj < start_uj:
        return None
    return (end_uj - start_uj) / 1_000_000.0


def _joules_to_wh(value_j: float | None) -> float | None:
    return value_j / 3600.0 if value_j is not None else None


def _run_one(
    name: str,
    fn: Callable[[], tuple[np.ndarray, np.ndarray]],
    run_idx: int,
    args: argparse.Namespace,
    block_size: int | None,
    device: str,
) -> BenchmarkResult:
    notes = []
    if psutil is None:
        notes.append("instala psutil para CPU/RAM")
    if shutil.which("nvidia-smi") is None:
        notes.append("sin nvidia-smi para VRAM/potencia GPU")

    cpu_energy_start = _read_rapl_energy_uj()
    if cpu_energy_start is None:
        notes.append("sin RAPL para potencia CPU")

    codecarbon_tracker = start_codecarbon_tracker(args.sample_interval, notes)
    with ResourceSampler(args.sample_interval) as sampler:
        start = time.perf_counter()
        try:
            fn()
        finally:
            elapsed = time.perf_counter() - start
            codecarbon_metrics = stop_codecarbon_tracker(codecarbon_tracker, notes)

    cpu_energy_j = _energy_delta_j(cpu_energy_start, _read_rapl_energy_uj())
    gpu_energy_j = sampler.gpu_energy_j
    cpu_power_w = cpu_energy_j / elapsed if cpu_energy_j is not None and elapsed > 0 else None
    gpu_power_w = sampler.gpu_power_avg_w
    measured = [value for value in (cpu_power_w, gpu_power_w) if value is not None]
    total_power_w = sum(measured) if measured else None
    measured_energy = [value for value in (cpu_energy_j, gpu_energy_j) if value is not None]
    total_energy_j = sum(measured_energy) if measured_energy else None

    return BenchmarkResult(
        version=name,
        run=run_idx,
        users=args.users,
        items=args.items,
        dim=args.dim,
        k=args.k,
        block_size=block_size,
        device=device,
        time_s=elapsed,
        cpu_percent_avg=sampler.cpu_percent_avg,
        cpu_percent_peak=sampler.cpu_percent_peak,
        ram_rss_peak_mb=sampler.ram_rss_peak_mb,
        vram_peak_mb=sampler.vram_peak_mb,
        cpu_power_w=cpu_power_w,
        gpu_power_w=gpu_power_w,
        total_power_w=total_power_w,
        cpu_energy_wh=_joules_to_wh(cpu_energy_j),
        gpu_energy_wh=_joules_to_wh(gpu_energy_j),
        total_energy_wh=_joules_to_wh(total_energy_j),
        codecarbon_energy_kWh=codecarbon_metrics["codecarbon_energy_kWh"],
        codecarbon_emissions_kg=codecarbon_metrics["codecarbon_emissions_kg"],
        codecarbon_cpu_energy_kWh=codecarbon_metrics["codecarbon_cpu_energy_kWh"],
        codecarbon_gpu_energy_kWh=codecarbon_metrics["codecarbon_gpu_energy_kWh"],
        codecarbon_ram_energy_kWh=codecarbon_metrics["codecarbon_ram_energy_kWh"],
        notes="; ".join(dict.fromkeys(notes)),
    )


def _format(value: object, digits: int = 3) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def print_results(results: list[BenchmarkResult]) -> None:
    headers = [
        "version",
        "run",
        "time_s",
        "cpu_avg_%",
        "cpu_peak_%",
        "ram_peak_MB",
        "vram_peak_MB",
        "cpu_power_W",
        "gpu_power_W",
        "total_power_W",
        "cpu_energy_Wh",
        "gpu_energy_Wh",
        "total_energy_Wh",
        "codecarbon_energy_kWh",
        "codecarbon_emissions_kg",
        "codecarbon_cpu_energy_kWh",
        "codecarbon_gpu_energy_kWh",
        "codecarbon_ram_energy_kWh",
        "notes",
    ]
    rows = [
        [
            result.version,
            result.run,
            _format(result.time_s),
            _format(result.cpu_percent_avg),
            _format(result.cpu_percent_peak),
            _format(result.ram_rss_peak_mb),
            _format(result.vram_peak_mb),
            _format(result.cpu_power_w),
            _format(result.gpu_power_w),
            _format(result.total_power_w),
            _format(result.cpu_energy_wh, digits=6),
            _format(result.gpu_energy_wh, digits=6),
            _format(result.total_energy_wh, digits=6),
            _format(result.codecarbon_energy_kWh, digits=6),
            _format(result.codecarbon_emissions_kg, digits=6),
            _format(result.codecarbon_cpu_energy_kWh, digits=6),
            _format(result.codecarbon_gpu_energy_kWh, digits=6),
            _format(result.codecarbon_ram_energy_kWh, digits=6),
            result.notes,
        ]
        for result in results
    ]
    widths = [
        max(len(str(row[idx])) for row in [headers, *rows])
        for idx in range(len(headers))
    ]

    print(" | ".join(value.ljust(widths[idx]) for idx, value in enumerate(headers)))
    print("-+-".join("-" * width for width in widths))
    for row in rows:
        print(" | ".join(str(value).ljust(widths[idx]) for idx, value in enumerate(row)))


def write_csv(path: str, results: list[BenchmarkResult]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(asdict(results[0]).keys()))
        writer.writeheader()
        writer.writerows(asdict(result) for result in results)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark de recomendacion: tiempo, CPU, RAM, VRAM y potencia electrica.",
    )
    parser.add_argument("--users", type=int, default=1000)
    parser.add_argument("--items", type=int, default=5000)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--max-interactions", type=int, default=50)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--block-size", type=int, default=1000)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument(
        "--versions",
        nargs="+",
        choices=["numpy", "torch_gpu", "ray"],
        default=["numpy", "torch_gpu", "ray"],
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample-interval", type=float, default=0.05)
    parser.add_argument("--csv", help="Ruta opcional para guardar resultados CSV.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if "torch_gpu" in args.versions and args.device == "cpu":
        raise ValueError("torch_gpu requiere CUDA. Usa --versions numpy para CPU.")

    item_embeddings = generate_items_embeddings(args.items, args.dim, seed=args.seed)
    interactions = generate_user_interactions(
        args.users,
        args.items,
        args.max_interactions,
        seed=args.seed + 1,
    )
    user_profiles = build_user_profiles(None, item_embeddings, interactions)

    benchmarks: list[tuple[str, Callable[[], tuple[np.ndarray, np.ndarray]], int | None, str]] = []
    if "numpy" in args.versions:
        benchmarks.append(
            (
                "numpy",
                lambda: top_k_recommendations(
                    user_profiles,
                    item_embeddings,
                    interactions,
                    args.k,
                ),
                None,
                "cpu",
            )
        )
    if "torch_gpu" in args.versions:
        benchmarks.append(
            (
                "torch_gpu",
                lambda: top_k_recommendations_torch(
                    user_profiles,
                    item_embeddings,
                    interactions,
                    args.k,
                    block_size=args.block_size,
                    device=args.device,
                ),
                args.block_size,
                "cuda" if args.device == "auto" else args.device,
            )
        )
    if "ray" in args.versions:
        benchmarks.append(
            (
                "ray",
                lambda: top_k_recommendations_ray(
                    user_profiles,
                    item_embeddings,
                    interactions,
                    args.k,
                    block_size=args.block_size,
                    device=args.device,
                ),
                args.block_size,
                args.device,
            )
        )

    for _, fn, _, _ in benchmarks:
        for _ in range(args.warmup):
            fn()

    results = []
    for name, fn, block_size, device in benchmarks:
        for run_idx in range(1, args.runs + 1):
            results.append(_run_one(name, fn, run_idx, args, block_size, device))

    print_results(results)
    if args.csv:
        write_csv(args.csv, results)
        print(f"\nCSV guardado en: {args.csv}")

    if "ray" in args.versions:
        import ray

        if ray.is_initialized():
            ray.shutdown()


if __name__ == "__main__":
    main()
