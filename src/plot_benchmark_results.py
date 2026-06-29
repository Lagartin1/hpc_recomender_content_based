"""Generate benchmark metric plots from result CSV files.

The script reads one or more benchmark CSVs and creates:

- a multi-page PDF with one plot per available metric
- one PNG per metric, useful for inserting figures into a report
"""

from __future__ import annotations

import argparse
import csv
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import EngFormatter


TITLE_FONT_SIZE = 22
AXIS_LABEL_FONT_SIZE = 18
TICK_LABEL_FONT_SIZE = 15
LEGEND_FONT_SIZE = 15

DEFAULT_INPUTS = [
    "results/amazon_local_cpu_benchmark.csv",
    "results/amazon_local_cuda_benchmark.csv",
    "results/amazon_patagon_numpy_resume_benchmark.csv",
    "results/amazon_patagon_ray_cpu_resume_benchmark.csv",
    "results/amazon_patagon_cuda_resume_benchmark.csv",
]

DEFAULT_EXCLUDED_SCENARIOS = {
    ("patagon", "10000", "50000"),
}

DEFAULT_METRICS = [
    "recommendation_time_s",
    "throughput_comparisons_s",
    "speedup",
    "spedup",
    "speedup_throughput",
    "speedup_time",
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
]

METRIC_LABELS = {
    "recommendation_time_s": "Tiempo de recomendacion (s)",
    "throughput_comparisons_s": "Throughput (comparaciones/s)",
    "speedup": "Speedup",
    "spedup": "Speedup",
    "speedup_x": "Speedup",
    "speedup_vs_numpy": "Speedup vs NumPy",
    "speedup_throughput": "Speedup por throughput vs NumPy",
    "speedup_time": "Speedup por tiempo vs NumPy",
    "cpu_avg_%": "CPU promedio (%)",
    "cpu_peak_%": "CPU maximo (%)",
    "ram_peak_MB": "RAM maxima (MB)",
    "vram_peak_MB": "VRAM maxima (MB)",
    "cpu_power_W": "Potencia CPU promedio (W)",
    "gpu_power_W": "Potencia GPU promedio (W)",
    "total_power_W": "Potencia total promedio (W)",
    "cpu_energy_Wh": "Energia CPU (Wh)",
    "gpu_energy_Wh": "Energia GPU (Wh)",
    "total_energy_Wh": "Energia total (Wh)",
    "codecarbon_energy_kWh": "Energia CodeCarbon (kWh)",
    "codecarbon_emissions_kg": "Emisiones CodeCarbon (kg CO2eq)",
    "codecarbon_cpu_energy_kWh": "Energia CPU CodeCarbon (kWh)",
    "codecarbon_gpu_energy_kWh": "Energia GPU CodeCarbon (kWh)",
    "codecarbon_ram_energy_kWh": "Energia RAM CodeCarbon (kWh)",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Crea graficos por metrica desde CSVs de benchmark."
    )
    parser.add_argument(
        "--input",
        nargs="+",
        default=DEFAULT_INPUTS,
        help=(
            "Archivos CSV o patrones glob. Por defecto usa los CSV locales y "
            "los CSV resume de Patagon si existen."
        ),
    )
    parser.add_argument(
        "--output-pdf",
        default="results/benchmark_metrics.pdf",
        help="Ruta del PDF multipagina de salida.",
    )
    parser.add_argument(
        "--output-dir",
        default="results/plots",
        help="Directorio para guardar un PNG por metrica.",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=DEFAULT_METRICS,
        help="Metricas a graficar. Por defecto usa las metricas del trabajo.",
    )
    parser.add_argument(
        "--include-errors",
        action="store_true",
        help="Incluye filas cuyo status no sea ok si tienen valores numericos.",
    )
    parser.add_argument(
        "--title-prefix",
        default="Benchmark recomendador content-based",
        help="Prefijo para los titulos de los graficos.",
    )
    parser.add_argument(
        "--series",
        choices=["backend", "environment_backend", "source_backend"],
        default="backend",
        help="Como agrupar las lineas. Por defecto: backend para mostrar los 4 metodos.",
    )
    return parser.parse_args()


def expand_inputs(patterns: Iterable[str]) -> list[Path]:
    paths: list[Path] = []
    for pattern in patterns:
        path = Path(pattern)
        if path.is_file():
            paths.append(path)
            continue

        matched = sorted(Path().glob(pattern))
        if matched:
            paths.extend(path for path in matched if path.is_file())

    unique_paths = sorted(set(paths))
    if not unique_paths:
        raise FileNotFoundError(
            "No se encontraron CSVs. Usa --input results/archivo.csv o un patron glob."
        )
    return unique_paths


def parse_float(value: str | None) -> float | None:
    if value is None:
        return None

    text = value.strip()
    if not text or text.upper() == "N/A":
        return None

    try:
        number = float(text)
    except ValueError:
        return None

    if math.isnan(number) or math.isinf(number):
        return None
    return number


def load_rows(paths: Iterable[Path], include_errors: bool) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    excluded_rows = 0
    for path in paths:
        with path.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                if not include_errors and row.get("status", "").lower() != "ok":
                    continue
                if is_excluded_scenario(row):
                    excluded_rows += 1
                    continue
                row["source_file"] = path.name
                rows.append(row)

    if not rows:
        raise ValueError("No hay filas validas para graficar despues de aplicar filtros.")
    if excluded_rows:
        print(f"[filter] filas excluidas por escenario: {excluded_rows}")
    add_derived_speedups(rows)
    return rows


def is_excluded_scenario(row: dict[str, str]) -> bool:
    key = (
        row.get("environment", "").strip().lower(),
        normalize_count(row.get("requested_users") or row.get("users")),
        normalize_count(row.get("requested_items") or row.get("items")),
    )
    return key in DEFAULT_EXCLUDED_SCENARIOS


def normalize_count(value: str | None) -> str:
    number = parse_float(value)
    if number is None:
        return (value or "").strip()
    return f"{number:g}"


def add_derived_speedups(rows: list[dict[str, str]]) -> None:
    baselines: dict[tuple[str, str], dict[str, float]] = {}

    for row in rows:
        if row.get("backend", "").strip() != "numpy":
            continue

        key = baseline_key(row)
        recommendation_time = parse_float(row.get("recommendation_time_s"))
        throughput = parse_float(row.get("throughput_comparisons_s"))
        baseline = baselines.setdefault(key, {})
        if recommendation_time and recommendation_time > 0:
            baseline["recommendation_time_s"] = recommendation_time
        if throughput and throughput > 0:
            baseline["throughput_comparisons_s"] = throughput

    for row in rows:
        baseline = baselines.get(baseline_key(row), {})

        throughput = parse_float(row.get("throughput_comparisons_s"))
        baseline_throughput = baseline.get("throughput_comparisons_s")
        if throughput and baseline_throughput and baseline_throughput > 0:
            row.setdefault("speedup_throughput", str(throughput / baseline_throughput))

        recommendation_time = parse_float(row.get("recommendation_time_s"))
        baseline_time = baseline.get("recommendation_time_s")
        if recommendation_time and baseline_time and recommendation_time > 0:
            row.setdefault("speedup_time", str(baseline_time / recommendation_time))


def baseline_key(row: dict[str, str]) -> tuple[str, str]:
    return (row.get("environment", "").strip(), scenario_label(row))


def scenario_label(row: dict[str, str]) -> str:
    environment = row.get("environment", "").strip()
    requested_users = row.get("requested_users") or row.get("users") or "?"
    requested_items = row.get("requested_items") or row.get("items") or "?"

    requested_label = f"{compact_count(requested_users)} U x {compact_count(requested_items)} I"
    prefix = f"{environment.title()}\n" if environment else ""
    return f"{prefix}{requested_label}"


def series_label(row: dict[str, str], mode: str) -> str:
    environment = row.get("environment", "").strip()
    backend = row.get("backend", "").strip()
    source_file = row.get("source_file", "serie")

    if mode == "backend":
        return backend or environment or source_file
    if mode == "source_backend":
        return f"{source_file} / {backend}" if backend else source_file

    if environment and backend:
        return f"{environment} / {backend}"
    return backend or environment or source_file


def scenario_sort_key(label: str) -> tuple[int, int, str]:
    match = re.search(r"([\d.]+)([KMB]?) U x ([\d.]+)([KMB]?) I", label)
    if not match:
        return (0, 0, label)
    users = expand_compact_count(match.group(1), match.group(2))
    items = expand_compact_count(match.group(3), match.group(4))
    return (users, items, label)


def compact_count(value: str) -> str:
    number = parse_float(value)
    if number is None:
        return value

    if number >= 1_000_000_000:
        return f"{number / 1_000_000_000:g}B"
    if number >= 1_000_000:
        return f"{number / 1_000_000:g}M"
    if number >= 1_000:
        return f"{number / 1_000:g}K"
    return f"{number:g}"


def expand_compact_count(value: str, suffix: str) -> int:
    multiplier = {"": 1, "K": 1_000, "M": 1_000_000, "B": 1_000_000_000}[suffix]
    return int(float(value) * multiplier)


def filename_for_metric(metric: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]+", "_", metric).strip("_").lower()


def metric_value(row: dict[str, str], metric: str) -> str | None:
    if metric in row:
        return row.get(metric)

    normalized_metric = metric.lower()
    for key, value in row.items():
        if key.lower() == normalized_metric:
            return value
    return None


def collect_metric_points(
    rows: Iterable[dict[str, str]], metric: str, series_mode: str
) -> dict[str, dict[str, float]]:
    grouped: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))

    for row in rows:
        value = parse_float(metric_value(row, metric))
        if value is None:
            continue
        grouped[series_label(row, series_mode)][scenario_label(row)].append(value)

    return {
        series: {scenario: sum(values) / len(values) for scenario, values in scenarios.items()}
        for series, scenarios in grouped.items()
    }


def plot_metric(
    metric: str,
    points_by_series: dict[str, dict[str, float]],
    title_prefix: str,
) -> plt.Figure:
    scenarios = sorted(
        {scenario for points in points_by_series.values() for scenario in points},
        key=scenario_sort_key,
    )
    x_positions = list(range(len(scenarios)))

    fig, ax = plt.subplots(figsize=(13.5, 7.0))
    for series in sorted(points_by_series):
        values = [points_by_series[series].get(scenario) for scenario in scenarios]
        x = [pos for pos, value in zip(x_positions, values) if value is not None]
        y = [value for value in values if value is not None]
        ax.plot(x, y, marker="o", markersize=7, linewidth=3, label=series)

    metric_label = METRIC_LABELS.get(metric, metric)
    ax.set_title(
        f"{title_prefix}: {metric_label}",
        fontsize=TITLE_FONT_SIZE,
        fontweight="bold",
        pad=14,
    )
    ax.set_xlabel(
        "Escenario de prueba (entorno y tamano solicitado)",
        fontsize=AXIS_LABEL_FONT_SIZE,
    )
    ax.set_ylabel(metric_label, fontsize=AXIS_LABEL_FONT_SIZE)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(
        scenarios,
        rotation=20,
        ha="right",
        fontsize=TICK_LABEL_FONT_SIZE,
    )
    ax.tick_params(axis="y", labelsize=TICK_LABEL_FONT_SIZE)
    ax.yaxis.set_major_formatter(EngFormatter())
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    ax.legend(
        loc="upper left",
        ncol=1,
        fontsize=LEGEND_FONT_SIZE,
        frameon=True,
        framealpha=0.88,
        facecolor="white",
        edgecolor="#d0d0d0",
    )
    fig.subplots_adjust(left=0.10, right=0.98, top=0.86, bottom=0.22)
    return fig


def main() -> int:
    args = parse_args()
    csv_paths = expand_inputs(args.input)
    rows = load_rows(csv_paths, args.include_errors)

    output_pdf = Path(args.output_pdf)
    output_dir = Path(args.output_dir)
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    created_metrics: list[str] = []
    with PdfPages(output_pdf) as pdf:
        for metric in args.metrics:
            points = collect_metric_points(rows, metric, args.series)
            if not points:
                print(f"[skip] {metric}: sin valores numericos")
                continue

            fig = plot_metric(metric, points, args.title_prefix)
            png_path = output_dir / f"{filename_for_metric(metric)}.png"
            fig.savefig(png_path, dpi=200)
            pdf.savefig(fig)
            plt.close(fig)
            created_metrics.append(metric)
            print(f"[ok] {metric}: {png_path}")

    if not created_metrics:
        output_pdf.unlink(missing_ok=True)
        raise ValueError("No se genero ningun grafico; revisa las metricas solicitadas.")

    print(f"PDF generado: {output_pdf}")
    print(f"Metricas graficadas: {len(created_metrics)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
