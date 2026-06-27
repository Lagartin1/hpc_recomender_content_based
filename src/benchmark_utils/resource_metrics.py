import glob
import os
import shutil
import subprocess
import threading
import time
from typing import Callable, TypeVar

try:
    import psutil
except ImportError:  # pragma: no cover - dependencia opcional
    psutil = None

try:
    from codecarbon import EmissionsTracker
except ImportError:  # pragma: no cover - dependencia opcional
    EmissionsTracker = None


T = TypeVar("T")


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

    def __enter__(self) -> "ResourceSampler":
        if self._process:
            self._process.cpu_percent(interval=None)
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join()

    def _sample_loop(self) -> None:
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

            gpu = query_nvidia_smi()
            if gpu:
                vram_mb, power_w = gpu
                self.vram_samples_mb.append(vram_mb)
                if power_w is not None:
                    self.gpu_power_samples_w.append((timestamp, power_w))

            self._stop.wait(self.interval_s)

    @property
    def cpu_percent_avg(self) -> float | None:
        return mean(self.cpu_samples)

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
        return mean([power_w for _, power_w in self.gpu_power_samples_w])


def measure_call(fn: Callable[[], T], interval_s: float = 0.05) -> tuple[T, float, dict[str, object]]:
    notes = []
    if psutil is None:
        notes.append("instala psutil para CPU/RAM")
    if shutil.which("nvidia-smi") is None:
        notes.append("sin nvidia-smi para VRAM/potencia GPU")
    if EmissionsTracker is None:
        notes.append("instala codecarbon para estimacion energetica")

    cpu_energy_start = read_rapl_energy_uj()
    if cpu_energy_start is None:
        notes.append("sin RAPL para potencia CPU")

    tracker = start_codecarbon_tracker(interval_s, notes)
    with ResourceSampler(interval_s) as sampler:
        start = time.perf_counter()
        try:
            result = fn()
        finally:
            elapsed = time.perf_counter() - start
            codecarbon_metrics = stop_codecarbon_tracker(tracker, notes)

    cpu_energy_j = energy_delta_j(cpu_energy_start, read_rapl_energy_uj())
    gpu_energy_j = sampler.gpu_energy_j
    cpu_power_w = cpu_energy_j / elapsed if cpu_energy_j is not None and elapsed > 0 else None
    gpu_power_w = sampler.gpu_power_avg_w
    measured_power = [value for value in (cpu_power_w, gpu_power_w) if value is not None]
    measured_energy = [value for value in (cpu_energy_j, gpu_energy_j) if value is not None]
    total_energy_j = sum(measured_energy) if measured_energy else None

    metrics = {
        "cpu_avg_%": sampler.cpu_percent_avg,
        "cpu_peak_%": sampler.cpu_percent_peak,
        "ram_peak_MB": sampler.ram_rss_peak_mb,
        "vram_peak_MB": sampler.vram_peak_mb,
        "cpu_power_W": cpu_power_w,
        "gpu_power_W": gpu_power_w,
        "total_power_W": sum(measured_power) if measured_power else None,
        "cpu_energy_Wh": joules_to_wh(cpu_energy_j),
        "gpu_energy_Wh": joules_to_wh(gpu_energy_j),
        "total_energy_Wh": joules_to_wh(total_energy_j),
        "notes": "; ".join(dict.fromkeys(notes)),
    }
    metrics.update(codecarbon_metrics)
    return result, elapsed, metrics


def empty_resource_metrics() -> dict[str, object]:
    return {
        "cpu_avg_%": "",
        "cpu_peak_%": "",
        "ram_peak_MB": "",
        "vram_peak_MB": "",
        "cpu_power_W": "",
        "gpu_power_W": "",
        "total_power_W": "",
        "cpu_energy_Wh": "",
        "gpu_energy_Wh": "",
        "total_energy_Wh": "",
        "codecarbon_energy_kWh": "",
        "codecarbon_emissions_kg": "",
        "codecarbon_cpu_energy_kWh": "",
        "codecarbon_gpu_energy_kWh": "",
        "codecarbon_ram_energy_kWh": "",
        "notes": "",
    }


def empty_codecarbon_metrics() -> dict[str, object]:
    return {
        "codecarbon_energy_kWh": "",
        "codecarbon_emissions_kg": "",
        "codecarbon_cpu_energy_kWh": "",
        "codecarbon_gpu_energy_kWh": "",
        "codecarbon_ram_energy_kWh": "",
    }


def start_codecarbon_tracker(interval_s: float, notes: list[str]):
    if EmissionsTracker is None:
        return None

    kwargs = {
        "measure_power_secs": interval_s,
        "save_to_file": False,
        "log_level": "error",
    }
    try:
        tracker = EmissionsTracker(**kwargs)
    except TypeError:
        tracker = EmissionsTracker(save_to_file=False)

    try:
        tracker.start()
    except Exception as exc:  # pragma: no cover - depende del host
        notes.append(f"CodeCarbon no disponible: {exc}")
        return None
    return tracker


def stop_codecarbon_tracker(tracker, notes: list[str]) -> dict[str, object]:
    metrics = empty_codecarbon_metrics()
    if tracker is None:
        return metrics

    try:
        emissions_kg = tracker.stop()
    except Exception as exc:  # pragma: no cover - depende del host
        notes.append(f"CodeCarbon fallo al detenerse: {exc}")
        return metrics

    data = getattr(tracker, "final_emissions_data", None)
    metrics["codecarbon_emissions_kg"] = emissions_kg
    metrics["codecarbon_energy_kWh"] = emission_value(data, "energy_consumed")
    metrics["codecarbon_cpu_energy_kWh"] = emission_value(data, "cpu_energy")
    metrics["codecarbon_gpu_energy_kWh"] = emission_value(data, "gpu_energy")
    metrics["codecarbon_ram_energy_kWh"] = emission_value(data, "ram_energy")
    return metrics


def emission_value(data, name: str):
    if data is None:
        return ""
    if isinstance(data, dict):
        return data.get(name, "")
    return getattr(data, name, "")


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def query_nvidia_smi() -> tuple[float, float | None] | None:
    if shutil.which("nvidia-smi") is None:
        return None

    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.used,power.draw",
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


def read_rapl_energy_uj() -> int | None:
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


def energy_delta_j(start_uj: int | None, end_uj: int | None) -> float | None:
    if start_uj is None or end_uj is None:
        return None
    if end_uj < start_uj:
        return None
    return (end_uj - start_uj) / 1_000_000.0


def joules_to_wh(value_j: float | None) -> float | None:
    return value_j / 3600.0 if value_j is not None else None
