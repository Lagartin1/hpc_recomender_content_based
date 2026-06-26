# HPC recommender content based

Este proyecto implementa y compara un recomendador basado en contenido para
escenarios de computo de alto rendimiento. El sistema representa usuarios e
items como vectores de embeddings, construye perfiles de usuario a partir de
los items con los que ya interactuaron y recomienda los `k` items mas similares
usando similitud coseno.

La idea principal es evaluar distintas formas de ejecutar el mismo calculo:

- Una version vectorizada en CPU con NumPy.
- Una version distribuida por bloques con Ray.
- Una version con PyTorch, capaz de usar CPU o CUDA cuando hay GPU disponible.

El benchmark genera datos sinteticos, ejecuta las variantes seleccionadas y
reporta tiempo, uso de CPU/RAM, uso de VRAM y consumo energetico cuando el
sistema expone esos contadores. Esto permite comparar rendimiento y costo de
recursos entre CPU, GPU y ejecucion distribuida.

## Benchmarks

Instala dependencias:

```bash
python3 -m venv venv
venv/bin/pip install -r requirements.txt
```

Ejecuta ambas versiones con datos sinteticos:

```bash
venv/bin/python src/benchmark.py
```

Ejemplo con tamanos personalizados y salida CSV:

```bash
venv/bin/python src/benchmark.py \
  --users 5000 \
  --items 20000 \
  --dim 128 \
  --k 10 \
  --runs 5 \
  --warmup 1 \
  --block-size 1000 \
  --device auto \
  --csv benchmark_results.csv
```

Comparar solo CPU vectorizado contra Ray CPU:

```bash
venv/bin/python src/benchmark.py --versions vectorized ray --device cpu
```

Comparar CPU vectorizado contra PyTorch vectorizado en GPU:

```bash
venv/bin/python src/benchmark.py --versions vectorized vectorized_gpu --device cuda
```

Forzar Ray con CUDA:

```bash
venv/bin/python src/benchmark.py --versions ray --device cuda
```

Metricas reportadas:

- `time_s`: tiempo total de ejecucion.
- `cpu_avg_%` y `cpu_peak_%`: uso de CPU del proceso y procesos hijos.
- `ram_peak_MB`: maximo RSS observado del proceso y procesos hijos.
- `vram_peak_MB`: memoria GPU observada con `nvidia-smi`, si existe.
- `cpu_power_W`: potencia promedio CPU, calculada desde RAPL como joules / segundos.
- `gpu_power_W`: potencia promedio GPU reportada por `power.draw` desde `nvidia-smi`.
- `total_power_W`: suma de potencia CPU/GPU disponible.
- `cpu_energy_Wh`: energia CPU consumida en watt-hora.
- `gpu_energy_Wh`: energia GPU consumida en watt-hora.
- `total_energy_Wh`: suma de energia CPU/GPU disponible en watt-hora.

Cuando una metrica sale como `N/A`, el sistema no expuso ese contador o falta la herramienta necesaria.
