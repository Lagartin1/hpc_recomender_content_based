# HPC recommender content based

Este proyecto implementa y compara un recomendador basado en contenido para
escenarios de computo de alto rendimiento. El sistema representa usuarios e
items como vectores de embeddings, construye perfiles de usuario a partir de los
items con los que ya interactuaron y recomienda los `k` items mas similares
usando similitud coseno.

La comparacion se enfoca en cuatro variantes:

- CPU vectorizado con NumPy.
- CPU distribuido por bloques con Ray.
- GPU CUDA con PyTorch.
- GPU CUDA distribuido con Ray.

Los benchmarks generan o cargan datos, ejecutan las variantes seleccionadas y
reportan tiempo, uso de CPU/RAM, uso de VRAM y consumo energetico cuando Ubuntu
expone esos contadores.

## Requisitos en Ubuntu

Requisitos base:

```bash
sudo apt update
sudo apt install -y python3 python3-venv python3-pip git wget
```

Para ejecutar con GPU se necesita:

- Driver NVIDIA instalado en Ubuntu.
- CUDA visible para PyTorch.
- `nvidia-smi` disponible en la terminal.

Comprueba la GPU:

```bash
nvidia-smi
```

## Instalacion

Crea el entorno virtual e instala las dependencias:

```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

`requirements.txt` instala PyTorch con CUDA 12.8. Para comprobar que PyTorch ve
la GPU:

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'sin CUDA')"
```

Si `torch.cuda.is_available()` imprime `False`, ejecuta solo los backends CPU o
revisa el driver NVIDIA instalado en Ubuntu.

## Benchmark sintetico

Ejecuta todas las versiones disponibles con datos sinteticos:

```bash
source venv/bin/activate
python src/benchmark_syntetic/benchmark.py
```

Ejemplo con tamanos personalizados y salida CSV:

```bash
python src/benchmark_syntetic/benchmark.py \
  --users 5000 \
  --items 20000 \
  --dim 128 \
  --k 10 \
  --runs 5 \
  --warmup 1 \
  --block-size 1000 \
  --device auto \
  --csv results/synthetic_benchmark.csv
```

Comparar solo CPU NumPy contra Ray CPU:

```bash
python src/benchmark_syntetic/benchmark.py --versions numpy ray --device cpu
```

Comparar CPU NumPy contra PyTorch en GPU:

```bash
python src/benchmark_syntetic/benchmark.py --versions numpy torch_gpu --device cuda
```

Forzar Ray con CUDA:

```bash
python src/benchmark_syntetic/benchmark.py --versions ray --device cuda
```

## Dataset Amazon Reviews 2023

Los archivos de datos no se suben al repositorio. Descarga la categoria
`Electronics` de Amazon Reviews 2023 y dejala en `data/` con estos nombres:

```text
data/Electronics.jsonl.gz
data/meta_Electronics.jsonl.gz
```

Descarga en Ubuntu:

```bash
mkdir -p data
wget -O data/Electronics.jsonl.gz \
  https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_2023/raw/review_categories/Electronics.jsonl.gz
wget -O data/meta_Electronics.jsonl.gz \
  https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_2023/raw/meta_categories/meta_Electronics.jsonl.gz
```

Los archivos son grandes. `data/` esta incluido en `.gitignore`, por lo que se
mantiene local y no se versiona.

## Benchmark con Amazon Electronics

Ejecucion local en Ubuntu:

```bash
source venv/bin/activate
mkdir -p results
python src/benchmark_amazon/run_amazon_local.py \
  --backends numpy,torch_gpu,ray_cpu,ray_cuda \
  --max-interactions 100000 \
  --k 10 \
  --csv results/amazon_local_benchmark.csv
```

Por defecto ejecuta estas configuraciones: `5000:10000`, `10000:25000` y
`20000:50000`, donde cada par representa `usuarios:items`. Para cada tamano
ejecuta `numpy`, `torch_gpu`, `ray_cpu` y `ray_cuda`.

Si la maquina no tiene GPU CUDA disponible, puedes ejecutar solo CPU:

```bash
python src/benchmark_amazon/run_amazon_local.py \
  --backends numpy,ray_cpu \
  --max-interactions 100000 \
  --k 10 \
  --csv results/amazon_local_cpu_benchmark.csv
```

Ejecucion para el perfil Patagon:

```bash
python src/benchmark_amazon/run_amazon_patagon.py \
  --backends numpy,torch_gpu,ray_cpu,ray_cuda \
  --max-interactions 1000000 \
  --block-size 1000 \
  --k 10 \
  --csv results/amazon_patagon_benchmark.csv
```

Por defecto ejecuta estas configuraciones: `10000:50000`,
`50000:100000` y `100000:200000`, donde cada par representa
`usuarios:items`.

Tambien puedes elegir un subconjunto de backends:

```bash
python src/benchmark_amazon/run_amazon_local.py \
  --backends numpy,ray_cpu \
  --csv results/amazon_local_cpu_benchmark.csv
```

```bash
python src/benchmark_amazon/run_amazon_local.py \
  --backends torch_gpu,ray_cuda \
  --csv results/amazon_local_cuda_benchmark.csv
```

Los scripts filtran interacciones positivas con `rating >= 4`, construyen
embeddings de items desde la metadata textual y crean perfiles de usuario
promediando los embeddings de productos valorados positivamente.

## Metricas reportadas

Los CSV incluyen una fila por configuracion ejecutada. Las metricas principales
son:

- `time_s`: tiempo total de ejecucion.
- `cpu_avg_%` y `cpu_peak_%`: uso de CPU del proceso y procesos hijos.
- `ram_peak_MB`: maximo RSS observado del proceso y procesos hijos.
- `vram_peak_MB`: memoria GPU observada con `nvidia-smi`, si existe.
- `cpu_power_W`: potencia promedio CPU, calculada desde RAPL como joules por segundo.
- `gpu_power_W`: potencia promedio GPU reportada por `nvidia-smi`.
- `total_power_W`: suma de potencia CPU/GPU disponible.
- `cpu_energy_Wh`: energia CPU consumida en watt-hora.
- `gpu_energy_Wh`: energia GPU consumida en watt-hora.
- `total_energy_Wh`: suma de energia CPU/GPU disponible en watt-hora.
- `codecarbon_energy_kWh`: energia estimada por CodeCarbon durante el recomendador.
- `codecarbon_emissions_kg`: emisiones estimadas por CodeCarbon durante el recomendador.
- `codecarbon_cpu_energy_kWh`, `codecarbon_gpu_energy_kWh` y `codecarbon_ram_energy_kWh`: desglose estimado por CodeCarbon cuando esta disponible.

Cuando una metrica aparece como `N/A`, Ubuntu no expuso ese contador o falta la
herramienta necesaria. Las mediciones de recursos y energia se toman durante la
ejecucion del recomendador; la carga y preparacion del dataset queda separada en
`load_time_s`.

## Graficos

Para generar un PDF con un grafico por cada metrica disponible en los CSV de
`results/`:

```bash
source venv/bin/activate
python src/plot_benchmark_results.py
```

El comando crea:

- `results/benchmark_metrics.pdf`: PDF multipagina con los graficos.
- `results/plots/*.png`: una imagen por metrica.

Tambien puedes indicar CSV concretos o cambiar la salida:

```bash
python src/plot_benchmark_results.py \
  --input results/amazon_local_cpu_benchmark.csv results/amazon_local_cuda_benchmark.csv \
  --output-pdf results/amazon_local_metricas.pdf \
  --output-dir results/plots_amazon_local
```

## Docker en Ubuntu

Antes de ejecutar los contenedores, descarga el dataset en `data/`. Los comandos
Docker montan `data/` dentro del contenedor como `/app/data`, por lo que deben
existir estos archivos en la maquina host:

```text
data/Electronics.jsonl.gz
data/meta_Electronics.jsonl.gz
```

Las imagenes principales ya estan creadas y publicadas en Docker Hub:

```text
lagartin1/hpc-recommender-local:latest
lagartin1/hpc-recommender-local-cuda:latest
```

Puedes descargarlas antes de ejecutar:

```bash
docker pull lagartin1/hpc-recommender-local:latest
docker pull lagartin1/hpc-recommender-local-cuda:latest
```

### Docker CPU

Imagen local CPU desde Docker Hub, ejecutando `numpy` y `ray_cpu`:

```bash
mkdir -p results
docker run --rm \
  -v "$PWD/data:/app/data:ro" \
  -v "$PWD/results:/app/results" \
  lagartin1/hpc-recommender-local:latest
```

El CSV queda en `results/amazon_local_cpu_benchmark.csv`.

Si necesitas reconstruir la imagen localmente:

```bash
docker build -f Dockerfile.local -t hpc-recommender-local .
```

### Problemas comunes de CUDA en Docker

Para usar GPU dentro de Docker instala NVIDIA Container Toolkit en Ubuntu:

```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt update
sudo apt install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

Verifica que Docker ve la GPU:

```bash
docker run --rm --gpus all nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04 nvidia-smi
```

Imagen local CUDA desde Docker Hub, ejecutando `torch_gpu` y `ray_cuda`:

```bash
mkdir -p results
docker run --rm --gpus all \
  -v "$PWD/data:/app/data:ro" \
  -v "$PWD/results:/app/results" \
  lagartin1/hpc-recommender-local-cuda:latest
```

El CSV queda en `results/amazon_local_cuda_benchmark.csv`.

Si necesitas reconstruir la imagen localmente:

```bash
docker build -f Dockerfile.cuda.local -t hpc-recommender-local-cuda .
```

## Patagon

El perfil Patagon usa configuraciones mas grandes para ejecutar los benchmarks
en el Patagon Supercomputer de la Universidad Austral de Chile.

### Docker Patagon CPU

Imagen Patagon CPU, ejecutando `numpy` y `ray_cpu`:

```bash
mkdir -p results
docker build -f Dockerfile.patagon -t hpc-recommender-patagon .
docker run --rm \
  -v "$PWD/data:/app/data:ro" \
  -v "$PWD/results:/app/results" \
  hpc-recommender-patagon
```

El CSV queda en `results/amazon_patagon_cpu_benchmark.csv`.

### Docker Patagon CUDA

Imagen Patagon CUDA, ejecutando `torch_gpu` y `ray_cuda`:

```bash
mkdir -p results
docker build -f Dockerfile.cuda.patagon -t hpc-recommender-patagon-cuda .
docker run --rm --gpus all \
  -v "$PWD/data:/app/data:ro" \
  -v "$PWD/results:/app/results" \
  hpc-recommender-patagon-cuda
```

El CSV queda en `results/amazon_patagon_cuda_benchmark.csv`.

Para cambiar parametros dentro del contenedor, reemplaza el comando por defecto:

```bash
docker run --rm --gpus all \
  -v "$PWD/data:/app/data:ro" \
  -v "$PWD/results:/app/results" \
  hpc-recommender-patagon-cuda \
  python3 src/benchmark_amazon/run_amazon_patagon.py \
    --backends torch_gpu,ray_cuda \
    --configs 10000:50000,50000:100000,100000:200000 \
    --csv /app/results/amazon_patagon_100k_200k.csv \
    --output-dir /app/results
```

### Agradecimientos


Esta investigación fue apoyada por el supercomputador Patagón de la Universidad Austral de Chile (FONDEQUIP EQM180042).

- Patagon Supercomputer, Austral University of Chile, 2021. https://patagon.uach.cl


```bibtex
@misc{patagon,
    howpublished = {{\url{https://patagon.uach.cl}}},
    author = {{Patag\'on Supercomputer}},
    year = {2021}
}
```

## Referencia del dataset

Este proyecto usa la categoria `Electronics` del dataset Amazon Reviews 2023,
publicado por McAuley Lab.

Sitio del dataset:

```text
https://amazon-reviews-2023.github.io/
```

Formato BibTeX:

```bibtex
@article{hou2024bridging,
  title={Bridging Language and Items for Retrieval and Recommendation},
  author={Hou, Yupeng and Li, Jiacheng and He, Zhankui and Yan, An and Chen, Xiusi and McAuley, Julian},
  journal={arXiv preprint arXiv:2403.03952},
  year={2024}
}
```
