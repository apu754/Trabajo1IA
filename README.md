# IA2 VideoGame Sales API (FastAPI + Docker)

API para **predecir ventas globales (en millones de copias)** de videojuegos usando:

- Un **modelo ganador (RandomForest)** entrenado con scikit-learn.
- Un **MLP (Keras)** opcional, seleccionable por request.

Los **artefactos del modelo** (preprocesador y modelos entrenados) se montan desde la carpeta `artifacts/` en tiempo de ejecución.

## Entrenamiento (Colab recomendado)

1. Abre el cuaderno:  
   [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/apu754/Trabajo1IA/blob/main/IA2_vgsales.ipynb)

2. Carga la base:

   - O sube tu `vgsales.csv` a Colab.
   - O usa la API de Kaggle (ver `data/README.md`) para descargar el dataset.
   - **Importante:** respeta la licencia del dataset; no lo redistribuyas en el repo si no está permitido.

3. Ejecuta el cuaderno completo (limpieza, EDA, modelos, K-Fold, GridSearchCV).

4. Al final, el cuaderno **exporta**:

   - `artifacts/preprocessor.pkl`
   - `artifacts/model_meta.json` (ej. `{"winner":"RandomForest","type":"sklearn"}`)
   - `artifacts/model_winner.pkl` (si winner es sklearn)
   - `artifacts/mlp_best.keras` (opcional, si quieres usar el MLP)

5. Descarga la carpeta `artifacts/` y llévala al servidor donde corre la API (o móntala como volumen en Docker).

---

### dataset

https://www.kaggle.com/datasets/gregorut/videogamesales

---

## Estructura

```bash
vgsales/
├─ app.py
├─ requirements.txt
├─ Dockerfile
├─ .dockerignore
├─ .gitignore
└─ artifacts/ #Poner los modelos
└─ .gitkeep
```

**Coloca dentro de `artifacts/`:**

- `preprocessor.pkl`
- `model_meta.json` → ejemplo: `{"winner":"RandomForest","type":"sklearn"}`
- `model_winner.pkl` → si el ganador es scikit-learn (RF/GB/ET)
- `model_winner.keras` → si el ganador fuera un MLP de Keras (opcional)
- `mlp_best.keras` → MLP adicional (opcional)

> **Importante versiones**
>
> - Si usas `.keras` (Keras 3), en Docker instala **TensorFlow 2.17** y **keras 3.x**.
> - Si ves errores al cargar `preprocessor.pkl` por scikit-learn, alinea la versión:
>   - Entrenaste con `scikit-learn==1.6.1` → usa esa misma en `requirements.txt`.

---

## Requisitos

- **Docker** (recomendado)  
  o
- Python 3.10+ y `pip`

---

## Correr con Docker

### 1) Build

```bash
docker build --no-cache -t vgsales-fastapi .
```

### 2) Run (Windows PowerShell)

Desde la carpeta del proyecto (donde está artifacts/):

```bash
docker run -d --name vgsales-api `
  --restart unless-stopped `
  -p 8000:8000 `
  -e DEFAULT_MODEL=winner `             # o mlp
  -e ARTIFACTS_DIR=/app/artifacts `
  -v "${PWD}\artifacts:/app/artifacts:ro" `
  vgsales-fastapi
```

Si tu ruta tiene espacios y te falla, usa ruta absoluta entre comillas:

```bash
-v "C:\Users\...\vgsales\artifacts:/app/artifacts:ro"
```

### 3) Endpoints

Salud: GET http://localhost:8000/health

Docs (Swagger): http://localhost:8000/docs

### 4) Probar rápido (PowerShell)

```bash
$body = @{
  Platform = 'PS4'
  Year     = 2016
  Genre    = 'Action'
  Publisher= 'Ubisoft'
} | ConvertTo-Json

# Predicción con el modelo por defecto (winner)
Invoke-RestMethod -Method POST -Uri "http://localhost:8000/predict" -ContentType "application/json" -Body $body

# Forzar MLP
Invoke-RestMethod -Method POST -Uri "http://localhost:8000/predict?model=mlp" -ContentType "application/json" -Body $body
```

---

### Batch:

```bash
$batch = @(
  @{Platform='X360'; Year=2011; Genre='Sports';  Publisher='Electronic Arts'},
  @{Platform='Wii';  Year=2009; Genre='Racing';  Publisher='Nintendo'},
  @{Platform='PS3';  Year=2012; Genre='Shooter'; Publisher='Unknown'}
) | ConvertTo-Json

Invoke-RestMethod -Method POST -Uri "http://localhost:8000/predict_batch?model=winner" -ContentType "application/json" -Body $batch
```

---

### Correr local (sin Docker)

```bash
python -m venv .venv
# Windows
.\.venv\Scripts\activate
# Linux/Mac
source .venv/bin/activate

pip install --no-cache-dir -r requirements.txt

# Si tus artefactos no están en ./artifacts:
# set ARTIFACTS_DIR=C:\ruta\a\artifacts   # Windows cmd
# $env:ARTIFACTS_DIR="C:\ruta\a\artifacts" # PowerShell
# export ARTIFACTS_DIR=/ruta/a/artifacts   # Linux/Mac

uvicorn app:app --host 0.0.0.0 --port 8000
```
