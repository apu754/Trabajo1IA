import os, json, joblib, numpy as np, pandas as pd
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from typing import List, Optional, Literal

# --- rutas de artefactos ---
ART_DIR    = os.environ.get("ARTIFACTS_DIR", "artifacts")
PRE_PATH   = os.path.join(ART_DIR, "preprocessor.pkl")
META_PATH  = os.path.join(ART_DIR, "model_meta.json")     # winner
KERAS_WIN  = os.path.join(ART_DIR, "model_winner.keras")  # si el winner fuera Keras
SKL_WIN    = os.path.join(ART_DIR, "model_winner.pkl")    # si el winner es sklearn
KERAS_MLP  = os.path.join(ART_DIR, "mlp_best.keras")      # mejor MLP (opcional)
MLP_META   = os.path.join(ART_DIR, "mlp_meta.json")       # opcional (solo info)

DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "winner")  # "winner" o "mlp"

# --- cargar preprocesador ---
pre = joblib.load(PRE_PATH)

# --- cargar winner (según meta) ---
meta_winner = json.load(open(META_PATH, encoding="utf-8"))
winner_type = meta_winner.get("type", "sklearn")
if winner_type == "keras":
    import tensorflow as tf
    winner_model = tf.keras.models.load_model(KERAS_WIN)
else:
    winner_model = joblib.load(SKL_WIN)

# --- intentar cargar el MLP mejor ---
mlp_model = None
mlp_loaded = False
mlp_error = None
try:
    import keras  # Keras 3 (nuevo formato .keras)
    mlp_model = keras.models.load_model(KERAS_MLP)
    mlp_loaded = True
except Exception as e1:
    try:
        import tensorflow as tf  # fallback a tf.keras
        mlp_model = tf.keras.models.load_model(KERAS_MLP)
        mlp_loaded = True
    except Exception as e2:
        mlp_model = None
        mlp_loaded = False
        mlp_error = f"{type(e2).__name__}: {e2}"

app = FastAPI(title="IA2 VideoGame Sales API", version="1.1")

class Game(BaseModel):
    Platform: str
    Year: Optional[int] = None
    Genre: str
    Publisher: str

def predict_core(df_like: pd.DataFrame, model) -> np.ndarray:
    Xp = pre.transform(df_like)
    ylog = model.predict(Xp)
    ylog = np.asarray(ylog).ravel()
    return np.expm1(ylog)  # ventas en millones

@app.get("/health")
def health():
    return {
        "status": "ok",
        "winner": meta_winner.get("winner"),
        "winner_type": winner_type,
        "mlp_loaded": mlp_loaded,
        "mlp_error": mlp_error,        
        "default_model": DEFAULT_MODEL,
    }

# -------- endpoint único con selector --------
@app.post("/predict")
def predict_one(
    game: Game,
    model: Literal["winner", "mlp"] = Query(DEFAULT_MODEL, description="Modelo a usar"),
):
    df = pd.DataFrame([game.dict()])
    if model == "mlp":
        if not mlp_loaded:
            raise HTTPException(status_code=400, detail="MLP no disponible (mlp_best.keras no encontrado).")
        y = predict_core(df, mlp_model)
        used = "mlp"
    else:
        y = predict_core(df, winner_model)
        used = "winner"
    return {"model_used": used, "Global_Sales_pred": float(y[0])}

@app.post("/predict_batch")
def predict_batch(
    games: List[Game],
    model: Literal["winner", "mlp"] = Query(DEFAULT_MODEL, description="Modelo a usar"),
):
    df = pd.DataFrame([g.dict() for g in games])
    if model == "mlp":
        if not mlp_loaded:
            raise HTTPException(status_code=400, detail="MLP no disponible (mlp_best.keras no encontrado).")
        y = predict_core(df, mlp_model)
        used = "mlp"
    else:
        y = predict_core(df, winner_model)
        used = "winner"
    return {"model_used": used, "Global_Sales_pred": [float(v) for v in y]}

# -------- opcional: endpoints separados --------
@app.post("/predict_winner")
def predict_winner(game: Game):
    df = pd.DataFrame([game.dict()])
    y = predict_core(df, winner_model)
    return {"model_used": "winner", "Global_Sales_pred": float(y[0])}

@app.post("/predict_mlp")
def predict_mlp(game: Game):
    if not mlp_loaded:
        raise HTTPException(status_code=400, detail="MLP no disponible (mlp_best.keras no encontrado).")
    df = pd.DataFrame([game.dict()])
    y = predict_core(df, mlp_model)
    return {"model_used": "mlp", "Global_Sales_pred": float(y[0])}
