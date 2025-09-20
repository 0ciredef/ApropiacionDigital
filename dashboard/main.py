import os
import re
import math
import json
import joblib
import pandas as pd
from typing import Optional, Dict, Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, validator

# ====================================================
# Config
# ====================================================
MODEL_PATH = os.getenv("MODEL_PATH", "mlp_mejor_pipeline.joblib")
MASTER_CSV_PATH = "https://aprodigcodata.s3.eu-north-1.amazonaws.com/maestro_global_variables_municipio.csv"

# ====================================================
# Utilidades
# ====================================================
def norm_mpio(code: str) -> str:
    """Normaliza código municipal: deja solo dígitos y completa a 5 con ceros a la izquierda."""
    s = str(code).strip()
    s = re.sub(r"\D", "", s)
    if len(s) > 5:
        s = s[-5:]
    return s.zfill(5)

def to_float_maybe_comma(x) -> Optional[float]:
    if x is None or (isinstance(x, float) and (math.isnan(x))):
        return None
    if isinstance(x, str):
        x = x.strip().replace(",", ".")
    try:
        return float(x)
    except Exception:
        return None

def pick(colnames, candidates):
    for c in candidates:
        if c in colnames:
            return c
    raise RuntimeError(f"No se encontró ninguna de {candidates} en columnas {list(colnames)}")

# ====================================================
# Carga maestro
# ====================================================


maestro_raw = pd.read_csv(MASTER_CSV_PATH, dtype=str)
maestro_raw.columns = maestro_raw.columns.str.strip()

# Descubrir columnas
COL_MPIO = None
for cand in ["COD_MUNICIPIO","cod_municipio","MUNICIPIO_COD","CODIGO_MUNICIPIO","MpCodigo"]:
    if cand in maestro_raw.columns:
        COL_MPIO = cand
        break
if COL_MPIO is None:
    raise RuntimeError(f"No encuentro columna de código municipal en el maestro. Probé varias opciones. Columnas: {list(maestro_raw.columns)}")

COL_DEPTO = None
for cand in ["COD_DEPARTAMENTO","cod_departamento","DEPARTAMENTO","DEPTO_COD","DEPARTAMENTO_COD"]:
    if cand in maestro_raw.columns:
        COL_DEPTO = cand
        break

COL_INDICE = None
for cand in ["INDICE","indice","INDICE_FIJO","INDICE_INTERNET"]:
    if cand in maestro_raw.columns:
        COL_INDICE = cand
        break
if COL_INDICE is None:
    raise RuntimeError("No encuentro columna INDICE en maestro (INDICE/indice/INDICE_FIJO/etc.).")

COL_IPM = None
for cand in ["ipm_depto","ipm","IPM_DEPTO"]:
    if cand in maestro_raw.columns:
        COL_IPM = cand
        break
if COL_IPM is None:
    raise RuntimeError("No encuentro columna IPM en maestro (ipm_depto/ipm/IPM_DEPTO).")

COL_SABER = None
for cand in ["saber_punt_global_mean","SABER_PUNT_GLOBAL_MEAN","saber_global"]:
    if cand in maestro_raw.columns:
        COL_SABER = cand
        break
if COL_SABER is None:
    raise RuntimeError("No encuentro columna saber_punt_global_mean en maestro.")

# Normaliza códigos
maestro_raw[COL_MPIO] = maestro_raw[COL_MPIO].astype(str).map(norm_mpio)

# Opcional: depuración de duplicados por municipio (nos quedamos con la primera aparición)
if maestro_raw.duplicated(subset=[COL_MPIO]).any():
    maestro_raw = maestro_raw.drop_duplicates(subset=[COL_MPIO], keep="first")

# Índice por municipio
maestro = maestro_raw.set_index(COL_MPIO)

# ====================================================
# Carga modelo
# ====================================================
if not os.path.exists(MODEL_PATH):
    raise RuntimeError(f"No se encontró MODEL_PATH: {MODEL_PATH}")

model = joblib.load(MODEL_PATH)

# Columnas esperadas por el pipeline (ajústalas si tu modelo pide otras)
NUM_FEATS = [
    "RANGO_EDAD","ESTRATO","PB1_bin","SEXO_bin",
    "P33","ipm_depto","INDICE","saber_punt_global_mean"
]
CAT_FEATS = []  # si tu pipeline usó OneHot para dept_code:
# CAT_FEATS = ["dept_code"]

FEATURE_ORDER = NUM_FEATS + CAT_FEATS

# ====================================================
# FastAPI
# ====================================================
app = FastAPI(title="API Apropiación Digital", version="1.0.0")

class PredictRequest(BaseModel):
    municipio_code: str = Field(..., description="Código DANE de municipio, puede venir con o sin ceros a la izquierda")
    RANGO_EDAD: float
    ESTRATO: float
    PB1_bin: float
    SEXO_bin: float
    P33: float
    dept_code: Optional[str] = Field(None, description="Código de departamento (opcional)")

    @validator("municipio_code", pre=True)
    def _norm_mpio_val(cls, v):
        return norm_mpio(v)

class PredictResponse(BaseModel):
    prediction: float
    features_used: Dict[str, Any]

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": os.path.abspath(MODEL_PATH)}

@app.get("/version")
def version():
    return {"api": "1.0.0"}

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    mpio = req.municipio_code

    # 1) Lookup en maestro
    if mpio not in maestro.index:
        raise HTTPException(
            status_code=404,
            detail=f"Código de municipio no encontrado en el maestro: {mpio}"
        )
    row = maestro.loc[mpio]

    # 2) Extrae y normaliza indicadores del maestro
    val_indice = to_float_maybe_comma(row.get(COL_INDICE))
    val_ipm    = to_float_maybe_comma(row.get(COL_IPM))
    val_saber  = to_float_maybe_comma(row.get(COL_SABER))

    missing = [name for name, v in [("INDICE", val_indice), ("ipm_depto", val_ipm), ("saber_punt_global_mean", val_saber)] if v is None]
    if missing:
        raise HTTPException(
            status_code=404,
            detail=f"Datos faltantes en maestro para {mpio}: {missing}"
        )

    # 3) dept_code: si no viene en request, intenta derivarlo del maestro (si existe la columna)
    dept_code = req.dept_code
    if dept_code is None and COL_DEPTO is not None and COL_DEPTO in maestro.columns:
        dept_code = str(row.get(COL_DEPTO))

    # 4) Arma dataframe con el orden/columnas que espera el pipeline
    data = {
        "RANGO_EDAD": req.RANGO_EDAD,
        "ESTRATO": req.ESTRATO,
        "PB1_bin": req.PB1_bin,
        "SEXO_bin": req.SEXO_bin,
        "P33": req.P33,
        "INDICE": val_indice,
        "ipm_depto": val_ipm,
        "saber_punt_global_mean": val_saber,
    }
    if "dept_code" in CAT_FEATS:
        data["dept_code"] = None if dept_code is None else str(dept_code)

    # Asegurar orden de columnas
    X = pd.DataFrame([data], columns=FEATURE_ORDER)

    # 5) Predicción
    try:
        y_hat = float(model.predict(X)[0])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al predecir: {e}")

    return PredictResponse(
        prediction=y_hat,
        features_used=data
    )

# Para ejecutar en dev:
# uvicorn main:app --reload
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
