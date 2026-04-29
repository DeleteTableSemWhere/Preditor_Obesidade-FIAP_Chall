import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Any

from ml.train import ENGINEERED_NUMERIC, CATEGORICAL_FEATURES, add_engineered_features

MODEL_PATH         = Path(__file__).parent / "model.pkl"
LABEL_ENCODER_PATH = Path(__file__).parent / "label_encoder.pkl"


def _load_impl():
    pipeline = joblib.load(MODEL_PATH)
    le       = joblib.load(LABEL_ENCODER_PATH)
    return pipeline, le


try:
    import streamlit as st
    load_artifacts = st.cache_resource(show_spinner="Carregando modelo...")(_load_impl)
except ImportError:
    from functools import lru_cache
    load_artifacts = lru_cache(maxsize=1)(_load_impl)


def predict(input_dict: dict[str, Any]) -> tuple[str, dict[str, float]]:
    pipeline, le = load_artifacts()

    df    = pd.DataFrame([input_dict])
    df    = add_engineered_features(df)
    X     = df[ENGINEERED_NUMERIC + CATEGORICAL_FEATURES]

    proba          = pipeline.predict_proba(X)[0]
    class_idx      = int(np.argmax(proba))
    classe_predita = le.inverse_transform([class_idx])[0]
    probabilidades = {label: float(p) for label, p in zip(le.classes_, proba)}

    return classe_predita, probabilidades
