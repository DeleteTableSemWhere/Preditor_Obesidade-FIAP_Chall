"""
Helpers de inferência utilizados pelo front-end Streamlit.

Funções
-------
load_artifacts()    → carrega modelo + label encoder (cache via @st.cache_resource ou singleton)
predict(input_dict) → retorna (classe_predita: str, probabilidades: dict)
"""
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Any

from ml.train import ENGINEERED_NUMERIC, CATEGORICAL_FEATURES, add_engineered_features

MODEL_PATH         = Path(__file__).parent / "model.pkl"
LABEL_ENCODER_PATH = Path(__file__).parent / "label_encoder.pkl"


def _load_impl():
    """Carrega os artefatos do disco sem cache."""
    pipeline = joblib.load(MODEL_PATH)
    le       = joblib.load(LABEL_ENCODER_PATH)
    return pipeline, le


try:
    import streamlit as st
    load_artifacts = st.cache_resource(show_spinner="Carregando modelo...")(_load_impl)
except ImportError:
    _cache: list = []

    def load_artifacts():
        if not _cache:
            _cache.extend(_load_impl())
        return _cache[0], _cache[1]


def predict(input_dict: dict[str, Any]) -> tuple[str, dict[str, float]]:
    """
    Parâmetros
    ----------
    input_dict : valores brutos das features no formato esperado pelo treino

    Retorna
    -------
    classe_predita  : rótulo legível do nível de obesidade
    probabilidades  : {rótulo_classe: probabilidade} para todas as classes
    """
    pipeline, le = load_artifacts()

    df    = pd.DataFrame([input_dict])
    df    = add_engineered_features(df)
    X     = df[ENGINEERED_NUMERIC + CATEGORICAL_FEATURES]

    proba          = pipeline.predict_proba(X)[0]
    class_idx      = int(np.argmax(proba))
    classe_predita = le.inverse_transform([class_idx])[0]
    probabilidades = {label: float(p) for label, p in zip(le.classes_, proba)}

    return classe_predita, probabilidades
