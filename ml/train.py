import os
os.environ["LOKY_MAX_CPU_COUNT"] = "1"

import json
import shutil
import joblib
import numpy as np
import pandas as pd
import sklearn
from datetime import datetime, timezone
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, accuracy_score, confusion_matrix, f1_score,
    roc_auc_score, roc_curve, auc,
)
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import (
    StratifiedKFold, cross_val_score, train_test_split
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

from db.supabase_client import fetch_all

MODEL_PATH = Path(__file__).parent / "model.pkl"
LABEL_ENCODER_PATH = Path(__file__).parent / "label_encoder.pkl"
METRICS_PATH = Path(__file__).parent / "metrics.json"
TARGET = "obesity"

# ── Grupos de colunas ────────────────────────────────────────────────────────
NUMERIC_FEATURES = ["age", "fcvc", "ncp", "ch2o", "faf", "tue"]
CATEGORICAL_FEATURES = [
    "gender", "family_history", "favc", "caec", "smoke", "scc", "calc", "mtrans"
]


# ── Funções de feature engineering ──────────────────────────────────────────

def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Cria features derivadas para enriquecer o sinal do modelo."""
    df = df.copy()
    # height e weight excluídos: o modelo aprende exclusivamente de hábitos de vida
    df["activity_hydration"] = df["faf"] * df["ch2o"]        # sinergia atividade × hidratação
    df["lifestyle_score"] = df["faf"] + df["ch2o"] - df["ncp"] + df["fcvc"]  # pontuação de hábitos
    return df


ENGINEERED_NUMERIC = NUMERIC_FEATURES + ["activity_hydration", "lifestyle_score"]


def load_data() -> pd.DataFrame:
    rows = fetch_all("obesity_data")
    print(f"{len(rows)} linhas carregadas do Supabase.")
    return pd.DataFrame(rows)


def prepare(df: pd.DataFrame):
    if "family_history_with_overweight" in df.columns:
        df = df.rename(columns={"family_history_with_overweight": "family_history"})
    if "nobeyesdad" in df.columns:
        df = df.rename(columns={"nobeyesdad": TARGET})
    if "obesity_level" in df.columns:
        df = df.rename(columns={"obesity_level": TARGET})

    df = add_engineered_features(df)
    X = df[ENGINEERED_NUMERIC + CATEGORICAL_FEATURES]
    y = df[TARGET]
    return X, y


# ── Construção do pipeline ────────────────────────────────────────────────────

def build_pipeline() -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), ENGINEERED_NUMERIC),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
        ]
    )
    clf = RandomForestClassifier(
        n_estimators=150,
        max_depth=15,
        min_samples_leaf=3,
        class_weight="balanced",
        random_state=42,
        n_jobs=1,
    )
    return Pipeline([("preprocessor", preprocessor), ("classifier", clf)])


# ── Rotina principal de treinamento ──────────────────────────────────────────

def train() -> None:
    df = load_data()
    X, y = prepare(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    le = LabelEncoder()
    y_train = le.fit_transform(y_train)
    y_test  = le.transform(y_test)

    pipeline = build_pipeline()

    # Validação cruzada estratificada
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="accuracy", n_jobs=1)
    print(f"\nAcurácia CV: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # Treino final com todo o conjunto de treino
    pipeline.fit(X_train, y_train)

    # Avaliação no conjunto hold-out
    y_pred  = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)
    acc      = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro")

    y_test_bin = label_binarize(y_test, classes=list(range(len(le.classes_))))
    roc_auc = roc_auc_score(y_test_bin, y_proba, multi_class="ovr", average="macro")

    roc_per_class = {}
    for i, cls in enumerate(le.classes_):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_proba[:, i])
        roc_per_class[cls] = {
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist(),
            "auc": round(float(auc(fpr, tpr)), 4),
        }

    print(f"Acurácia no hold-out: {acc:.4f}")
    print(f"Macro-F1 no hold-out: {macro_f1:.4f}")
    print(f"ROC-AUC Macro (OvR):  {roc_auc:.4f}\n")

    if acc < 0.75:
        print("AVISO: acurácia abaixo do limite mínimo de 75%.")

    report = classification_report(y_test, y_pred, target_names=le.classes_, output_dict=True)
    cm = confusion_matrix(y_test, y_pred).tolist()
    now = datetime.now(timezone.utc)
    trained_at = now.strftime("%Y-%m-%dT%H:%M:%SZ")

    # Nomes das features extraídos aqui (versão sklearn consistente com o treino)
    cat_encoder   = pipeline.named_steps["preprocessor"].named_transformers_["cat"]
    cat_feat_names = cat_encoder.get_feature_names_out(CATEGORICAL_FEATURES).tolist()
    all_feat_names = list(ENGINEERED_NUMERIC) + cat_feat_names
    feat_importances = [
        {"feature": name, "importance": round(float(imp), 6)}
        for name, imp in zip(all_feat_names, pipeline.named_steps["classifier"].feature_importances_)
    ]
    feat_importances.sort(key=lambda x: x["importance"], reverse=True)

    metrics = {
        "trained_at": trained_at,
        "n_samples": len(X),
        "sklearn_version": sklearn.__version__,
        "test_accuracy": round(acc, 4),
        "macro_f1": round(float(macro_f1), 4),
        "roc_auc_macro": round(float(roc_auc), 4),
        "roc_auc_per_class": roc_per_class,
        "cv_mean": round(float(cv_scores.mean()), 4),
        "cv_std": round(float(cv_scores.std()), 4),
        "cv_scores": [round(float(s), 4) for s in cv_scores],
        "class_names": list(le.classes_),
        "confusion_matrix": cm,
        "classification_report": report,
        "feature_importances": feat_importances,
    }
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    print(f"Métricas salvas em {METRICS_PATH}")

    history_path = METRICS_PATH.parent / "metrics_history.jsonl"
    with open(history_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(metrics) + "\n")
    print(f"Histórico atualizado em {history_path}")

    timestamp = now.strftime("%Y%m%d_%H%M%S")
    versioned_model_path = MODEL_PATH.parent / f"model_{timestamp}.pkl"
    versioned_le_path    = LABEL_ENCODER_PATH.parent / f"label_encoder_{timestamp}.pkl"

    joblib.dump(pipeline, MODEL_PATH)
    shutil.copy2(MODEL_PATH, versioned_model_path)
    joblib.dump(le, LABEL_ENCODER_PATH)
    shutil.copy2(LABEL_ENCODER_PATH, versioned_le_path)
    print(f"Modelo salvo em {MODEL_PATH} e {versioned_model_path}")
    print(f"Label encoder salvo em {LABEL_ENCODER_PATH} e {versioned_le_path}")

    # Mantém apenas as 2 versões mais recentes de cada artefato versionado
    for pattern in ("model_*.pkl", "label_encoder_*.pkl"):
        versioned = sorted(MODEL_PATH.parent.glob(pattern), key=lambda p: p.stat().st_mtime)
        for old in versioned[:-2]:
            old.unlink()
            print(f"Removido artefato antigo: {old.name}")


if __name__ == "__main__":
    train()
