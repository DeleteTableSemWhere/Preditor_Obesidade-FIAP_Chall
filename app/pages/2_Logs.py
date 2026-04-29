"""
Página 2 – Logs de Predições
KPIs expandidos → tabela de registros.
Datas convertidas para o fuso horário de Brasília (America/Sao_Paulo).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd
import streamlit as st

from db.supabase_client import fetch_all
from app.constants import OBESITY_LABELS_SHORT as OBESITY_LABELS, GENERO_PT

st.set_page_config(page_title="Logs | Preditor de Obesidade", page_icon="📋", layout="wide")
st.title("📋 Logs de Predições")


@st.cache_data(ttl=60)
def load_logs() -> pd.DataFrame:
    df = pd.DataFrame(fetch_all("predictions_log", order_by="created_at", desc=True))
    if df.empty:
        return df
    df["created_at"] = (
        pd.to_datetime(df["created_at"], utc=True)
        .dt.tz_convert("America/Sao_Paulo")
    )
    df["date"]   = df["created_at"].dt.normalize().dt.tz_localize(None)
    df["Classe"] = df["predicted_class"].map(OBESITY_LABELS).fillna(df["predicted_class"])
    df["Gênero"] = df["gender"].map(GENERO_PT).fillna(df["gender"])
    return df


df = load_logs()

if df.empty:
    st.info("Nenhuma predição registrada ainda. Use a página **Previsão** para gerar dados.")
    st.stop()

# ── KPI cards ────────────────────────────────────────────────────────────────
c1, c2, c3 = st.columns(3)
c1.metric("Total de Predições", len(df))
c2.metric("Classes Distintas",  df["predicted_class"].nunique())
c3.metric("Confiança Média",    f"{df['probability'].mean():.1%}")

st.divider()

# ── Tabela ───────────────────────────────────────────────────────────────────
st.subheader("Registros de Predições")
display = df[["created_at", "Classe", "probability", "age", "Gênero"]].copy()
display["created_at"] = display["created_at"].dt.strftime("%d/%m/%Y %H:%M")
display = display.rename(columns={
    "created_at": "Data/Hora",
    "probability": "Confiança",
    "age":         "Idade",
})
display["Confiança"] = display["Confiança"].map("{:.1%}".format)
st.dataframe(display, hide_index=True)
