"""
Página 1 – Previsão (ML)
Formulário com hábitos de vida e dados pessoais (sem altura/peso).
Resultado com gauge + probabilidades por classe.
O log no Supabase tem spinner próprio, separado da inferência do modelo.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import streamlit as st
import plotly.graph_objects as go

ROOT = Path(__file__).parent.parent.parent

from app.constants import OBESITY_LABELS, OBESITY_ORDER

st.set_page_config(page_title="Previsão | ObesityPredict", page_icon="🔬", layout="wide")

# ── Estilo do botão CTA ───────────────────────────────────────────────────────
st.markdown("""
<style>
div[data-testid="stFormSubmitButton"] button {
    background-color: #E74C3C !important;
    color: white !important;
    font-size: 1.15em !important;
    font-weight: 700 !important;
    border-radius: 8px !important;
    border: none !important;
    padding: 0.65em 1em !important;
    transition: background-color 0.2s;
}
div[data-testid="stFormSubmitButton"] button:hover {
    background-color: #C0392B !important;
}
</style>
""", unsafe_allow_html=True)

st.title("🔬 Previsão de Nível de Obesidade")
st.info("⚕️ Este resultado é uma ferramenta de apoio à triagem. Não substitui avaliação clínica.")
st.caption("Preencha os dados do paciente para obter a predição do modelo.")

# ── Constantes ────────────────────────────────────────────────────────────────
GAUGE_COLORS = ["#2ECC71","#82E0AA","#F7DC6F","#F0A500","#E67E22","#E74C3C","#922B21"]

# Mapeamentos PT → EN para que o modelo receba os valores originais do treino
SIM_NAO = {"Sim": "yes", "Não": "no"}
FREQ    = {"Nunca": "no", "Às vezes": "Sometimes", "Frequentemente": "Frequently", "Sempre": "Always"}
TRANSP  = {
    "Transporte Público": "Public_Transportation",
    "A Pé":               "Walking",
    "Automóvel":          "Automobile",
    "Moto":               "Motorbike",
    "Bicicleta":          "Bike",
}
GENERO = {"Masculino": "Male", "Feminino": "Female"}


from ml.predict import predict

# ── Gauge de nível de obesidade ───────────────────────────────────────────────
def build_gauge(predicted_class: str, probability: float) -> go.Figure:
    idx   = OBESITY_ORDER.index(predicted_class) if predicted_class in OBESITY_ORDER else 0
    label = OBESITY_LABELS.get(predicted_class, predicted_class)
    fig   = go.Figure(go.Indicator(
        mode="gauge+number",
        value=idx,
        title={"text": f"<b>{label}</b><br><span style='font-size:0.8em'>Confiança: {probability:.1%}</span>"},
        gauge={
            "axis": {
                "range": [0, 6],
                "tickvals": list(range(7)),
                "ticktext": list(OBESITY_LABELS.values()),
            },
            "bar": {"color": GAUGE_COLORS[idx]},
            "steps": [
                {
                    "range": [i, i + 1],
                    "color": f"rgba({int(GAUGE_COLORS[i][1:3],16)},"
                             f"{int(GAUGE_COLORS[i][3:5],16)},"
                             f"{int(GAUGE_COLORS[i][5:],16)},0.25)"
                }
                for i in range(7)
            ],
        },
    ))
    fig.update_layout(height=300, margin=dict(t=80, b=20, l=30, r=30))
    return fig


# ── Formulário ────────────────────────────────────────────────────────────────
with st.form("prediction_form"):
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Dados Pessoais")
        gender_pt         = st.selectbox("Gênero", list(GENERO.keys()))
        age               = st.number_input("Idade (anos)", min_value=1, max_value=120, value=25)
        family_history_pt = st.selectbox("Histórico familiar de sobrepeso?", list(SIM_NAO.keys()))

    with col2:
        st.subheader("Hábitos Alimentares")
        favc_pt  = st.selectbox("Consome alimentos hipercalóricos?", list(SIM_NAO.keys()))
        FCVC_OPT = {"Raramente": 1, "Às vezes": 2, "Sempre": 3}
        fcvc     = float(FCVC_OPT[st.selectbox("Frequência de verduras/vegetais",
                                                list(FCVC_OPT.keys()), index=1)])
        NCP_OPT  = {"1 refeição": 1, "2 refeições": 2, "3 refeições": 3, "4 ou mais": 4}
        ncp      = float(NCP_OPT[st.selectbox("Nº de refeições principais por dia",
                                               list(NCP_OPT.keys()), index=2)])
        caec_pt  = st.selectbox("Petiscos entre refeições", list(FREQ.keys()))
        calc_pt  = st.selectbox("Consumo de álcool", list(FREQ.keys()))

    with col3:
        st.subheader("Estilo de Vida")
        smoke_pt  = st.selectbox("Fumante?", list(SIM_NAO.keys()))
        CH2O_OPT  = {"Menos de 1L": 1, "1 a 2L": 2, "Mais de 2L": 3}
        ch2o      = float(CH2O_OPT[st.selectbox("Consumo diário de água", list(CH2O_OPT.keys()),
                                                  index=1)])
        scc_pt    = st.selectbox("Monitora calorias?", list(SIM_NAO.keys()))
        FAF_OPT   = {"Nunca": 0, "1–2x por semana": 1, "3–4x por semana": 2, "5x ou mais": 3}
        faf       = float(FAF_OPT[st.selectbox("Atividade física por semana",
                                                list(FAF_OPT.keys()), index=1)])
        TUE_OPT   = {"0–2h por dia": 0, "3–5h por dia": 1, "Mais de 5h": 2}
        tue       = float(TUE_OPT[st.selectbox("Tempo de uso de tecnologia por dia",
                                                list(TUE_OPT.keys()))])
        mtrans_pt = st.selectbox("Transporte principal", list(TRANSP.keys()))

    submitted = st.form_submit_button("🔍 Realizar Previsão", use_container_width=True)


# ── Predição e exibição ───────────────────────────────────────────────────────
if submitted:
    input_data = {
        "gender":         GENERO[gender_pt],
        "age":            age,
        "family_history": SIM_NAO[family_history_pt],
        "favc":           SIM_NAO[favc_pt],
        "fcvc":           fcvc,
        "ncp":            ncp,
        "caec":           FREQ[caec_pt],
        "smoke":          SIM_NAO[smoke_pt],
        "ch2o":           ch2o,
        "scc":            SIM_NAO[scc_pt],
        "faf":            faf,
        "tue":            tue,
        "calc":           FREQ[calc_pt],
        "mtrans":         TRANSP[mtrans_pt],
    }

    # Spinner exclusivo da inferência do modelo
    with st.spinner("Calculando predição..."):
        predicted_class, probabilities = predict(input_data)

    max_prob = probabilities[predicted_class]
    label    = OBESITY_LABELS.get(predicted_class, predicted_class)

    st.success(f"**Resultado:** {label}  |  Confiança: **{max_prob:.1%}**")

    col_gauge, col_proba = st.columns([1, 1])

    with col_gauge:
        st.plotly_chart(build_gauge(predicted_class, max_prob), use_container_width=True)

    with col_proba:
        st.subheader("Probabilidade por Classe")
        sorted_proba = sorted(probabilities.items(), key=lambda x: OBESITY_ORDER.index(x[0]))
        for cls, prob in sorted_proba:
            lbl = OBESITY_LABELS.get(cls, cls)
            st.metric(lbl, f"{prob:.1%}")
            st.progress(prob)

    # Spinner exclusivo do log no Supabase — não bloqueia a exibição do resultado
    with st.spinner("Salvando no log..."):
        try:
            from db.supabase_client import get_client
            log_record = {**input_data, "predicted_class": predicted_class, "probability": max_prob}
            get_client().table("predictions_log").insert(log_record).execute()
            st.caption("✅ Predição salva no log.")
        except Exception as exc:
            st.caption(f"⚠️ Não foi possível salvar o log: {exc}")
