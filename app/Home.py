"""
Preditor de Obesidade – Página inicial.
Executar com: streamlit run app/Home.py
"""
import streamlit as st

st.set_page_config(
    page_title="Preditor de Obesidade",
    page_icon="⚕️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Estilos ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.hero-title   { font-size: 2.8em; font-weight: 800; margin-bottom: 0.1em; }
.hero-sub     { font-size: 1.2em; color: #555; margin-bottom: 1.5em; }
.card         { border: 1px solid #3a3a4f; border-radius: 12px;
                padding: 1.2em 1.4em; height: 100%;
                background-color: #1E1E2E; color: white; }
.card-icon    { font-size: 2em; }
.card-title   { font-size: 1.1em; font-weight: 700; margin: 0.3em 0 0.2em; color: white; }
.card-desc    { font-size: 0.9em; color: #aab; }
.footer       { margin-top: 3em; text-align: center;
                font-size: 0.8em; color: #aaa; }
</style>
""", unsafe_allow_html=True)

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown('<div class="hero-title">⚕️ Preditor de Obesidade</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-sub">Triagem inteligente de níveis de obesidade com base em hábitos de vida e dados físicos</div>',
    unsafe_allow_html=True,
)

st.markdown("""
Segundo o Vigitel 2025 (Ministério da Saúde), a obesidade cresceu 118% no Brasil entre 2006 e 2024.
1 em cada 3 brasileiros vive com obesidade e 68% da população tem excesso de peso. A triagem precoce baseada em hábitos de vida permite intervenções mais eficazes antes do agravamento clínico. 
Este sistema usa Machine Learning para classificar pacientes em 7 níveis de obesidade em menos de 1 segundo, apoiando médicos e equipes de saúde na triagem inicial.
""")

st.markdown("""
<a href="/Previsão" target="_self">
  <button style="
    background-color: #E74C3C;
    color: white;
    font-size: 1.2em;
    font-weight: 700;
    padding: 0.7em 2em;
    border: none;
    border-radius: 8px;
    cursor: pointer;
    margin-top: 1em;
  ">Iniciar Previsão</button>
</a>
""", unsafe_allow_html=True)

st.divider()

# ── Cards de navegação ────────────────────────────────────────────────────────
st.subheader("Explore o sistema")

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.markdown("""
    <div class="card">
      <div class="card-icon">📊</div>
      <div class="card-title">Análise</div>
      <div class="card-desc">Distribuições, correlações e dicionário completo
      do dataset de treinamento.</div>
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown("""
    <div class="card">
      <div class="card-icon">🔬</div>
      <div class="card-title">Previsão (ML)</div>
      <div class="card-desc">Insira os dados do paciente e obtenha a predição
      instantânea com nível de confiança.</div>
    </div>
    """, unsafe_allow_html=True)

with c3:
    st.markdown("""
    <div class="card">
      <div class="card-icon">📋</div>
      <div class="card-title">Logs</div>
      <div class="card-desc">Histórico de todas as predições realizadas
      com KPIs de uso do sistema.</div>
    </div>
    """, unsafe_allow_html=True)

with c4:
    st.markdown("""
    <div class="card">
      <div class="card-icon">📖</div>
      <div class="card-title">Sobre o Projeto</div>
      <div class="card-desc">Arquitetura, métricas do modelo, limitações
      e contexto técnico completo.</div>
    </div>
    """, unsafe_allow_html=True)

# ── Rodapé ────────────────────────────────────────────────────────────────────
st.markdown(
    '<div class="footer">Projeto acadêmico · POSTECH FIAP Data Analytics · 2026</div>',
    unsafe_allow_html=True,
)
