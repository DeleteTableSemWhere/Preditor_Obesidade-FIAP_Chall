# [ObesityPredict: Inteligência Artificial para Triagem de Obesidade](https://preditor-obesidade-fiapchall-iouqvcxqrk6shsabazdbgz.streamlit.app/)

![Status do Projeto](https://img.shields.io/badge/Status-Finalizado-success)
![Acurácia](https://img.shields.io/badge/Acurácia_Teste-84%25-brightgreen)
![Python](https://img.shields.io/badge/Python-3.12%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)

## Sobre o Projeto

O **ObesityPredict** é um sistema de apoio à decisão médica desenvolvido para classificar pacientes em **7 níveis de obesidade** com base exclusivamente em **hábitos de vida e dados comportamentais** — sem utilizar altura ou peso como entrada.

Diferente de sistemas baseados em IMC, este projeto foca na **realidade clínica da triagem**: um médico já conhece o peso do paciente; o valor está em identificar risco de obesidade a partir de comportamentos modificáveis, permitindo intervenções mais eficazes antes do agravamento.

---

## 🚀 Principais Destaques

* **Exclusão Intencional de Altura e Peso:** Proxies diretos do IMC foram removidos para que o modelo aprenda `hábitos → classe`, não `IMC → classe`. Com 84% de acurácia sem biométricas, o sistema tem valor real de triagem comportamental.
* **Engenharia de Features Comportamentais:** Criação de `activity_hydration` (faf × ch2o) e `lifestyle_score` (faf + ch2o − ncp + fcvc) para capturar sinergias não-lineares entre hábitos.
* **Prevenção de Data Leakage:** Divisão estratificada (StratifiedKFold 5-folds + hold-out 20%) garante representatividade das 7 classes em cada fold sem contaminar o conjunto de teste.
* **Pipeline Completo em Produção:** Dados no Supabase → treinamento offline → artefatos versionados → inferência em tempo real via Streamlit Cloud.

Com o modelo **RandomForest** e engenharia de atributos refinada, atingimos resultados expressivos no conjunto de teste:

| Métrica | Resultado | Descrição |
| :--- | :--- | :--- |
| **Acurácia (Teste)** | **84,45%** | Taxa de acerto em dados inéditos, sem altura/peso. |
| **Macro-F1** | **83,43%** | Métrica primária para triagem clínica — penaliza erros em classes raras igualmente. |
| **ROC-AUC (Macro OvR)** | **0,9721** | Separabilidade entre as 7 classes sem variáveis biométricas. |
| **Acurácia CV (5-fold)** | **80,63% ± 1,49%** | Consistência em validação cruzada estratificada. |
| **Melhor modelo** | **RandomForest** | 150 árvores, max_depth=15, class_weight=balanced. |

---

## Engenharia de Dados e Modelagem

O projeto segue um pipeline linear e auditável:

### 1. Coleta de Dados (`Data Ingestion`)
* **Fonte:** Supabase PostgreSQL — tabela `obesity_data` (2.111 registros, paginação automática).
* **Origem:** UC Irvine ML Repository — *Estimation of Obesity Levels Based on Eating Habits and Physical Condition* (23% dados reais do México, Peru e Colômbia + 77% sintéticos via SMOTE).
* **Tratamento:** Normalização de nomes de colunas, renomeação de aliases históricos (`nobeyesdad`, `family_history_with_overweight`).

### 2. Engenharia de Features
Transformamos dados brutos em indicadores comportamentais preditivos com **blindagem total contra vazamento de dados**:

* **Atividade física:** FAF (dias/semana), TUE (tempo em telas).
* **Nutrição:** FCVC (frequência de vegetais), NCP (refeições/dia), CH2O (consumo de água), CAEC (lanches), CALC (álcool), FAVC (hipercalóricos).
* **Histórico:** Predisposição familiar, tabagismo, monitoramento calórico.
* **Interações engineeradas:** `activity_hydration = faf × ch2o`, `lifestyle_score = faf + ch2o − ncp + fcvc`.

### 3. Validação Temporal
* `StratifiedKFold(5)` garante que cada fold tenha as 7 classes representadas.
* Hold-out de 20% separado antes do treino para avaliação final ("Prova Real").
* Artefatos versionados por timestamp — mantidas as 2 versões mais recentes.

### 4. Modelagem e Avaliação
* **Pipeline sklearn:** `ColumnTransformer` (StandardScaler + OneHotEncoder) + `RandomForestClassifier`.
* **Métricas avaliadas:** Acurácia, Macro-F1, ROC-AUC (One-vs-Rest), Matriz de Confusão, Relatório por classe.
* **Artifact versioning:** `ml/model.pkl` + `ml/metrics.json` + snapshots `model_YYYYMMDD_HHMMSS.pkl`.

### 5. Modelos Avaliados

* **RandomForest (Melhor modelo — 84,45% de acurácia no hold-out)**
* Robusto a features correlacionadas; `class_weight="balanced"` compensa desbalanceamento residual entre as 7 classes.

---

## 📊 Resultados Alcançados

O modelo final classifica pacientes em 7 níveis sem balança ou fita métrica:

* **Acurácia em Teste:** 84,45% (dados comportamentais puros, sem altura/peso).
* **ROC-AUC:** 0,9721 — excelente separabilidade para problema multiclasse de 7 categorias.
* **Macro-F1:** 83,43% — critério primário para sistemas de triagem clínica.
* **Consistência:** Baixo overfitting (CV 80,63% vs hold-out 84,45%).

---

## Arquitetura do Sistema

```
app/          Frontend Streamlit (multi-página)
ml/           Pipeline de ML: treinamento, inferência, versionamento
db/           Cliente Supabase (singleton) + schema SQL
data/         CSV de treinamento (git-ignored)
```

**Fluxo de dados:**
- **Treino (offline):** `data/Obesity.csv` → Supabase `obesity_data` → `ml/train.py` → `ml/model.pkl` + `ml/metrics.json`
- **Inferência (runtime):** Formulário Streamlit → `ml/predict.py` (cached) → probabilidades por classe → log em `predictions_log`

---

## Executando o Projeto

### Pré-requisitos
* Python 3.12+
* Credenciais Supabase (`SUPABASE_URL` e `SUPABASE_KEY`)

### Instalação

1. Clone o repositório:
    ```bash
    git clone https://github.com/DeleteTableSemWhere/Preditor_Obesidade-FIAP_Chall.git
    cd Preditor_Obesidade-FIAP_Chall
    ```

2. Instale as dependências:
    ```bash
    pip install -r requirements.txt
    ```

3. Configure as credenciais (crie um `.env` na raiz):
    ```env
    SUPABASE_URL=sua_url_aqui
    SUPABASE_KEY=sua_chave_aqui
    ```

### Execução

```bash
# Ingerir dados (apenas na primeira vez)
python -m db.ingest_csv

# Treinar o modelo
python -m ml.train

# Rodar a aplicação
streamlit run app/Home.py
```

A aplicação estará disponível em `http://localhost:8501`.

### Docker

```bash
docker-compose up --build
```

---

## Páginas da Aplicação

| Página | Descrição |
| :--- | :--- |
| **Home** | Contexto do problema e navegação |
| **Análise** | Dashboard EDA: distribuições, correlações, balanço de classes |
| **Previsão** | Formulário de predição com gauge e probabilidades por classe |
| **Logs** | Histórico de predições com KPIs de uso |
| **Sobre** | Arquitetura, métricas, feature importance e limitações |

---

## Autores

Desenvolvido como parte do Tech Challenge (Fase 2) da Pós-Tech Data Analytics (FIAP).

<div align="center">
<table>
  <tr>
    <td align="center">
      <a href="https://github.com/BrunoAssis12">
        <img src="https://github.com/BrunoAssis12.png" width="100px;" alt=""/>
        <br /><sub><b>Bruno Assis</b></sub>
      </a><br />
      🚀 Data Scientist
    </td>
    <td align="center">
      <a href="https://github.com/gnunes-io">
        <img src="https://github.com/gnunes-io.png" width="100px;" alt=""/>
        <br /><sub><b>Gabriel Nunes</b></sub>
      </a><br />
      ⚙️ Data Architect
    </td>
    <td align="center">
      <a href="https://github.com/Jonathan-Paixao">
        <img src="https://github.com/Jonathan-Paixao.png" width="100px;" alt=""/>
        <br /><sub><b>Jonathan Paixão</b></sub>
      </a><br />
      🐍 Python Dev
    </td>
    <td align="center">
      <a href="https://github.com/rafaelvieiravidal-glitch">
        <img src="https://github.com/rafaelvieiravidal-glitch.png" width="100px;" alt=""/>
        <br /><sub><b>Rafael Vieira</b></sub>
      </a><br />
      📉 Quant
    </td>
    <td align="center">
      <a href="#">
        <img src="https://github.com/ghost.png" width="100px;" alt=""/>
        <br /><sub><b>Wagner da Silva</b></sub>
      </a><br />
      🧠 AI Engineer
    </td>
  </tr>
</table>
</div>

<br>

---

## 🛡️ Aviso Legal

Este projeto tem fins estritamente educacionais e acadêmicos. As classificações geradas pelo modelo de Machine Learning possuem margem de erro e **não constituem diagnóstico médico**. Qualquer resultado deve ser interpretado por um profissional de saúde qualificado com acesso ao histórico completo do paciente.
