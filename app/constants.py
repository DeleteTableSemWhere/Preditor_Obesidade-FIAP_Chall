import plotly.express as px

PALETTE = px.colors.qualitative.Set2

OBESITY_LABELS = {
    "Insufficient_Weight": "Peso Insuficiente",
    "Normal_Weight":       "Peso Normal",
    "Overweight_Level_I":  "Sobrepeso Nível I",
    "Overweight_Level_II": "Sobrepeso Nível II",
    "Obesity_Type_I":      "Obesidade Tipo I",
    "Obesity_Type_II":     "Obesidade Tipo II",
    "Obesity_Type_III":    "Obesidade Tipo III",
}

OBESITY_LABELS_SHORT = {
    "Insufficient_Weight": "Peso Insuficiente",
    "Normal_Weight":       "Peso Normal",
    "Overweight_Level_I":  "Sobrepeso I",
    "Overweight_Level_II": "Sobrepeso II",
    "Obesity_Type_I":      "Obesidade I",
    "Obesity_Type_II":     "Obesidade II",
    "Obesity_Type_III":    "Obesidade III",
}

OBESITY_ORDER = list(OBESITY_LABELS.keys())
