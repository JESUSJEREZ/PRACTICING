import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor

# CONFIGURACIÓN
st.set_page_config(page_title="SST Predictivo Pro", layout="wide")

st.title("📊 Dashboard Predictivo SST - Nivel Profesional")

# =========================
# CARGA DE DATOS
# =========================
st.sidebar.header("📂 Cargar datos")
file = st.sidebar.file_uploader("Sube un archivo Excel o CSV", type=["csv", "xlsx"])

@st.cache_data
def load_data(file):
    if file.name.endswith(".csv"):
        df = pd.read_csv(file)
    else:
        df = pd.read_excel(file)
    return df

# Datos por defecto
if file:
    df = load_data(file)
else:
    df = pd.DataFrame({
        "Mes": np.arange(1, 13),
        "Accidentes": [5,7,6,8,9,12,11,10,13,15,14,16]
    })

st.subheader("📈 Datos")
st.dataframe(df)

# =========================
# MODELO ML
# =========================
if "Mes" in df.columns and "Accidentes" in df.columns:

    X = df[["Mes"]]
    y = df["Accidentes"]

    model = RandomForestRegressor(n_estimators=100)
    model.fit(X, y)

    future = np.arange(13, 19).reshape(-1,1)
    preds = model.predict(future)

    df_pred = pd.DataFrame({
        "Mes": future.flatten(),
        "Accidentes": preds
    })

    # =========================
    # INTERFAZ
    # =========================
    tab1, tab2 = st.tabs(["📊 Dashboard", "🤖 Predicción"])

    with tab1:
        st.subheader("Histórico")
        fig = px.line(df, x="Mes", y="Accidentes", markers=True)
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.subheader("Predicción")
        fig2 = px.line(df, x="Mes", y="Accidentes", markers=True)
        fig2.add_scatter(
            x=df_pred["Mes"],
            y=df_pred["Accidentes"],
            mode="lines+markers",
            name="Predicción"
        )
        st.plotly_chart(fig2, use_container_width=True)

    # KPI simple
    st.metric("Accidentes promedio", round(df["Accidentes"].mean(), 2))

    # ALERTA
    if df["Accidentes"].mean() > 10:
        st.warning("⚠️ Riesgo alto de accidentalidad")
    else:
        st.success("✅ Riesgo controlado")

else:
    st.error("El archivo debe contener columnas: Mes y Accidentes")
