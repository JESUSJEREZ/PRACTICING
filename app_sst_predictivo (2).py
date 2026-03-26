
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve, ConfusionMatrixDisplay
)
import warnings
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────
st.set_page_config(
    page_title="SST Predictivo | Modelo de Riesgo",
    page_icon="🦺",
    layout="wide",
    initial_sidebar_state="expanded",
)

COLORS = {
    "primary": "#1E3A5F",
    "secondary": "#2E86AB",
    "danger":   "#E63946",
    "warning":  "#F4A261",
    "success":  "#2A9D8F",
    "light":    "#F8F9FA",
}

st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1E3A5F 0%, #2E86AB 100%);
        color: white; padding: 1.5rem 2rem; border-radius: 12px;
        margin-bottom: 1.5rem;
    }
    .kpi-card {
        background: white; border-radius: 10px; padding: 1.2rem;
        border: 1px solid #e0e0e0; text-align: center;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    }
    .kpi-value { font-size: 2rem; font-weight: 700; margin: 0; }
    .kpi-label { color: #666; font-size: 0.85rem; margin: 0; }
    .risk-high   { color: #E63946; }
    .risk-medium { color: #F4A261; }
    .risk-low    { color: #2A9D8F; }
    .section-title {
        font-size: 1.1rem; font-weight: 600; color: #1E3A5F;
        border-left: 4px solid #2E86AB; padding-left: 0.8rem;
        margin: 1.5rem 0 1rem 0;
    }
    .pred-box {
        padding: 1.5rem; border-radius: 12px; text-align: center;
        margin: 1rem 0;
    }
    .pred-box h2 { margin: 0; font-size: 1.8rem; }
    .pred-box p  { margin: 0.3rem 0 0; font-size: 0.9rem; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────
# DATA LOADER
# ─────────────────────────────────────────
@st.cache_data
def load_data(uploaded_file=None):
    if uploaded_file is not None:
        raw = pd.read_excel(uploaded_file, sheet_name="PREVSIS", skiprows=1)
    else:
        # Datos sintéticos de demostración basados en la distribución real
        np.random.seed(42)
        n = 1000
        cargos = ["Tripulante de Cabina Nacional","Agente de Servicio al Cliente",
                  "Tripulante de Cabina Internacional/Nacional","Jefe de Cabina Nacional",
                  "Aprendiz","Agente de Rampa","Tecnico IV de Mantenimiento Linea",
                  "Auxiliar de carga","Coordinador de Operaciones","Piloto"]
        naturalezas = ["1.Golpe, Contusión o Aplastamiento","1.Lesión Lumbar",
                       "1.Dolor Otico","1.Torcedura","1.Herida","1.Desgarro Muscular",
                       "1.Esguince","1.Otros"]
        mecanismos = ["1.Caída de Personas a Nivel de Piso (Resbalon o tropezon)",
                      "1.Golpes contra objetos","1.Golpes por objetos",
                      "1.Sobre-esfuerzo","1.Falso Movimiento","1.Atrapamiento",
                      "1.Caída de Objetos","1.Otros"]
        agentes = ["1.Movimiento del cuerpo","1.Herramientas, Maquinarias, Implementos o Utensilios",
                   "1.Aeronave y sus componentes","1.Equipaje","1.Ambiente de trabajo",
                   "1.Cambios de presión brusca","1.Medio de transporte","1.Otros"]
        estaciones = ["Bogotá","Rionegro","Cali","El Salvador","San Jose de Costa Rica",
                      "Medellín","Madrid","Lima"]
        partes = ["1.Espalda","1.Oreja (oido)","1.Manos","1.Dedos de mano","1.Pie",
                  "Múltiples partes","1.Cabeza","1.Lesiones generales u otros"]
        generos = ["Femenino","Masculino","Sin informacion"]
        clasificaciones = ["Leves","Severos","Graves"]
        tipos = ["Propio del Trabajo","Tránsito","Itinere/Fuera de Jornada Laboral",
                 "Violencia","Otro"]

        antic = np.random.choice([2,3,4,5,8,10,13,15,20,"Sin información"], size=n,
                                 p=[0.15,0.11,0.07,0.05,0.05,0.03,0.04,0.03,0.02,0.45])

        dias_inc = np.zeros(n, dtype=int)
        for i in range(n):
            r = np.random.random()
            dias_inc[i] = 0 if r < 0.43 else (3 if r < 0.57 else (5 if r < 0.65 else
                          np.random.randint(6, 30)))

        data = {
            "ID Reporte": range(9500, 9500+n),
            "Fecha de ocurrencia": pd.date_range("2024-01-01", periods=n, freq="8h"),
            "Mes": np.random.choice(["Jan 2024","Feb 2024","Mar 2024","Apr 2024",
                                     "May 2024","Jun 2024","Jul 2024","Aug 2024",
                                     "Sep 2024","Oct 2024","Nov 2024","Dec 2024",
                                     "Jan 2025","Feb 2025","Mar 2025"], size=n),
            "Clasificación del accidente": np.random.choice(clasificaciones, size=n, p=[0.93,0.05,0.02]),
            "Tipo de accidente": np.random.choice(tipos, size=n, p=[0.87,0.05,0.03,0.03,0.02]),
            "Estación donde ocurrió": np.random.choice(estaciones, size=n,
                                      p=[0.40,0.11,0.06,0.06,0.06,0.05,0.03,0.23]),
            "Género": np.random.choice(generos, size=n, p=[0.42,0.27,0.31]),
            "Puesto/Cargo": np.random.choice(cargos, size=n,
                            p=[0.28,0.10,0.04,0.03,0.03,0.03,0.03,0.03,0.02,0.41]), # Corrected probability here
            "Antigüedad en años": antic,
            "Naturaleza": np.random.choice(naturalezas, size=n,
                          p=[0.39,0.09,0.07,0.06,0.04,0.04,0.03,0.28]),
            "Parte del cuerpo afectada": np.random.choice(partes, size=n,
                                          p=[0.11,0.10,0.09,0.08,0.07,0.07,0.06,0.42]),
            "Agente de la lesión": np.random.choice(agentes, size=n,
                                    p=[0.16,0.15,0.09,0.06,0.06,0.04,0.04,0.40]),
            "Mecanismo de accidente/ Tipo de contacto": np.random.choice(mecanismos, size=n,
                                    p=[0.12,0.08,0.07,0.05,0.05,0.05,0.05,0.53]),
            "Total días de incapacidad": dias_inc,
        }
        return pd.DataFrame(data)

    raw.columns = raw.iloc[0]
    raw = raw.drop(index=0).reset_index(drop=True)
    return raw


@st.cache_data
def prepare_model_data(df):
    cols = {
        "Clasificación del accidente": "clasificacion",
        "Tipo de accidente": "tipo",
        "Estación donde ocurrió": "estacion",
        "Género": "genero",
        "Puesto/Cargo": "cargo",
        "Antigüedad en años": "antiguedad",
        "Naturaleza": "naturaleza",
        "Parte del cuerpo afectada": "parte_cuerpo",
        "Agente de la lesión": "agente",
        "Mecanismo de accidente/ Tipo de contacto": "mecanismo",
        "Total días de incapacidad": "dias_incapacidad",
    }
    available = {k: v for k, v in cols.items() if k in df.columns}
    mdf = df[list(available.keys())].copy()
    mdf.columns = list(available.values())

    if "dias_incapacidad" in mdf.columns:
        mdf["dias_incapacidad"] = pd.to_numeric(mdf["dias_incapacidad"], errors="coerce").fillna(0)
    else:
        mdf["dias_incapacidad"] = 0

    if "antiguedad" in mdf.columns:
        mdf["antiguedad"] = pd.to_numeric(
            mdf["antiguedad"].replace("Sin información", np.nan), errors="coerce"
        ).fillna(mdf["antiguedad"].replace("Sin información", np.nan).apply(
            pd.to_numeric, errors="coerce").median()
        ).fillna(5)

    mdf["con_incapacidad"] = (mdf["dias_incapacidad"] > 0).astype(int)
    mdf["incapacidad_alta"] = (mdf["dias_incapacidad"] >= 5).astype(int)

    cat_cols = [c for c in mdf.columns if mdf[c].dtype == object]
    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        mdf[col] = mdf[col].fillna("Sin información").astype(str)
        mdf[col] = le.fit_transform(mdf[col])
        encoders[col] = le

    return mdf, encoders


@st.cache_resource
def train_model(mdf, target="con_incapacidad", model_type="Random Forest"):
    features = [c for c in mdf.columns if c not in
                ["con_incapacidad","incapacidad_alta","dias_incapacidad"]]
    X = mdf[features]
    y = mdf[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    if model_type == "Random Forest":
        model = RandomForestClassifier(n_estimators=200, max_depth=8,
                                       class_weight="balanced", random_state=42, n_jobs=-1)
    elif model_type == "Gradient Boosting":
        model = GradientBoostingClassifier(n_estimators=150, max_depth=5,
                                           learning_rate=0.05, random_state=42)
    else:
        model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    cv_scores = cross_val_score(model, X, y, cv=5, scoring="roc_auc")

    return model, X_train, X_test, y_train, y_test, y_pred, y_prob, cv_scores, features


# ─────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/8/80/Avianca_logo.svg/320px-Avianca_logo.svg.png",
             width=140)
    st.markdown("### ⚙️ Configuración")

    uploaded = st.file_uploader("📂 Cargar DATA_2025.xlsx", type=["xlsx"])
    model_type = st.selectbox("🤖 Algoritmo", ["Random Forest","Gradient Boosting","Regresión Logística"])
    target_var = st.selectbox("🎯 Variable objetivo",
        ["con_incapacidad", "incapacidad_alta"],
        format_func=lambda x: "¿Tendrá incapacidad?" if x == "con_incapacidad"
                              else "¿Incapacidad ≥5 días?")

    st.markdown("---")
    st.markdown("### 🔍 Predicción individual")
    st.caption("Ingresa datos de un caso nuevo:")

    cargo_opts = ["Tripulante de Cabina Nacional","Agente de Servicio al Cliente",
                  "Jefe de Cabina Nacional","Agente de Rampa","Aprendiz",
                  "Tecnico IV de Mantenimiento Linea","Auxiliar de carga","Piloto","Otro"]
    pred_cargo      = st.selectbox("Cargo", cargo_opts)
    pred_genero     = st.selectbox("Género", ["Femenino","Masculino","Sin informacion"])
    pred_antiguedad = st.slider("Antigüedad (años)", 0, 30, 3)
    pred_naturaleza = st.selectbox("Naturaleza lesión", [
        "1.Golpe, Contusión o Aplastamiento","1.Lesión Lumbar","1.Dolor Otico",
        "1.Torcedura","1.Herida","1.Desgarro Muscular","1.Esguince","1.Otros"])
    pred_mecanismo  = st.selectbox("Mecanismo", [
        "1.Caída de Personas a Nivel de Piso (Resbalon o tropezon)",
        "1.Golpes contra objetos","1.Golpes por objetos","1.Sobre-esfuerzo",
        "1.Falso Movimiento","1.Atrapamiento","1.Otros"])
    pred_agente     = st.selectbox("Agente de lesión", [
        "1.Movimiento del cuerpo","1.Herramientas, Maquinarias, Implementos o Utensilios",
        "1.Aeronave y sus componentes","1.Equipaje","1.Ambiente de trabajo","1.Otros"])
    pred_parte      = st.selectbox("Parte del cuerpo", [
        "1.Espalda","1.Oreja (oido)","1.Manos","1.Dedos de mano","1.Pie",
        "Múltiples partes","1.Cabeza","1.Lesiones generales u otros"])
    pred_estacion   = st.selectbox("Estación", ["Bogotá","Rionegro","Cali","Medellín","Otros"])
    pred_tipo       = st.selectbox("Tipo accidente", ["Propio del Trabajo","Tránsito","Otro"])
    pred_clasif     = st.selectbox("Clasificación", ["Leves","Severos","Graves"])


# ─────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────
df_raw = load_data(uploaded)
mdf, encoders = prepare_model_data(df_raw)
model, X_train, X_test, y_train, y_test, y_pred, y_prob, cv_scores, feat_cols =     train_model(mdf, target=target_var, model_type=model_type)

# ─────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1 style="margin:0;font-size:1.8rem;">🦺 SST Predictivo — Modelo de Riesgo de Incapacidad</h1>
    <p style="margin:0.3rem 0 0;opacity:0.85;font-size:0.95rem;">
        Análisis exploratorio + Machine Learning sobre datos reales de accidentalidad PREVSIS
    </p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
# TABS
# ─────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["📊 Panorama", "🤖 Modelo ML", "🔮 Predicción", "📋 Datos"])


# ══════════════════════════════════════════
# TAB 1: PANORAMA
# ══════════════════════════════════════════
with tab1:
    # KPIs
    total = len(df_raw)
    total_incap_col = "Total días de incapacidad"
    if total_incap_col in df_raw.columns:
        dias_num = pd.to_numeric(df_raw[total_incap_col], errors="coerce").fillna(0)
        pct_incap = (dias_num > 0).mean() * 100
        avg_dias  = dias_num[dias_num > 0].mean()
    else:
        pct_incap, avg_dias = 0, 0

    graves_col = "Clasificación del accidente"
    if graves_col in df_raw.columns:
        n_graves = (df_raw[graves_col].astype(str).str.strip() == "Graves").sum()
    else:
        n_graves = 0

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""<div class="kpi-card">
            <p class="kpi-value" style="color:#1E3A5F;">{total:,}</p>
            <p class="kpi-label">Total incidentes</p></div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class="kpi-card">
            <p class="kpi-value risk-warning">{pct_incap:.1f}%</p>
            <p class="kpi-label">Con incapacidad</p></div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div class="kpi-card">
            <p class="kpi-value risk-medium">{avg_dias:.1f}</p>
            <p class="kpi-label">Días prom. de incapacidad</p></div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""<div class="kpi-card">
            <p class="kpi-value risk-high">{n_graves}</p>
            <p class="kpi-label">Accidentes graves</p></div>""", unsafe_allow_html=True)

    st.markdown('<p class="section-title">Distribución por variables clave</p>', unsafe_allow_html=True)
    col_a, col_b = st.columns(2)

    with col_a:
        if "Clasificación del accidente" in df_raw.columns:
            vc = df_raw["Clasificación del accidente"].value_counts().reset_index()
            vc.columns = ["Clasificación","Cantidad"]
            fig = px.pie(vc, values="Cantidad", names="Clasificación",
                         color_discrete_sequence=["#2E86AB","#F4A261","#E63946","#aaa"],
                         title="Clasificación del accidente")
            fig.update_layout(margin=dict(t=40,b=10))
            st.plotly_chart(fig, use_container_width=True)

    with col_b:
        if "Tipo de accidente" in df_raw.columns:
            vc2 = df_raw["Tipo de accidente"].value_counts().head(6).reset_index()
            vc2.columns = ["Tipo","Cantidad"]
            fig2 = px.bar(vc2, x="Cantidad", y="Tipo", orientation="h",
                          color="Cantidad", color_continuous_scale="Blues",
                          title="Tipo de accidente (Top 6)")
            fig2.update_layout(margin=dict(t=40,b=10), coloraxis_showscale=False)
            st.plotly_chart(fig2, use_container_width=True)

    col_c, col_d = st.columns(2)
    with col_c:
        if "Naturaleza" in df_raw.columns:
            vc3 = df_raw["Naturaleza"].value_counts().head(8).reset_index()
            vc3.columns = ["Naturaleza","Cantidad"]
            vc3["Naturaleza"] = vc3["Naturaleza"].str.replace(r"^\d+\. ", "", regex=True).str.strip()
            fig3 = px.bar(vc3, x="Naturaleza", y="Cantidad",
                          color="Cantidad", color_continuous_scale="Teal",
                          title="Naturaleza de la lesión")
            fig3.update_layout(margin=dict(t=40,b=80), xaxis_tickangle=-35,
                               coloraxis_showscale=False)
            st.plotly_chart(fig3, use_container_width=True)

    with col_d:
        if "Parte del cuerpo afectada" in df_raw.columns:
            vc4 = df_raw["Parte del cuerpo afectada"].value_counts().head(8).reset_index()
            vc4.columns = ["Parte","Cantidad"]
            vc4["Parte"] = vc4["Parte"].str.replace(r"^\d+\. ", "", regex=True).str.strip()
            fig4 = px.bar(vc4, x="Parte", y="Cantidad",
                          color="Cantidad", color_continuous_scale="Oranges",
                          title="Parte del cuerpo afectada")
            fig4.update_layout(margin=dict(t=40,b=80), xaxis_tickangle=-35,
                               coloraxis_showscale=False)
            st.plotly_chart(fig4, use_container_width=True)

    if "Puesto/Cargo" in df_raw.columns and "Total días de incapacidad" in df_raw.columns:
        st.markdown('<p class="section-title">Días de incapacidad promedio por cargo (Top 10)</p>',
                    unsafe_allow_html=True)
        tmp = df_raw[["Puesto/Cargo","Total días de incapacidad"]].copy()
        tmp["Total días de incapacidad"] = pd.to_numeric(tmp["Total días de incapacidad"], errors="coerce")
        avg_cargo = (tmp.groupby("Puesto/Cargo")["Total días de incapacidad"]
                     .mean().sort_values(ascending=False).head(10).reset_index())
        avg_cargo.columns = ["Cargo","Días promedio"]
        fig5 = px.bar(avg_cargo, x="Días promedio", y="Cargo", orientation="h",
                      color="Días promedio", color_continuous_scale="RdYlGn_r",
                      title="Riesgo relativo por cargo — días promedio de incapacidad")
        fig5.update_layout(margin=dict(t=40,b=10), coloraxis_showscale=False, height=360)
        st.plotly_chart(fig5, use_container_width=True)


# ══════════════════════════════════════════
# TAB 2: MODELO ML
# ══════════════════════════════════════════
with tab2:
    rep = classification_report(y_test, y_pred, output_dict=True)
    auc = roc_auc_score(y_test, y_prob)

    st.markdown('<p class="section-title">Métricas del modelo</p>', unsafe_allow_html=True)
    m1, m2, m3, m4, m5 = st.columns(5)
    for col_w, label, val, color in [
        (m1, "AUC-ROC",    f"{auc:.3f}",                          "#1E3A5F"),
        (m2, "CV Score",   f"{cv_scores.mean():.3f} ±{cv_scores.std():.3f}", "#2E86AB"),
        (m3, "Accuracy",   f"{rep['accuracy']:.2%}",              "#2A9D8F"),
        (m4, "Precisión",  f"{rep['1']['precision']:.2%}",        "#F4A261"),
        (m5, "Recall",     f"{rep['1']['recall']:.2%}",           "#E63946"),
    ]:
        col_w.markdown(f"""<div class="kpi-card">
            <p class="kpi-value" style="color:{color};">{val}</p>
            <p class="kpi-label">{label}</p></div>""", unsafe_allow_html=True)

    st.caption(f"Modelo: **{model_type}** · Entrenado con {len(X_train):,} casos · Validado con {len(X_test):,} casos")

    col_roc, col_cm = st.columns(2)

    with col_roc:
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines",
            line=dict(color="#2E86AB", width=2.5),
            name=f"ROC (AUC={auc:.3f})"))
        fig_roc.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines",
            line=dict(color="#aaa", dash="dash"), name="Aleatorio"))
        fig_roc.update_layout(title="Curva ROC", xaxis_title="FPR",
            yaxis_title="TPR", height=350, margin=dict(t=40,b=30))
        st.plotly_chart(fig_roc, use_container_width=True)

    with col_cm:
        cm = confusion_matrix(y_test, y_pred)
        labels = ["Sin incapacidad","Con incapacidad"]
        fig_cm = px.imshow(cm, text_auto=True, color_continuous_scale="Blues",
            x=labels, y=labels,
            labels=dict(x="Predicho", y="Real", color="Casos"),
            title="Matriz de confusión")
        fig_cm.update_layout(height=350, margin=dict(t=40,b=30))
        st.plotly_chart(fig_cm, use_container_width=True)

    if hasattr(model, "feature_importances_"):
        st.markdown('<p class="section-title">Importancia de variables (Top 10)</p>',
                    unsafe_allow_html=True)
        fi = pd.DataFrame({"Variable": feat_cols,
                           "Importancia": model.feature_importances_})
        fi = fi.sort_values("Importancia", ascending=False).head(10)
        label_map = {
            "naturaleza":"Naturaleza lesión","parte_cuerpo":"Parte del cuerpo",
            "agente":"Agente lesión","mecanismo":"Mecanismo","cargo":"Cargo",
            "genero":"Género","antiguedad":"Antigüedad","estacion":"Estación",
            "tipo":"Tipo accidente","clasificacion":"Clasificación",
        }
        fi["Variable"] = fi["Variable"].map(label_map).fillna(fi["Variable"])
        fig_fi = px.bar(fi, x="Importancia", y="Variable", orientation="h",
                        color="Importancia", color_continuous_scale="Blues",
                        title="¿Qué variables predicen mejor la incapacidad?")
        fig_fi.update_layout(height=370, margin=dict(t=40,b=20),
                             coloraxis_showscale=False)
        st.plotly_chart(fig_fi, use_container_width=True)

    with st.expander("📄 Reporte completo de clasificación"):
        rep_df = pd.DataFrame(rep).T.round(3)
        st.dataframe(rep_df, use_container_width=True)


# ══════════════════════════════════════════
# TAB 3: PREDICCIÓN INDIVIDUAL
# ══════════════════════════════════════════
with tab3:
    st.markdown('<p class="section-title">Predicción de riesgo para un caso nuevo</p>',
                unsafe_allow_html=True)
    st.info("Ajusta los parámetros en el panel izquierdo y haz clic en **Predecir**.")

    predict_btn = st.button("🔮 Calcular riesgo de incapacidad", type="primary", use_container_width=True)

    if predict_btn:
        input_dict = {
            "clasificacion": pred_clasif,
            "tipo":           pred_tipo,
            "estacion":       pred_estacion,
            "genero":         pred_genero,
            "cargo":          pred_cargo,
            "antiguedad":     float(pred_antiguedad),
            "naturaleza":     pred_naturaleza,
            "parte_cuerpo":   pred_parte,
            "agente":         pred_agente,
            "mecanismo":      pred_mecanismo,
        }

        input_row = {}
        for col in feat_cols:
            if col == "antiguedad":
                input_row[col] = float(pred_antiguedad)
            elif col in input_dict and col in encoders:
                le = encoders[col]
                val = input_dict[col]
                if val in le.classes_:
                    input_row[col] = le.transform([val])[0]
                else:
                    input_row[col] = le.transform([le.classes_[0]])[0]
            else:
                input_row[col] = 0

        X_new = pd.DataFrame([input_row])
        prob = model.predict_proba(X_new)[0][1]
        nivel = "Alto" if prob >= 0.6 else ("Medio" if prob >= 0.35 else "Bajo")
        color_bg = {"Alto":"#FFE5E5","Medio":"#FFF3E0","Bajo":"#E5F7F4"}[nivel]
        color_tx = {"Alto":"#E63946","Medio":"#F4A261","Bajo":"#2A9D8F"}[nivel]
        icon     = {"Alto":"🔴","Medio":"🟡","Bajo":"🟢"}[nivel]

        st.markdown(f"""
        <div class="pred-box" style="background:{color_bg}; border:2px solid {color_tx};">
            <h2 style="color:{color_tx};">{icon} Riesgo {nivel}</h2>
            <p style="color:{color_tx};font-size:1.4rem;font-weight:700;">{prob:.1%} de probabilidad de incapacidad</p>
            <p style="color:#555;">Variable objetivo: {target_var.replace('_',' ')}</p>
        </div>""", unsafe_allow_html=True)

        gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=round(prob * 100, 1),
            number={"suffix": "%", "font": {"size": 36}},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": color_tx},
                "steps": [
                    {"range": [0, 35],  "color": "#E5F7F4"},
                    {"range": [35, 60], "color": "#FFF3E0"},
                    {"range": [60, 100],"color": "#FFE5E5"},
                ],
                "threshold": {"line": {"color": color_tx, "width": 4},
                              "thickness": 0.8, "value": round(prob * 100, 1)},
            },
            title={"text": "Índice de riesgo de incapacidad"},
        ))
        gauge.update_layout(height=300, margin=dict(t=40,b=10))
        st.plotly_chart(gauge, use_container_width=True)

        st.markdown("#### 💡 Recomendaciones automáticas")
        recs = []
        if "Lesión Lumbar" in pred_naturaleza or "Sobre-esfuerzo" in pred_mecanismo:
            recs.append("⚠️ **Riesgo ergonómico detectado**: revisar técnica de levantamiento y dotación de fajas.")
        if "Espalda" in pred_parte or "Manos" in pred_parte:
            recs.append("🏥 **Parte del cuerpo crítica**: priorizar evaluación médica inmediata y seguimiento.")
        if pred_antiguedad <= 2:
            recs.append("📚 **Trabajador nuevo**: reforzar inducción SST y acompañamiento durante primeros 6 meses.")
        if "Turbulencia" in pred_agente or "Aeronave" in pred_agente:
            recs.append("✈️ **Riesgo aeronáutico**: verificar procedimientos de aseguramiento durante maniobras.")
        if "Tránsito" in pred_tipo:
            recs.append("🚗 **Accidente de tránsito**: validar adherencia a política de conducción segura.")
        if nivel == "Alto":
            recs.append("🚨 **Nivel ALTO**: notificar inmediatamente a ARL y activar protocolo de investigación.")
        if not recs:
            recs.append("✅ Perfil de riesgo moderado. Mantener controles SST habituales y seguimiento periódico.")
        for r in recs:
            st.markdown(r)


# ══════════════════════════════════════════
# TAB 4: DATOS
# ══════════════════════════════════════════
with tab4:
    st.markdown('<p class="section-title">Datos fuente</p>', unsafe_allow_html=True)
    col_show = [c for c in [
        "ID Reporte","Fecha de ocurrencia","Mes","Clasificación del accidente",
        "Tipo de accidente","Género","Puesto/Cargo","Antigüedad en años",
        "Naturaleza","Parte del cuerpo afectada","Total días de incapacidad"
    ] if c in df_raw.columns]
    st.dataframe(df_raw[col_show].head(200), use_container_width=True, height=420)

    buf = BytesIO()
    df_raw[col_show].to_excel(buf, index=False)
    st.download_button("⬇️ Descargar datos filtrados (.xlsx)",
                       data=buf.getvalue(),
                       file_name="sst_datos_filtrados.xlsx",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    st.markdown("---")
    st.markdown(f"""
    **📦 Stack tecnológico:**
    `Python 3.x` · `Streamlit` · `Scikit-learn ({model_type})` · `Plotly` · `Pandas`

    **🎯 Variable objetivo:** `{target_var}` — clasifica si un incidente derivará en incapacidad laboral.

    **📐 Features usados:** {", ".join(feat_cols)}
    """)
