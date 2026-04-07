# =============================================================================
# SST PREDICTIVO — Modelo de Riesgo de Incapacidad Laboral
# Basado en datos PREVSIS (accidentalidad real 2024-2026)
#
# INSTALACIÓN:
#   pip install streamlit scikit-learn pandas plotly openpyxl
#
# EJECUCIÓN LOCAL:
#   streamlit run app_sst_predictivo.py
# =============================================================================

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)

# =============================================================================
# CONFIGURACION DE PAGINA
# =============================================================================
st.set_page_config(
    page_title="SST Predictivo | Riesgo de Incapacidad",
    page_icon="🦺",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .block-container { padding-top: 1.5rem; }
    .kpi-card {
        background: white;
        border-radius: 10px;
        padding: 1.1rem 1rem;
        border: 1px solid #e0e0e0;
        text-align: center;
    }
    .kpi-value { font-size: 2rem; font-weight: 700; margin: 0; }
    .kpi-label { color: #666; font-size: 0.82rem; margin: 0.2rem 0 0; }
    .section-title {
        font-size: 1.05rem;
        font-weight: 600;
        color: #1E3A5F;
        border-left: 4px solid #2E86AB;
        padding-left: 0.75rem;
        margin: 1.5rem 0 0.8rem;
    }
    .pred-box {
        padding: 1.4rem;
        border-radius: 12px;
        text-align: center;
        margin: 0.8rem 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# =============================================================================
# CONSTANTES — VALORES REALES DEL ARCHIVO PREVSIS
# =============================================================================

CARGOS = [
    "Tripulante de Cabina Nacional",
    "Agente de Servicio al Cliente",
    "Tripulante de Cabina Internacional/Nacional",
    "Jefe de Cabina Nacional",
    "Aprendiz",
    "Tecnico IV de Mantenimiento Linea",
    "Agente de Rampa",
    "Auxiliar de carga",
    "Copiloto A320",
    "Tripulante Cabina de Pasajeros Regions",
]

NATURALEZAS = [
    "1.Golpe, Contusion o Aplastamiento",
    "1.Otros",
    "1.Lesion Lumbar",
    "1.Dolor Otico",
    "1.Torcedura",
    "1.Herida",
    "1.Desgarro Muscular",
    "1.Esguince",
    "1.Trauma Superficial (cortada, rasguño, puncion, pinchazo)",
    "1.Contacto / Irritacion con Sustancias Quimicas",
]

MECANISMOS = [
    "1.Caida de Personas a Nivel de Piso  (Resbalon o tropezon )",
    "1.Otros",
    "1.Golpes contra objetos",
    "1.Golpes por objetos",
    "1.Falso Movimiento",
    "1.Sobre-esfuerzo (levantar objetos, halar, empujar, manipular o lanzar objetos)",
    "1.Caida de Objetos",
    "1.Atrapamiento",
    "1.Contacto con Herramientas o Superficies Cortantes y/o punzantes",
    "1.Aterrizaje Anormal (Hard Landing)",
]

AGENTES = [
    "1.Movimiento del cuerpo",
    "1.Herramientas, Maquinarias, Implementos o Utensilios",
    "1.Otros",
    "1.Aeronave y sus componentes",
    "1.Equipaje",
    "1.Ambiente de trabajo",
    "1.Cambios de presion brusca",
    "1.Medio de transporte",
    "1.Medios de transporte de colaboradores",
    "1.Equipo de Tierra",
]

PARTES = [
    "1.Espalda",
    "1.Oreja (oido)",
    "1.Manos",
    "1.Dedos de mano",
    "1.Pie",
    "1.Multiples partes",
    "1.Cabeza",
    "1.Lesiones generales u otros",
    "1.Miembros superiores",
    "1.Miembros inferiores",
]

ESTACIONES = [
    "Bogota", "Rionegro", "El Salvador", "Cali",
    "San Jose de Costa Rica", "Medellin", "Madrid",
    "Barranquilla", "Cartagena", "Quito",
]

TIPOS = [
    "Propio del Trabajo",
    "Transito",
    "Itinere/Fuera de Jornada Laboral",
    "Violencia",
    "Otro",
    "Biologico",
    "Recreativo o Cultural",
    "Deportivo",
]

CLASIFICACIONES = ["Leves", "Severos", "Graves"]
GENEROS = ["Femenino", "Masculino", "Sin informacion"]

KEY_COLS = [
    "Clasificacion del accidente",
    "Tipo de accidente",
    "Estacion donde ocurrio",
    "Genero",
    "Puesto/Cargo",
    "Antiguedad en anos",
    "Naturaleza",
    "Parte del cuerpo afectada",
    "Agente de la lesion",
    "Mecanismo de accidente/ Tipo de contacto",
    "Total dias de incapacidad",
]

COL_RENAME_ORIG = {
    "Clasificación del accidente":              "Clasificacion del accidente",
    "Tipo de accidente":                        "Tipo de accidente",
    "Estación donde ocurrió":                   "Estacion donde ocurrio",
    "Género":                                   "Genero",
    "Puesto/Cargo":                             "Puesto/Cargo",
    "Antigüedad en años":                       "Antiguedad en anos",
    "Naturaleza":                               "Naturaleza",
    "Parte del cuerpo afectada":                "Parte del cuerpo afectada",
    "Agente de la lesión":                      "Agente de la lesion",
    "Mecanismo de accidente/ Tipo de contacto": "Mecanismo de accidente/ Tipo de contacto",
    "Total días de incapacidad":                "Total dias de incapacidad",
}

COL_MODEL_RENAME = {
    "Clasificacion del accidente":              "clasificacion",
    "Tipo de accidente":                        "tipo",
    "Estacion donde ocurrio":                   "estacion",
    "Genero":                                   "genero",
    "Puesto/Cargo":                             "cargo",
    "Antiguedad en anos":                       "antiguedad",
    "Naturaleza":                               "naturaleza",
    "Parte del cuerpo afectada":                "parte_cuerpo",
    "Agente de la lesion":                      "agente",
    "Mecanismo de accidente/ Tipo de contacto": "mecanismo",
    "Total dias de incapacidad":                "dias_incapacidad",
}

FEAT_LABEL = {
    "cargo":         "Cargo",
    "mecanismo":     "Mecanismo del accidente",
    "parte_cuerpo":  "Parte del cuerpo",
    "naturaleza":    "Naturaleza de lesion",
    "agente":        "Agente de lesion",
    "antiguedad":    "Antiguedad (anos)",
    "estacion":      "Estacion",
    "genero":        "Genero",
    "clasificacion": "Clasificacion",
    "tipo":          "Tipo de accidente",
}

# =============================================================================
# FUNCIONES DE PROCESAMIENTO
# =============================================================================

@st.cache_data(show_spinner="Cargando datos...")
def load_data(file_bytes: bytes) -> pd.DataFrame:
    raw = pd.read_excel(BytesIO(file_bytes), sheet_name="PREVSIS", skiprows=1)
    raw.columns = raw.iloc[0]
    raw = raw.drop(index=0).reset_index(drop=True)
    raw = raw.dropna(axis=1, how="all")
    raw = raw.rename(columns=COL_RENAME_ORIG)
    return raw


@st.cache_data(show_spinner="Preparando features...")
def prepare_features(file_bytes: bytes):
    raw = load_data(file_bytes)

    missing = [c for c in KEY_COLS if c not in raw.columns]
    if missing:
        st.error(f"Columnas no encontradas en el archivo: {missing}")
        st.stop()

    mdf = raw[KEY_COLS].copy()
    mdf = mdf.rename(columns=COL_MODEL_RENAME)

    # Dias de incapacidad — numérico
    mdf["dias_incapacidad"] = pd.to_numeric(
        mdf["dias_incapacidad"], errors="coerce"
    ).fillna(0)

    # Antiguedad — CORRECCIÓN: replace con dict para reemplazar valores individuales
    antiguedad_str = (
        mdf["antiguedad"]
        .astype(str)
        .str.strip()
        .replace({"Sin informacion": np.nan, "nan": np.nan, "None": np.nan, "": np.nan})
    )
    mdf["antiguedad"] = pd.to_numeric(antiguedad_str, errors="coerce").fillna(5)

    # Variables target
    mdf["con_incapacidad"] = (mdf["dias_incapacidad"] > 0).astype(int)
    mdf["incapacidad_alta"] = (mdf["dias_incapacidad"] >= 5).astype(int)

    # Encoding categóricas
    encoders = {}
    cat_cols = [c for c in mdf.columns if mdf[c].dtype == object]
    for col in cat_cols:
        le = LabelEncoder()
        mdf[col] = mdf[col].fillna("Sin informacion").astype(str).str.strip()
        mdf[col] = le.fit_transform(mdf[col])
        encoders[col] = le

    # Garantizar que todo sea float (defensa extra contra dtype object residual)
    for col in mdf.columns:
        mdf[col] = pd.to_numeric(mdf[col], errors="coerce").fillna(0)

    return mdf, encoders


@st.cache_data(show_spinner="Entrenando modelo...")
def train_model(file_bytes: bytes, target: str, algo: str) -> dict:
    mdf, encoders = prepare_features(file_bytes)

    feat_cols = [
        c for c in mdf.columns
        if c not in ["con_incapacidad", "incapacidad_alta", "dias_incapacidad"]
    ]
    X = mdf[feat_cols].astype(float)
    y = mdf[target].astype(int)

    if y.nunique() < 2:
        st.error(
            "La variable objetivo solo tiene una clase. "
            "Revisa que el archivo tenga registros con y sin incapacidad."
        )
        st.stop()

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    if algo == "Random Forest":
        model = RandomForestClassifier(
            n_estimators=200, max_depth=8,
            class_weight="balanced", random_state=42, n_jobs=-1,
        )
    elif algo == "Gradient Boosting":
        model = GradientBoostingClassifier(
            n_estimators=150, max_depth=5,
            learning_rate=0.05, random_state=42,
        )
    else:
        model = LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=42
        )

    model.fit(X_tr, y_tr)
    y_pred    = model.predict(X_te)
    y_prob    = model.predict_proba(X_te)[:, 1]
    cv_scores = cross_val_score(model, X, y, cv=5, scoring="roc_auc")
    auc       = roc_auc_score(y_te, y_prob)
    rep       = classification_report(y_te, y_pred, output_dict=True)
    cm        = confusion_matrix(y_te, y_pred)
    fpr, tpr, _ = roc_curve(y_te, y_prob)

    return {
        "model":     model,
        "encoders":  encoders,
        "feat_cols": feat_cols,
        "X_tr":      X_tr,
        "X_te":      X_te,
        "y_tr":      y_tr,
        "y_te":      y_te,
        "y_pred":    y_pred,
        "y_prob":    y_prob,
        "cv_scores": cv_scores,
        "auc":       auc,
        "rep":       rep,
        "cm":        cm,
        "fpr":       fpr,
        "tpr":       tpr,
    }


def encode_input(value: str, encoder: LabelEncoder) -> int:
    value = str(value).strip()
    if value in encoder.classes_:
        return int(encoder.transform([value])[0])
    return 0


# =============================================================================
# SIDEBAR
# =============================================================================

with st.sidebar:
    st.markdown("## 🦺 SST Predictivo")
    st.markdown("---")

    uploaded = st.file_uploader(
        "Subir DATA_2025.xlsx",
        type=["xlsx"],
        help="Archivo exportado del sistema PREVSIS",
    )

    st.markdown("### Configuracion del modelo")
    algo = st.selectbox(
        "Algoritmo",
        ["Random Forest", "Gradient Boosting", "Regresion Logistica"],
    )
    target = st.selectbox(
        "Variable objetivo",
        ["con_incapacidad", "incapacidad_alta"],
        format_func=lambda x: (
            "Tendra alguna incapacidad?" if x == "con_incapacidad"
            else "Incapacidad >= 5 dias?"
        ),
    )

    st.markdown("---")
    st.markdown("### Prediccion individual")
    st.caption("Configura un caso nuevo:")

    p_cargo      = st.selectbox("Cargo", CARGOS)
    p_genero     = st.selectbox("Genero", GENEROS)
    p_antiguedad = st.slider("Antiguedad (anos)", 0, 30, 3)
    p_naturaleza = st.selectbox("Naturaleza de lesion", NATURALEZAS)
    p_mecanismo  = st.selectbox("Mecanismo", MECANISMOS)
    p_agente     = st.selectbox("Agente de lesion", AGENTES)
    p_parte      = st.selectbox("Parte del cuerpo", PARTES)
    p_estacion   = st.selectbox("Estacion", ESTACIONES)
    p_tipo       = st.selectbox("Tipo de accidente", TIPOS)
    p_clasif     = st.selectbox("Clasificacion", CLASIFICACIONES)

    predecir_btn = st.button(
        "Calcular riesgo", type="primary", use_container_width=True
    )

# =============================================================================
# VALIDACION Y CARGA
# =============================================================================

if uploaded is None:
    st.warning("Carga el archivo DATA_2025.xlsx en el panel izquierdo para comenzar.")
    st.info(
        "La app espera la hoja PREVSIS con las columnas estandar del sistema "
        "de accidentalidad. Una vez cargado, el modelo se entrena automaticamente."
    )
    st.stop()

raw_bytes = uploaded.read()

try:
    raw_df = load_data(raw_bytes)
except Exception as e:
    st.error(f"Error al leer el archivo: {e}")
    st.stop()

try:
    art = train_model(raw_bytes, target, algo)
except Exception as e:
    st.error(f"Error al entrenar el modelo: {e}")
    st.stop()

model      = art["model"]
encoders   = art["encoders"]
feat_cols  = art["feat_cols"]
auc        = art["auc"]
cv_scores  = art["cv_scores"]
rep        = art["rep"]
cm         = art["cm"]
fpr        = art["fpr"]
tpr        = art["tpr"]
y_te       = art["y_te"]
y_pred_arr = art["y_pred"]
y_prob_arr = art["y_prob"]
X_tr       = art["X_tr"]
X_te       = art["X_te"]

# =============================================================================
# HEADER
# =============================================================================

st.markdown(
    """
    <div style="background:linear-gradient(135deg,#1E3A5F,#2E86AB);
                color:white;padding:1.4rem 1.8rem;border-radius:12px;margin-bottom:1.2rem;">
        <h1 style="margin:0;font-size:1.7rem;">
            🦺 SST Predictivo — Riesgo de Incapacidad
        </h1>
        <p style="margin:0.3rem 0 0;opacity:.85;font-size:.9rem;">
            Machine Learning sobre datos reales PREVSIS · Clasificacion binaria
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

tab1, tab2, tab3, tab4 = st.tabs(
    ["📊 Panorama", "🤖 Modelo ML", "🔮 Prediccion", "📋 Datos"]
)

# =============================================================================
# TAB 1 — PANORAMA
# =============================================================================
with tab1:

    total    = len(raw_df)
    dias_col = "Total dias de incapacidad"
    dias_num = pd.to_numeric(
        raw_df.get(dias_col, pd.Series([0] * total)), errors="coerce"
    ).fillna(0)

    pct_incap = float((dias_num > 0).mean() * 100)
    avg_dias  = float(dias_num[dias_num > 0].mean()) if (dias_num > 0).any() else 0.0
    clas_col  = "Clasificacion del accidente"
    n_graves  = int(
        (raw_df.get(clas_col, pd.Series()).astype(str).str.strip() == "Graves").sum()
    )

    c1, c2, c3, c4 = st.columns(4)
    for col_w, val, label, color in [
        (c1, f"{total:,}",           "Total incidentes",        "#1E3A5F"),
        (c2, f"{pct_incap:.1f}%",    "Con incapacidad",         "#F4A261"),
        (c3, f"{avg_dias:.1f} dias", "Promedio de incapacidad", "#2E86AB"),
        (c4, f"{n_graves}",          "Accidentes graves",       "#E63946"),
    ]:
        col_w.markdown(
            f'<div class="kpi-card">'
            f'<p class="kpi-value" style="color:{color};">{val}</p>'
            f'<p class="kpi-label">{label}</p></div>',
            unsafe_allow_html=True,
        )

    st.markdown("")
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown('<p class="section-title">Clasificacion del accidente</p>',
                    unsafe_allow_html=True)
        if clas_col in raw_df.columns:
            vc = (
                raw_df[clas_col].astype(str).str.strip()
                .value_counts().reset_index()
            )
            vc.columns = ["Clasificacion", "Cantidad"]
            fig = px.pie(
                vc, values="Cantidad", names="Clasificacion",
                color_discrete_sequence=["#2E86AB", "#F4A261", "#E63946", "#aaa"],
            )
            fig.update_layout(margin=dict(t=10, b=10), height=280)
            st.plotly_chart(fig, use_container_width=True)

    with col_b:
        st.markdown('<p class="section-title">Tipo de accidente</p>',
                    unsafe_allow_html=True)
        if "Tipo de accidente" in raw_df.columns:
            vc2 = (
                raw_df["Tipo de accidente"].astype(str).str.strip()
                .value_counts().head(7).reset_index()
            )
            vc2.columns = ["Tipo", "Cantidad"]
            fig2 = px.bar(
                vc2, x="Cantidad", y="Tipo", orientation="h",
                color="Cantidad", color_continuous_scale="Blues",
            )
            fig2.update_layout(
                margin=dict(t=10, b=10), height=280,
                coloraxis_showscale=False,
            )
            st.plotly_chart(fig2, use_container_width=True)

    col_c, col_d = st.columns(2)

    with col_c:
        st.markdown('<p class="section-title">Naturaleza de la lesion</p>',
                    unsafe_allow_html=True)
        if "Naturaleza" in raw_df.columns:
            vc3 = (
                raw_df["Naturaleza"].astype(str).str.strip()
                .str.replace(r"^\d+\.", "", regex=True).str.strip()
                .value_counts().head(8).reset_index()
            )
            vc3.columns = ["Naturaleza", "Cantidad"]
            fig3 = px.bar(
                vc3, x="Naturaleza", y="Cantidad",
                color="Cantidad", color_continuous_scale="Teal",
            )
            fig3.update_layout(
                margin=dict(t=10, b=90), height=310,
                xaxis_tickangle=-35, coloraxis_showscale=False,
            )
            st.plotly_chart(fig3, use_container_width=True)

    with col_d:
        st.markdown('<p class="section-title">Parte del cuerpo afectada</p>',
                    unsafe_allow_html=True)
        if "Parte del cuerpo afectada" in raw_df.columns:
            vc4 = (
                raw_df["Parte del cuerpo afectada"].astype(str).str.strip()
                .str.replace(r"^\d+\.", "", regex=True).str.strip()
                .value_counts().head(8).reset_index()
            )
            vc4.columns = ["Parte", "Cantidad"]
            fig4 = px.bar(
                vc4, x="Parte", y="Cantidad",
                color="Cantidad", color_continuous_scale="Oranges",
            )
            fig4.update_layout(
                margin=dict(t=10, b=90), height=310,
                xaxis_tickangle=-35, coloraxis_showscale=False,
            )
            st.plotly_chart(fig4, use_container_width=True)

    st.markdown(
        '<p class="section-title">Dias de incapacidad promedio por cargo (Top 10)</p>',
        unsafe_allow_html=True,
    )
    if "Puesto/Cargo" in raw_df.columns and dias_col in raw_df.columns:
        tmp = raw_df[["Puesto/Cargo", dias_col]].copy()
        tmp[dias_col] = pd.to_numeric(tmp[dias_col], errors="coerce")
        avg_cargo = (
            tmp.groupby("Puesto/Cargo")[dias_col]
            .mean()
            .sort_values(ascending=False)
            .head(10)
            .reset_index()
        )
        avg_cargo.columns = ["Cargo", "Dias promedio"]
        avg_cargo["Dias promedio"] = avg_cargo["Dias promedio"].round(2)
        fig5 = px.bar(
            avg_cargo, x="Dias promedio", y="Cargo", orientation="h",
            color="Dias promedio", color_continuous_scale="RdYlGn_r",
            text="Dias promedio",
        )
        fig5.update_traces(texttemplate="%{text:.1f}", textposition="outside")
        fig5.update_layout(
            margin=dict(t=10, b=10), height=380, coloraxis_showscale=False,
        )
        st.plotly_chart(fig5, use_container_width=True)


# =============================================================================
# TAB 2 — MODELO ML
# =============================================================================
with tab2:

    st.markdown('<p class="section-title">Metricas de evaluacion</p>',
                unsafe_allow_html=True)

    m1, m2, m3, m4, m5 = st.columns(5)
    cv_mean = float(cv_scores.mean())
    cv_std  = float(cv_scores.std())
    for col_w, label, val, color in [
        (m1, "AUC-ROC",   f"{auc:.3f}",                    "#1E3A5F"),
        (m2, "CV AUC",    f"{cv_mean:.3f}+-{cv_std:.3f}",  "#2E86AB"),
        (m3, "Accuracy",  f"{rep['accuracy']:.1%}",         "#2A9D8F"),
        (m4, "Precision", f"{rep['1']['precision']:.1%}",   "#F4A261"),
        (m5, "Recall",    f"{rep['1']['recall']:.1%}",      "#E63946"),
    ]:
        col_w.markdown(
            f'<div class="kpi-card">'
            f'<p class="kpi-value" style="color:{color};">{val}</p>'
            f'<p class="kpi-label">{label}</p></div>',
            unsafe_allow_html=True,
        )

    st.caption(
        f"Modelo: {algo} | Train: {len(X_tr):,} registros | "
        f"Test: {len(X_te):,} registros | Target: {target}"
    )

    col_roc, col_cm = st.columns(2)

    with col_roc:
        st.markdown('<p class="section-title">Curva ROC</p>',
                    unsafe_allow_html=True)
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(
            x=list(fpr), y=list(tpr), mode="lines",
            line=dict(color="#2E86AB", width=2.5),
            name=f"Modelo (AUC={auc:.3f})",
        ))
        fig_roc.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1], mode="lines",
            line=dict(color="#aaa", dash="dash"),
            name="Clasificador aleatorio",
        ))
        fig_roc.update_layout(
            xaxis_title="Tasa de falsos positivos",
            yaxis_title="Tasa de verdaderos positivos",
            legend=dict(x=0.5, y=0.1),
            height=340, margin=dict(t=10, b=30),
        )
        st.plotly_chart(fig_roc, use_container_width=True)

    with col_cm:
        st.markdown('<p class="section-title">Matriz de confusion</p>',
                    unsafe_allow_html=True)
        labels = ["Sin incapacidad", "Con incapacidad"]
        fig_cm = px.imshow(
            cm, text_auto=True, color_continuous_scale="Blues",
            x=labels, y=labels,
            labels=dict(x="Predicho", y="Real", color="Casos"),
        )
        fig_cm.update_layout(height=340, margin=dict(t=10, b=30))
        st.plotly_chart(fig_cm, use_container_width=True)

    if hasattr(model, "feature_importances_"):
        st.markdown(
            '<p class="section-title">Importancia de variables (Top 10)</p>',
            unsafe_allow_html=True,
        )
        fi = pd.DataFrame({
            "Variable":    feat_cols,
            "Importancia": model.feature_importances_,
        })
        fi = fi.sort_values("Importancia", ascending=False).head(10).copy()
        fi["Variable"] = fi["Variable"].map(FEAT_LABEL).fillna(fi["Variable"])
        fi["Pct"] = (fi["Importancia"] * 100).round(1)

        fig_fi = px.bar(
            fi, x="Pct", y="Variable", orientation="h",
            color="Pct", color_continuous_scale="Blues",
            text="Pct",
        )
        fig_fi.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig_fi.update_layout(
            xaxis_title="Importancia (%)",
            height=380, margin=dict(t=10, b=10),
            coloraxis_showscale=False,
        )
        st.plotly_chart(fig_fi, use_container_width=True)

    with st.expander("Reporte completo de clasificacion"):
        rep_df = pd.DataFrame(rep).T.round(3)
        st.dataframe(rep_df, use_container_width=True)

    with st.expander("Interpretacion del modelo"):
        st.markdown(
            f"**AUC-ROC: {auc:.3f}**\n\n"
            "- Mayor a 0.70 = modelo util para decisiones preventivas\n"
            f"- El modelo discrimina correctamente en {auc*100:.1f}% de los casos.\n\n"
            f"**Variable objetivo:** {target}\n\n"
            "- con_incapacidad: predice si habra algun dia de incapacidad\n"
            "- incapacidad_alta: predice incapacidades prolongadas (mayor a 5 dias)\n\n"
            "**Uso recomendado:**\n"
            "1. Usar el score de probabilidad (no solo la clase predicha).\n"
            "2. Casos con probabilidad mayor al 60% -> intervencion inmediata.\n"
            "3. Actualizar el modelo cada 6 meses con datos nuevos del PREVSIS."
        )


# =============================================================================
# TAB 3 — PREDICCION INDIVIDUAL
# =============================================================================
with tab3:

    st.markdown('<p class="section-title">Prediccion de riesgo para un caso nuevo</p>',
                unsafe_allow_html=True)
    st.info("Ajusta los parametros en el panel izquierdo y pulsa Calcular riesgo.")

    if predecir_btn:
        input_map = {
            "clasificacion": p_clasif,
            "tipo":          p_tipo,
            "estacion":      p_estacion,
            "genero":        p_genero,
            "cargo":         p_cargo,
            "antiguedad":    float(p_antiguedad),
            "naturaleza":    p_naturaleza,
            "parte_cuerpo":  p_parte,
            "agente":        p_agente,
            "mecanismo":     p_mecanismo,
        }

        input_row = {}
        for col in feat_cols:
            if col == "antiguedad":
                input_row[col] = float(p_antiguedad)
            elif col in input_map and col in encoders:
                input_row[col] = encode_input(input_map[col], encoders[col])
            else:
                input_row[col] = 0

        X_new = pd.DataFrame([input_row]).astype(float)
        prob  = float(model.predict_proba(X_new)[0][1])

        nivel  = "Alto" if prob >= 0.60 else ("Medio" if prob >= 0.35 else "Bajo")
        bg_col = {"Alto": "#FFE5E5", "Medio": "#FFF3E0", "Bajo": "#E5F7F4"}[nivel]
        tx_col = {"Alto": "#E63946", "Medio": "#F4A261", "Bajo": "#2A9D8F"}[nivel]
        icono  = {"Alto": "ALTO", "Medio": "MEDIO", "Bajo": "BAJO"}[nivel]

        col_gauge, col_recs = st.columns([1, 1])

        with col_gauge:
            st.markdown(
                f'<div class="pred-box" style="background:{bg_col};border:2px solid {tx_col};">'
                f'<h2 style="color:{tx_col};font-size:1.6rem;">Riesgo {icono}</h2>'
                f'<p style="color:{tx_col};font-size:1.3rem;font-weight:700;">'
                f'{prob:.1%} de probabilidad</p>'
                f'<p style="color:#555;font-size:.85rem;">Variable: {target}</p>'
                f'</div>',
                unsafe_allow_html=True,
            )

            gauge_fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=round(prob * 100, 1),
                number={"suffix": "%", "font": {"size": 32, "color": tx_col}},
                gauge={
                    "axis": {"range": [0, 100], "tickwidth": 1},
                    "bar": {"color": tx_col, "thickness": 0.25},
                    "steps": [
                        {"range": [0, 35],   "color": "#E5F7F4"},
                        {"range": [35, 60],  "color": "#FFF3E0"},
                        {"range": [60, 100], "color": "#FFE5E5"},
                    ],
                    "threshold": {
                        "line": {"color": tx_col, "width": 4},
                        "thickness": 0.8,
                        "value": round(prob * 100, 1),
                    },
                },
                title={"text": "Indice de riesgo", "font": {"size": 14}},
            ))
            gauge_fig.update_layout(height=260, margin=dict(t=30, b=10, l=20, r=20))
            st.plotly_chart(gauge_fig, use_container_width=True)

        with col_recs:
            st.markdown("#### Recomendaciones automaticas")
            recs = []

            if "Lumbar" in p_naturaleza or "esfuerzo" in p_mecanismo:
                recs.append(
                    "**Riesgo ergonomico**: verificar tecnica de levantamiento, "
                    "entregar faja lumbar y revisar puesto de trabajo."
                )
            if any(x in p_parte for x in ["Espalda", "Miembros"]):
                recs.append(
                    "**Parte critica afectada**: priorizar evaluacion medica "
                    "ocupacional y seguimiento con medicina del trabajo."
                )
            if p_antiguedad <= 2:
                recs.append(
                    "**Trabajador nuevo** (<=2 anos): reforzar induccion SST "
                    "y asignar tutor durante primeros 6 meses."
                )
            if any(x in p_agente for x in ["Aeronave", "presion"]):
                recs.append(
                    "**Riesgo aeronautico**: verificar procedimientos de "
                    "aseguramiento en maniobras y uso correcto de equipos."
                )
            if "Transito" in p_tipo:
                recs.append(
                    "**Accidente de transito**: validar politica de conduccion "
                    "segura y verificar condiciones del vehiculo."
                )
            if "Violencia" in p_tipo:
                recs.append(
                    "**Evento de violencia**: activar protocolo de atencion "
                    "psicosocial y reportar a seguridad corporativa."
                )
            if nivel == "Alto":
                recs.append(
                    "**NIVEL ALTO**: notificar ARL en las proximas 24 h y "
                    "activar protocolo de investigacion de accidente."
                )
            if not recs:
                recs.append(
                    "Perfil de riesgo moderado. Mantener controles SST "
                    "estandar y seguimiento periodico."
                )

            for r in recs:
                st.markdown(f"- {r}")

            st.markdown("---")
            st.markdown("**Resumen del caso evaluado:**")
            resumen_items = [
                ("Cargo",          p_cargo),
                ("Genero",         p_genero),
                ("Antiguedad",     f"{p_antiguedad} anos"),
                ("Naturaleza",     p_naturaleza),
                ("Mecanismo",      p_mecanismo),
                ("Parte afectada", p_parte),
                ("Tipo accidente", p_tipo),
                ("Clasificacion",  p_clasif),
            ]
            for k, v in resumen_items:
                st.markdown(f"- **{k}:** {v}")

    else:
        st.markdown(
            '<div style="text-align:center;padding:3rem;color:#888;">'
            "<p>Configura el caso en el panel izquierdo</p>"
            "<p>y pulsa <strong>Calcular riesgo</strong></p>"
            "</div>",
            unsafe_allow_html=True,
        )


# =============================================================================
# TAB 4 — DATOS
# =============================================================================
with tab4:

    show_cols = [
        c for c in [
            "ID Reporte", "Fecha de ocurrencia", "Mes",
            "Clasificacion del accidente",
            "Tipo de accidente",
            "Genero", "Puesto/Cargo", "Antiguedad en anos",
            "Naturaleza", "Parte del cuerpo afectada",
            "Agente de la lesion",
            "Mecanismo de accidente/ Tipo de contacto",
            "Total dias de incapacidad",
        ]
        if c in raw_df.columns
    ]

    st.markdown(
        f'<p class="section-title">Vista de datos ({len(raw_df):,} registros)</p>',
        unsafe_allow_html=True,
    )

    col_fil1, col_fil2 = st.columns(2)
    clas_opts = raw_df["Clasificacion del accidente"].dropna().unique().tolist() \
        if "Clasificacion del accidente" in raw_df.columns else []
    tipo_opts = raw_df["Tipo de accidente"].dropna().unique().tolist() \
        if "Tipo de accidente" in raw_df.columns else []

    with col_fil1:
        fil_clasif = st.multiselect("Filtrar por clasificacion", options=clas_opts)
    with col_fil2:
        fil_tipo = st.multiselect("Filtrar por tipo", options=tipo_opts)

    df_view = raw_df[show_cols].copy() if show_cols else raw_df.copy()
    if fil_clasif and "Clasificacion del accidente" in df_view.columns:
        df_view = df_view[df_view["Clasificacion del accidente"].isin(fil_clasif)]
    if fil_tipo and "Tipo de accidente" in df_view.columns:
        df_view = df_view[df_view["Tipo de accidente"].isin(fil_tipo)]

    st.dataframe(df_view, use_container_width=True, height=420)
    st.caption(f"Mostrando {len(df_view):,} registros")

    buf = BytesIO()
    df_view.to_excel(buf, index=False)
    st.download_button(
        label="Descargar seleccion (.xlsx)",
        data=buf.getvalue(),
        file_name="sst_datos_filtrados.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

    st.markdown("---")
    st.markdown(
        f"**Stack:** Python · Streamlit · Scikit-learn ({algo}) · Plotly · Pandas\n\n"
        f"**Features del modelo:** {', '.join(feat_cols)}\n\n"
        f"**Variable objetivo:** {target}"
    )
