# =============================================================================
# SST PREDICTIVO — Dashboard Integrado de Accidentalidad
# Soporta: DATA_2025.xlsx (PREVSIS) + Tablero_Accidentalidad_2026_Con_HC.xlsx
#
# INSTALACIÓN:
#   pip install streamlit scikit-learn pandas plotly openpyxl numpy
#
# EJECUTAR:
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
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve,
)

# =============================================================================
# PÁGINA
# =============================================================================
st.set_page_config(
    page_title="SST Predictivo | Dashboard Accidentalidad",
    page_icon="🦺",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.block-container{padding-top:1.2rem}
.kpi-card{background:white;border-radius:10px;padding:1rem;border:1px solid #e0e0e0;text-align:center}
.kpi-value{font-size:1.9rem;font-weight:700;margin:0}
.kpi-label{color:#666;font-size:.8rem;margin:.15rem 0 0}
.section-title{font-size:1rem;font-weight:600;color:#1E3A5F;
  border-left:4px solid #2E86AB;padding-left:.7rem;margin:1.3rem 0 .7rem}
.pred-box{padding:1.3rem;border-radius:12px;text-align:center;margin:.7rem 0}
.badge-green{background:#E5F7F4;color:#2A9D8F;padding:2px 8px;border-radius:4px;font-size:.78rem;font-weight:600}
.badge-yellow{background:#FFF3E0;color:#F4A261;padding:2px 8px;border-radius:4px;font-size:.78rem;font-weight:600}
.badge-red{background:#FFE5E5;color:#E63946;padding:2px 8px;border-radius:4px;font-size:.78rem;font-weight:600}
.meta-line{border-top:2px dashed #E63946;opacity:.7}
</style>
""", unsafe_allow_html=True)

# =============================================================================
# HELPERS DE CÁLCULO NORMATIVO (GTC 3701 / Resolución 2400)
# =============================================================================
META_IF   = 2.54   # Meta IF corporativa (dato extraído del archivo)
META_IS   = 3.00   # Referencia IS sectorial

def calc_if(n_at: float, hht: float) -> float:
    """IF = (N° AT * 240000) / HHT"""
    if hht and hht > 0:
        return round((n_at * 240_000) / hht, 4)
    return 0.0

def calc_is(n_dias: float, hht: float) -> float:
    """IS = (N° Días AT * 240000) / HHT"""
    if hht and hht > 0:
        return round((n_dias * 240_000) / hht, 4)
    return 0.0

def calc_severidad(n_at: float, n_dias: float) -> float:
    """Severidad Media = N° Días / N° AT"""
    if n_at and n_at > 0:
        return round(n_dias / n_at, 2)
    return 0.0

def nivel_if(valor: float) -> str:
    if valor <= META_IF:
        return "BAJO"
    elif valor <= META_IF * 1.5:
        return "MEDIO"
    return "ALTO"

def badge_nivel(nivel: str) -> str:
    clases = {"BAJO": "badge-green", "MEDIO": "badge-yellow", "ALTO": "badge-red"}
    return f'<span class="{clases.get(nivel, "badge-yellow")}">{nivel}</span>'

# =============================================================================
# LOADERS PARA CADA TIPO DE ARCHIVO
# =============================================================================

@st.cache_data(show_spinner="Cargando Tablero...")
def load_tablero(file_bytes: bytes):
    """Parsea Tablero_Accidentalidad_2026_Con_HC.xlsx → Data sheet."""
    xl = pd.read_excel(BytesIO(file_bytes), sheet_name="Data", skiprows=1)
    xl.columns = xl.iloc[0]
    xl = xl.drop(index=0).reset_index(drop=True)

    # Mitad GENERAL (columnas 1-13)
    gen = xl.iloc[:, 1:14].copy()
    gen.columns = ["Mes","Meses","Unidad_Negocio","Estacion","Poblacion",
                   "N_AT","N_AT_IND","N_Dias_Total","N_Dias_Ind","HHT","IF","IF_IND","IS"]

    # Mitad AGIL (columnas 16-27)
    agil = xl.iloc[:, 16:28].copy()
    agil.columns = ["Mes","Meses","Unidad_Negocio","Estacion","Poblacion",
                    "N_AT","N_AT_IND","N_Dias_Total","N_Dias_Ind","HHT","IF","IF_IND","IS"]

    results = {}
    for nombre, df in [("GENERAL", gen), ("AGIL", agil)]:
        df = df[df["Mes"].notna() & (df["Mes"].astype(str) != "Mes")].copy()
        for col in ["N_AT","N_AT_IND","N_Dias_Total","N_Dias_Ind","HHT","IF","IF_IND","IS"]:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
        df = df[df["N_AT"] > 0].copy()
        df["fuente"] = nombre
        results[nombre] = df

    return results


@st.cache_data(show_spinner="Cargando Indicadores...")
def load_indicadores(file_bytes: bytes):
    """Parsea hoja Data Indicadores → series IF/IS por estación."""
    xl = pd.read_excel(BytesIO(file_bytes), sheet_name="Data Indicadores", skiprows=0)
    xl.columns = xl.iloc[0]
    xl = xl.drop(index=0).reset_index(drop=True)

    # IF Total por estación (filas 0-11, columnas 1-14)
    if_df = xl.iloc[0:12, 1:15].copy()
    if_df.columns = ["Estacion","JAN","FEB","MAR","APR","MAY",
                     "JUN","JUL","AUG","SEP","OCT","NOV","DEC","ACUM"]
    if_df = if_df[if_df["Estacion"].notna() & (if_df["Estacion"].astype(str).str.len() > 1)]
    meses = ["JAN","FEB","MAR","APR","MAY","JUN","JUL","AUG","SEP","OCT","NOV","DEC","ACUM"]
    for c in meses:
        if_df[c] = pd.to_numeric(if_df[c], errors="coerce").fillna(0)

    # IF Indicador (columnas 16-28)
    if_ind = xl.iloc[0:12, 16:29].copy()
    if_ind.columns = ["Estacion","JAN","FEB","MAR","APR","MAY",
                      "JUN","JUL","AUG","SEP","OCT","NOV","DEC"]
    if_ind = if_ind[if_ind["Estacion"].notna() & (if_ind["Estacion"].astype(str).str.len() > 1)]
    for c in ["JAN","FEB","MAR","APR","MAY","JUN","JUL","AUG","SEP","OCT","NOV","DEC"]:
        if_ind[c] = pd.to_numeric(if_ind[c], errors="coerce").fillna(0)

    return if_df.reset_index(drop=True), if_ind.reset_index(drop=True)


@st.cache_data(show_spinner="Cargando PREVSIS...")
def load_prevsis(file_bytes: bytes):
    """Parsea DATA_2025.xlsx (PREVSIS) hoja PREVSIS."""
    COL_ORIG = {
        "Clasificación del accidente":              "Clasificacion",
        "Tipo de accidente":                        "Tipo",
        "Estación donde ocurrió":                   "Estacion",
        "Género":                                   "Genero",
        "Puesto/Cargo":                             "Cargo",
        "Antigüedad en años":                       "Antiguedad",
        "Naturaleza":                               "Naturaleza",
        "Parte del cuerpo afectada":                "Parte_cuerpo",
        "Agente de la lesión":                      "Agente",
        "Mecanismo de accidente/ Tipo de contacto": "Mecanismo",
        "Total días de incapacidad":                "Dias_incapacidad",
    }
    raw = pd.read_excel(BytesIO(file_bytes), sheet_name="PREVSIS", skiprows=1)
    raw.columns = raw.iloc[0]
    raw = raw.drop(index=0).reset_index(drop=True)
    raw = raw.rename(columns=COL_ORIG)
    cols = [v for v in COL_ORIG.values() if v in raw.columns]
    df = raw[cols].copy()
    df["Dias_incapacidad"] = pd.to_numeric(df["Dias_incapacidad"], errors="coerce").fillna(0)
    df["Antiguedad"] = pd.to_numeric(
        df.get("Antiguedad", pd.Series(dtype=str)).astype(str).replace("Sin información", np.nan),
        errors="coerce"
    ).fillna(5)
    return df


@st.cache_resource(show_spinner="Entrenando modelo ML...")
def train_ml(file_bytes: bytes, target: str, algo: str):
    df = load_prevsis(file_bytes)
    df["con_incapacidad"] = (df["Dias_incapacidad"] > 0).astype(int)
    df["incapacidad_alta"] = (df["Dias_incapacidad"] >= 5).astype(int)

    enc = {}
    for col in [c for c in df.columns if df[c].dtype == object]:
        le = LabelEncoder()
        df[col] = df[col].fillna("Sin informacion").astype(str)
        df[col] = le.fit_transform(df[col])
        enc[col] = le

    feats = [c for c in df.columns if c not in
             ["con_incapacidad", "incapacidad_alta", "Dias_incapacidad"]]
    X, y = df[feats], df[target]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=.25, random_state=42, stratify=y)

    if algo == "Random Forest":
        m = RandomForestClassifier(n_estimators=200, max_depth=8,
                                   class_weight="balanced", random_state=42, n_jobs=-1)
    elif algo == "Gradient Boosting":
        m = GradientBoostingClassifier(n_estimators=150, max_depth=5,
                                       learning_rate=.05, random_state=42)
    else:
        m = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)

    m.fit(X_tr, y_tr)
    yp = m.predict(X_te)
    yprob = m.predict_proba(X_te)[:, 1]
    cv = cross_val_score(m, X, y, cv=5, scoring="roc_auc")
    fpr, tpr, _ = roc_curve(y_te, yprob)

    return dict(model=m, enc=enc, feats=feats,
                auc=roc_auc_score(y_te, yprob), cv=cv,
                rep=classification_report(y_te, yp, output_dict=True),
                cm=confusion_matrix(y_te, yp),
                fpr=fpr, tpr=tpr,
                X_tr=X_tr, X_te=X_te, y_te=y_te)

# =============================================================================
# SIDEBAR
# =============================================================================
with st.sidebar:
    st.markdown("## 🦺 SST Predictivo")
    st.markdown("---")

    st.markdown("### 📂 Archivos de datos")
    f_tablero = st.file_uploader("Tablero Accidentalidad (.xlsx)",
                                  type=["xlsx"], key="tablero",
                                  help="Tablero_Accidentalidad_2026_Con_HC.xlsx")
    f_prevsis = st.file_uploader("Base PREVSIS (.xlsx)",
                                  type=["xlsx"], key="prevsis",
                                  help="DATA_2025.xlsx — hoja PREVSIS")

    st.markdown("---")
    st.markdown("### ⚙️ Modelo ML")
    algo = st.selectbox("Algoritmo", ["Random Forest","Gradient Boosting","Regresion Logistica"])
    target = st.selectbox("Variable objetivo",
        ["con_incapacidad","incapacidad_alta"],
        format_func=lambda x: "Tendra incapacidad?" if x=="con_incapacidad"
                              else "Incapacidad >= 5 dias?")

    st.markdown("---")
    st.markdown("### 🔮 Prediccion individual")
    p_cargo = st.selectbox("Cargo", [
        "Tripulante de Cabina Nacional","Agente de Servicio al Cliente",
        "Tripulante de Cabina Internacional/Nacional","Jefe de Cabina Nacional",
        "Aprendiz","Tecnico IV de Mantenimiento Linea","Agente de Rampa",
        "Auxiliar de carga","Piloto","Otro"])
    p_genero     = st.selectbox("Genero", ["Femenino","Masculino","Sin informacion"])
    p_antiguedad = st.slider("Antiguedad (anos)", 0, 30, 3)
    p_naturaleza = st.selectbox("Naturaleza lesion", [
        "1.Golpe, Contusion o Aplastamiento","1.Lesion Lumbar","1.Dolor Otico",
        "1.Torcedura","1.Herida","1.Desgarro Muscular","1.Esguince","1.Otros"])
    p_mecanismo  = st.selectbox("Mecanismo", [
        "1.Caida de Personas a Nivel de Piso  (Resbalon o tropezon )",
        "1.Golpes contra objetos","1.Golpes por objetos",
        "1.Sobre-esfuerzo (levantar objetos, halar, empujar, manipular o lanzar objetos)",
        "1.Falso Movimiento","1.Atrapamiento","1.Otros"])
    p_agente     = st.selectbox("Agente lesion", [
        "1.Movimiento del cuerpo","1.Herramientas, Maquinarias, Implementos o Utensilios",
        "1.Aeronave y sus componentes","1.Equipaje","1.Ambiente de trabajo","1.Otros"])
    p_parte      = st.selectbox("Parte del cuerpo", [
        "1.Espalda","1.Oreja (oido)","1.Manos","1.Dedos de mano","1.Pie",
        "1.Multiples partes","1.Cabeza","1.Lesiones generales u otros"])
    p_estacion   = st.selectbox("Estacion", [
        "Bogota","Rionegro","Cali","Medellin","El Salvador",
        "San Jose de Costa Rica","Madrid","Barranquilla","Quito"])
    p_tipo       = st.selectbox("Tipo accidente", [
        "Propio del Trabajo","Transito","Itinere/Fuera de Jornada Laboral",
        "Violencia","Otro"])
    p_clasif     = st.selectbox("Clasificacion", ["Leves","Severos","Graves"])
    predecir_btn = st.button("Calcular riesgo", type="primary", use_container_width=True)

# =============================================================================
# VALIDACIÓN Y CARGA
# =============================================================================
if f_tablero is None and f_prevsis is None:
    st.markdown("""
    <div style="background:linear-gradient(135deg,#1E3A5F,#2E86AB);color:white;
                padding:1.6rem 2rem;border-radius:12px;margin-bottom:1rem;">
        <h1 style="margin:0;font-size:1.8rem;">🦺 SST Predictivo — Dashboard Integrado</h1>
        <p style="margin:.4rem 0 0;opacity:.85;">IF · IS · Severidad normativa · Prediccion ML · Indicadores por estacion</p>
    </div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.info("**📊 Tablero Accidentalidad**\n\nCarga `Tablero_Accidentalidad_2026_Con_HC.xlsx` "
                "para ver IF, IS, severidad por estacion y mes.")
    with c2:
        st.info("**🧠 Base PREVSIS**\n\nCarga `DATA_2025.xlsx` "
                "para activar el modelo predictivo de incapacidad por caso.")

    st.markdown("""
    ---
    ### Formulas normativas SST (GTC 3701 / Resolución 2400)

    | Indicador | Formula | Meta referencia |
    |---|---|---|
    | **IF** – Indice de Frecuencia | (N° AT × 240.000) / HHT | ≤ 2.54 |
    | **IS** – Indice de Severidad | (N° Dias AT × 240.000) / HHT | ≤ 3.00 |
    | **Severidad Media** | N° Dias AT / N° AT | — |
    | **ILI** – Indice Lesiones Incapacitantes | (IF × IS) / 1.000 | — |
    """)
    st.stop()

tablero_data = load_tablero(f_tablero.read()) if f_tablero else None
prevsis_bytes = f_prevsis.read() if f_prevsis else None

# =============================================================================
# HEADER
# =============================================================================
st.markdown("""
<div style="background:linear-gradient(135deg,#1E3A5F,#2E86AB);color:white;
            padding:1.3rem 1.8rem;border-radius:12px;margin-bottom:1rem;">
    <h1 style="margin:0;font-size:1.65rem;">🦺 SST Predictivo — Dashboard Integrado</h1>
    <p style="margin:.3rem 0 0;opacity:.85;font-size:.88rem;">
        IF · IS · Severidad · Indicadores por estacion · Prediccion ML de incapacidad
    </p>
</div>
""", unsafe_allow_html=True)

# Tabs dinámicos según archivos cargados
tab_labels = ["📊 Indicadores IF/IS"]
if tablero_data:
    tab_labels += ["🏢 Por Estacion", "📈 Evolucion Mensual"]
if prevsis_bytes:
    tab_labels += ["🤖 Modelo ML", "🔮 Prediccion"]
if tablero_data or prevsis_bytes:
    tab_labels += ["📋 Datos"]

tabs = st.tabs(tab_labels)
tab_idx = {name: i for i, name in enumerate(tab_labels)}

# =============================================================================
# TAB: INDICADORES IF/IS
# =============================================================================
with tabs[tab_idx["📊 Indicadores IF/IS"]]:

    if tablero_data:
        df_gen = tablero_data["GENERAL"]
        df_agil = tablero_data["AGIL"]

        # KPIs globales
        n_at_total   = int(df_gen["N_AT"].sum())
        dias_total    = int(df_gen["N_Dias_Total"].sum())
        hht_total     = float(df_gen["HHT"].sum())
        if_global     = calc_if(n_at_total, hht_total)
        is_global     = calc_is(dias_total, hht_total)
        severidad_med = calc_severidad(n_at_total, dias_total)
        ili           = round((if_global * is_global) / 1000, 4)
        nivel         = nivel_if(if_global)

        st.markdown('<p class="section-title">KPIs Globales — Acumulado 2026</p>',
                    unsafe_allow_html=True)
        c1,c2,c3,c4,c5,c6 = st.columns(6)
        kpis = [
            (c1, f"{n_at_total}",        "N° AT Total",       "#1E3A5F"),
            (c2, f"{if_global:.3f}",     "IF Total",          "#E63946" if nivel=="ALTO" else "#F4A261" if nivel=="MEDIO" else "#2A9D8F"),
            (c3, f"{META_IF}",           "Meta IF",           "#888"),
            (c4, f"{is_global:.3f}",     "IS Total",          "#2E86AB"),
            (c5, f"{severidad_med:.1f}", "Severidad Media",   "#F4A261"),
            (c6, f"{ili:.4f}",           "ILI",               "#9C27B0"),
        ]
        for col_w, val, label, color in kpis:
            col_w.markdown(
                f'<div class="kpi-card"><p class="kpi-value" style="color:{color};">{val}</p>'
                f'<p class="kpi-label">{label}</p></div>', unsafe_allow_html=True)

        niv_badge = badge_nivel(nivel)
        st.markdown(
            f"Nivel de riesgo IF: {niv_badge} &nbsp;|&nbsp; "
            f"Meta IF corporativa: **{META_IF}** &nbsp;|&nbsp; "
            f"Formula: IF = (N°AT × 240.000) / HHT",
            unsafe_allow_html=True)

        st.markdown("")

        # IF/IS por estacion
        st.markdown('<p class="section-title">IF e IS por Estacion</p>', unsafe_allow_html=True)
        est_grp = df_gen.groupby("Estacion").agg(
            N_AT=("N_AT","sum"),
            N_Dias=("N_Dias_Total","sum"),
            HHT=("HHT","sum")
        ).reset_index()
        est_grp["IF_calc"]  = est_grp.apply(lambda r: calc_if(r["N_AT"], r["HHT"]), axis=1)
        est_grp["IS_calc"]  = est_grp.apply(lambda r: calc_is(r["N_Dias"], r["HHT"]), axis=1)
        est_grp["Severidad"]= est_grp.apply(lambda r: calc_severidad(r["N_AT"], r["N_Dias"]), axis=1)
        est_grp["Nivel_IF"] = est_grp["IF_calc"].apply(nivel_if)
        est_grp = est_grp[est_grp["N_AT"] > 0].sort_values("IF_calc", ascending=False)

        col_if, col_is = st.columns(2)
        with col_if:
            color_map = {"ALTO":"#E63946","MEDIO":"#F4A261","BAJO":"#2A9D8F"}
            colors = est_grp["Nivel_IF"].map(color_map).tolist()
            fig_if = go.Figure()
            fig_if.add_trace(go.Bar(
                x=est_grp["Estacion"], y=est_grp["IF_calc"],
                marker_color=colors,
                text=[f"{v:.2f}" for v in est_grp["IF_calc"]],
                textposition="outside", name="IF"))
            fig_if.add_hline(y=META_IF, line_dash="dash", line_color="#E63946",
                             annotation_text=f"Meta {META_IF}",
                             annotation_position="bottom right")
            fig_if.update_layout(title="Indice de Frecuencia (IF) por Estacion",
                                 yaxis_title="IF", height=320,
                                 margin=dict(t=35,b=20), showlegend=False)
            st.plotly_chart(fig_if, use_container_width=True)

        with col_is:
            fig_is = px.bar(est_grp, x="Estacion", y="IS_calc",
                            color="IS_calc", color_continuous_scale="Oranges",
                            text=[f"{v:.2f}" for v in est_grp["IS_calc"]],
                            title="Indice de Severidad (IS) por Estacion")
            fig_is.update_traces(textposition="outside")
            fig_is.add_hline(y=META_IS, line_dash="dash", line_color="#F4A261",
                             annotation_text=f"Ref IS {META_IS}",
                             annotation_position="bottom right")
            fig_is.update_layout(yaxis_title="IS", height=320,
                                 margin=dict(t=35,b=20), coloraxis_showscale=False)
            st.plotly_chart(fig_is, use_container_width=True)

        # Tabla resumen
        st.markdown('<p class="section-title">Tabla resumen por Estacion</p>',
                    unsafe_allow_html=True)
        tabla = est_grp[["Estacion","N_AT","N_Dias","IF_calc","IS_calc","Severidad","Nivel_IF"]].copy()
        tabla.columns = ["Estacion","N° AT","N° Dias AT","IF","IS","Severidad","Nivel IF"]
        tabla["IF"] = tabla["IF"].round(3)
        tabla["IS"] = tabla["IS"].round(3)
        tabla["Severidad"] = tabla["Severidad"].round(2)

        def color_nivel(val):
            if val == "ALTO":   return "background-color:#FFE5E5;color:#E63946;font-weight:600"
            if val == "MEDIO":  return "background-color:#FFF3E0;color:#F4A261;font-weight:600"
            return "background-color:#E5F7F4;color:#2A9D8F;font-weight:600"

        st.dataframe(
            tabla.style.applymap(color_nivel, subset=["Nivel IF"]),
            use_container_width=True, height=280)

        # IF vs Meta — scatter
        st.markdown('<p class="section-title">Mapa IF vs Severidad — Cuadrante de Riesgo</p>',
                    unsafe_allow_html=True)
        fig_scatter = px.scatter(
            est_grp, x="IF_calc", y="Severidad", size="N_AT",
            color="Nivel_IF", text="Estacion",
            color_discrete_map={"ALTO":"#E63946","MEDIO":"#F4A261","BAJO":"#2A9D8F"},
            title="Cuadrante de riesgo: IF vs Severidad Media (tamaño = N° AT)")
        fig_scatter.add_vline(x=META_IF, line_dash="dash", line_color="#E63946",
                              annotation_text="Meta IF")
        fig_scatter.update_traces(textposition="top center")
        fig_scatter.update_layout(height=380, margin=dict(t=40,b=20))
        st.plotly_chart(fig_scatter, use_container_width=True)

    else:
        st.info("Carga el **Tablero Accidentalidad** en el panel izquierdo para ver los indicadores IF/IS.")
        st.markdown("""
        ### Referencia normativa IF/IS — GTC 3701
        | Indicador | Formula | Meta |
        |---|---|---|
        | **IF** Indice de Frecuencia | (N°AT × 240.000) / HHT | ≤ 2.54 |
        | **IS** Indice de Severidad | (N°Dias × 240.000) / HHT | ≤ 3.00 |
        | **Severidad Media** | N°Dias / N°AT | — |
        | **ILI** | (IF × IS) / 1.000 | — |
        """)

# =============================================================================
# TAB: POR ESTACION
# =============================================================================
if "🏢 Por Estacion" in tab_idx:
    with tabs[tab_idx["🏢 Por Estacion"]]:
        df_gen = tablero_data["GENERAL"]

        estaciones = sorted(df_gen["Estacion"].unique().tolist())
        sel_est = st.selectbox("Selecciona estacion", estaciones)

        df_est = df_gen[df_gen["Estacion"] == sel_est].copy()
        n_at   = int(df_est["N_AT"].sum())
        n_dias = int(df_est["N_Dias_Total"].sum())
        hht    = float(df_est["HHT"].sum())
        if_v   = calc_if(n_at, hht)
        is_v   = calc_is(n_dias, hht)
        sev    = calc_severidad(n_at, n_dias)
        nivel  = nivel_if(if_v)

        st.markdown(f'<p class="section-title">Indicadores — {sel_est}</p>',
                    unsafe_allow_html=True)
        c1,c2,c3,c4,c5 = st.columns(5)
        for col_w, val, label, color in [
            (c1, str(n_at),           "N° AT",          "#1E3A5F"),
            (c2, f"{if_v:.3f}",       "IF Calculado",   "#E63946" if nivel=="ALTO" else "#F4A261" if nivel=="MEDIO" else "#2A9D8F"),
            (c3, f"{META_IF}",        "Meta IF",        "#888"),
            (c4, f"{is_v:.3f}",       "IS Calculado",   "#2E86AB"),
            (c5, f"{sev:.1f} dias",   "Severidad Media","#F4A261"),
        ]:
            col_w.markdown(
                f'<div class="kpi-card"><p class="kpi-value" style="color:{color};">{val}</p>'
                f'<p class="kpi-label">{label}</p></div>', unsafe_allow_html=True)

        st.markdown(f"Nivel IF: {badge_nivel(nivel)}", unsafe_allow_html=True)
        st.markdown("")

        # Top poblaciones por IF
        pop_grp = df_est.groupby("Poblacion").agg(
            N_AT=("N_AT","sum"), N_Dias=("N_Dias_Total","sum"), HHT=("HHT","sum")
        ).reset_index()
        pop_grp["IF"] = pop_grp.apply(lambda r: calc_if(r["N_AT"], r["HHT"]), axis=1)
        pop_grp["IS"] = pop_grp.apply(lambda r: calc_is(r["N_Dias"], r["HHT"]), axis=1)
        pop_grp = pop_grp[pop_grp["N_AT"] > 0].sort_values("IF", ascending=False).head(12)

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown('<p class="section-title">IF por Poblacion (Top 12)</p>',
                        unsafe_allow_html=True)
            fig_pop = px.bar(pop_grp, x="IF", y="Poblacion", orientation="h",
                             color="IF", color_continuous_scale="Reds",
                             text=[f"{v:.2f}" for v in pop_grp["IF"]])
            fig_pop.update_traces(textposition="outside")
            fig_pop.add_vline(x=META_IF, line_dash="dash", line_color="#E63946",
                              annotation_text=f"Meta {META_IF}")
            fig_pop.update_layout(height=380, margin=dict(t=10,b=10),
                                  coloraxis_showscale=False)
            st.plotly_chart(fig_pop, use_container_width=True)

        with col_b:
            st.markdown('<p class="section-title">IS por Poblacion</p>',
                        unsafe_allow_html=True)
            fig_is2 = px.bar(pop_grp.sort_values("IS", ascending=False),
                             x="IS", y="Poblacion", orientation="h",
                             color="IS", color_continuous_scale="Oranges",
                             text=[f"{v:.2f}" for v in pop_grp.sort_values("IS",ascending=False)["IS"]])
            fig_is2.update_traces(textposition="outside")
            fig_is2.update_layout(height=380, margin=dict(t=10,b=10),
                                  coloraxis_showscale=False)
            st.plotly_chart(fig_is2, use_container_width=True)

        st.markdown('<p class="section-title">Detalle por Mes</p>', unsafe_allow_html=True)
        mes_grp = df_est.groupby("Mes").agg(
            N_AT=("N_AT","sum"), N_Dias=("N_Dias_Total","sum"), HHT=("HHT","sum")
        ).reset_index()
        mes_grp["IF"] = mes_grp.apply(lambda r: calc_if(r["N_AT"],r["HHT"]), axis=1)
        mes_grp["IS"] = mes_grp.apply(lambda r: calc_is(r["N_Dias"],r["HHT"]), axis=1)
        st.dataframe(mes_grp.round(3), use_container_width=True)

# =============================================================================
# TAB: EVOLUCION MENSUAL
# =============================================================================
if "📈 Evolucion Mensual" in tab_idx:
    with tabs[tab_idx["📈 Evolucion Mensual"]]:
        df_gen = tablero_data["GENERAL"]
        if_ind, if_ind_meta = load_indicadores(f_tablero.getvalue() if f_tablero else b"")

        st.markdown('<p class="section-title">IF mensual por Estacion — Acumulado 2026</p>',
                    unsafe_allow_html=True)

        meses_disp = ["JAN","FEB","MAR","APR","MAY","JUN","JUL","AUG","SEP","OCT","NOV","DEC"]
        if_plot = if_ind[if_ind["ACUM"] > 0].copy()
        if_plot_m = if_plot.melt(id_vars="Estacion", value_vars=meses_disp,
                                  var_name="Mes", value_name="IF")
        if_plot_m = if_plot_m[if_plot_m["IF"] > 0]

        if not if_plot_m.empty:
            fig_ev = px.line(if_plot_m, x="Mes", y="IF", color="Estacion",
                             markers=True,
                             title="Evolucion IF mensual por Estacion")
            fig_ev.add_hline(y=META_IF, line_dash="dash", line_color="#E63946",
                             annotation_text=f"Meta IF {META_IF}",
                             annotation_position="bottom right")
            fig_ev.update_layout(height=380, margin=dict(t=40,b=20))
            st.plotly_chart(fig_ev, use_container_width=True)
        else:
            st.info("Solo hay datos de Enero y Febrero 2026. La evolucion se mostrara al cargar mas meses.")

        st.markdown('<p class="section-title">Comparativo YTD IF — Estaciones activas</p>',
                    unsafe_allow_html=True)
        acum_df = if_ind[if_ind["ACUM"] > 0][["Estacion","ACUM"]].copy()
        acum_df.columns = ["Estacion","IF_YTD"]
        acum_df = acum_df.sort_values("IF_YTD", ascending=False)
        acum_df["Nivel"] = acum_df["IF_YTD"].apply(nivel_if)

        fig_ytd = px.bar(acum_df, x="Estacion", y="IF_YTD",
                         color="Nivel",
                         color_discrete_map={"ALTO":"#E63946","MEDIO":"#F4A261","BAJO":"#2A9D8F"},
                         text=[f"{v:.2f}" for v in acum_df["IF_YTD"]],
                         title="IF Acumulado YTD 2026 por Estacion")
        fig_ytd.add_hline(y=META_IF, line_dash="dash", line_color="#E63946",
                          annotation_text=f"Meta {META_IF}")
        fig_ytd.update_traces(textposition="outside")
        fig_ytd.update_layout(height=360, margin=dict(t=40,b=20))
        st.plotly_chart(fig_ytd, use_container_width=True)

        st.markdown('<p class="section-title">Tabla IF/IS Indicador vs Total</p>',
                    unsafe_allow_html=True)
        merge = if_ind[["Estacion","JAN","FEB","ACUM"]].merge(
            if_ind_meta[["Estacion","JAN","FEB"]].rename(
                columns={"JAN":"JAN_IND","FEB":"FEB_IND"}),
            on="Estacion", how="left")
        merge.columns = ["Estacion","IF_ENE","IF_FEB","IF_ACUM","IF_IND_ENE","IF_IND_FEB"]
        st.dataframe(merge.round(3), use_container_width=True)

# =============================================================================
# TAB: MODELO ML
# =============================================================================
if "🤖 Modelo ML" in tab_idx:
    with tabs[tab_idx["🤖 Modelo ML"]]:
        art = train_ml(prevsis_bytes, target, algo)

        st.markdown('<p class="section-title">Metricas del modelo</p>', unsafe_allow_html=True)
        m1,m2,m3,m4,m5 = st.columns(5)
        cv_m, cv_s = float(art["cv"].mean()), float(art["cv"].std())
        for col_w, label, val, color in [
            (m1, "AUC-ROC",    f"{art['auc']:.3f}",                    "#1E3A5F"),
            (m2, "CV AUC",     f"{cv_m:.3f}+-{cv_s:.3f}",              "#2E86AB"),
            (m3, "Accuracy",   f"{art['rep']['accuracy']:.1%}",         "#2A9D8F"),
            (m4, "Precision",  f"{art['rep']['1']['precision']:.1%}",   "#F4A261"),
            (m5, "Recall",     f"{art['rep']['1']['recall']:.1%}",      "#E63946"),
        ]:
            col_w.markdown(
                f'<div class="kpi-card"><p class="kpi-value" style="color:{color};">{val}</p>'
                f'<p class="kpi-label">{label}</p></div>', unsafe_allow_html=True)

        st.caption(f"Algoritmo: {algo} | Train: {len(art['X_tr']):,} | Test: {len(art['X_te']):,} | Target: {target}")

        col_roc, col_cm = st.columns(2)
        with col_roc:
            fig_roc = go.Figure()
            fig_roc.add_trace(go.Scatter(x=list(art["fpr"]), y=list(art["tpr"]),
                mode="lines", line=dict(color="#2E86AB",width=2.5),
                name=f"Modelo AUC={art['auc']:.3f}"))
            fig_roc.add_trace(go.Scatter(x=[0,1],y=[0,1],mode="lines",
                line=dict(color="#aaa",dash="dash"), name="Aleatorio"))
            fig_roc.update_layout(title="Curva ROC", xaxis_title="FPR",
                yaxis_title="TPR", height=330, margin=dict(t=35,b=20))
            st.plotly_chart(fig_roc, use_container_width=True)

        with col_cm:
            labels = ["Sin incapacidad","Con incapacidad"]
            fig_cm = px.imshow(art["cm"], text_auto=True,
                color_continuous_scale="Blues", x=labels, y=labels,
                labels=dict(x="Predicho",y="Real",color="Casos"),
                title="Matriz de confusion")
            fig_cm.update_layout(height=330, margin=dict(t=35,b=20))
            st.plotly_chart(fig_cm, use_container_width=True)

        if hasattr(art["model"], "feature_importances_"):
            fi = pd.DataFrame({"Variable": art["feats"],
                               "Pct": art["model"].feature_importances_ * 100})
            fi = fi.sort_values("Pct", ascending=False).head(10)
            fig_fi = px.bar(fi, x="Pct", y="Variable", orientation="h",
                color="Pct", color_continuous_scale="Blues",
                text=[f"{v:.1f}%" for v in fi["Pct"]],
                title="Importancia de variables (%)")
            fig_fi.update_traces(textposition="outside")
            fig_fi.update_layout(height=360, margin=dict(t=35,b=10),
                                 coloraxis_showscale=False)
            st.plotly_chart(fig_fi, use_container_width=True)

        with st.expander("Reporte completo de clasificacion"):
            st.dataframe(pd.DataFrame(art["rep"]).T.round(3), use_container_width=True)

# =============================================================================
# TAB: PREDICCION
# =============================================================================
if "🔮 Prediccion" in tab_idx:
    with tabs[tab_idx["🔮 Prediccion"]]:
        st.markdown('<p class="section-title">Prediccion de riesgo individual</p>',
                    unsafe_allow_html=True)
        st.info("Configura el caso en el panel izquierdo y pulsa **Calcular riesgo**.")

        if predecir_btn:
            art = train_ml(prevsis_bytes, target, algo)
            input_map = {
                "Clasificacion": p_clasif, "Tipo": p_tipo, "Estacion": p_estacion,
                "Genero": p_genero, "Cargo": p_cargo, "Antiguedad": float(p_antiguedad),
                "Naturaleza": p_naturaleza, "Parte_cuerpo": p_parte,
                "Agente": p_agente, "Mecanismo": p_mecanismo,
            }
            row = {}
            for col in art["feats"]:
                if col == "Antiguedad":
                    row[col] = float(p_antiguedad)
                elif col in input_map and col in art["enc"]:
                    le = art["enc"][col]
                    v = input_map[col]
                    row[col] = int(le.transform([v])[0]) if v in le.classes_ else 0
                else:
                    row[col] = 0

            prob  = float(art["model"].predict_proba(pd.DataFrame([row]))[0][1])
            nivel = "Alto" if prob >= .60 else ("Medio" if prob >= .35 else "Bajo")
            bg    = {"Alto":"#FFE5E5","Medio":"#FFF3E0","Bajo":"#E5F7F4"}[nivel]
            tx    = {"Alto":"#E63946","Medio":"#F4A261","Bajo":"#2A9D8F"}[nivel]

            col_g, col_r = st.columns([1,1])
            with col_g:
                st.markdown(
                    f'<div class="pred-box" style="background:{bg};border:2px solid {tx};">'
                    f'<h2 style="color:{tx};font-size:1.6rem;">Riesgo {nivel.upper()}</h2>'
                    f'<p style="color:{tx};font-size:1.3rem;font-weight:700;">{prob:.1%} probabilidad</p>'
                    f'<p style="color:#555;font-size:.82rem;">Variable: {target}</p></div>',
                    unsafe_allow_html=True)

                gauge = go.Figure(go.Indicator(
                    mode="gauge+number", value=round(prob*100,1),
                    number={"suffix":"%","font":{"size":30,"color":tx}},
                    gauge={"axis":{"range":[0,100]},
                           "bar":{"color":tx,"thickness":.25},
                           "steps":[{"range":[0,35],"color":"#E5F7F4"},
                                    {"range":[35,60],"color":"#FFF3E0"},
                                    {"range":[60,100],"color":"#FFE5E5"}],
                           "threshold":{"line":{"color":tx,"width":4},
                                        "thickness":.8,"value":round(prob*100,1)}},
                    title={"text":"Indice de riesgo","font":{"size":13}}))
                gauge.update_layout(height=250, margin=dict(t=30,b=5,l=15,r=15))
                st.plotly_chart(gauge, use_container_width=True)

                # IF hipotetico
                st.markdown("---")
                st.markdown("**Calcular IF e IS para este caso:**")
                hht_inp = st.number_input("HHT (Horas Hombre Trabajadas)", value=200000, step=10000)
                n_dias_inp = st.number_input("Dias de incapacidad esperados", value=3, step=1)
                if hht_inp > 0:
                    if_caso = calc_if(1, hht_inp)
                    is_caso = calc_is(n_dias_inp, hht_inp)
                    st.metric("IF del caso", f"{if_caso:.3f}",
                              delta=f"{if_caso - META_IF:+.3f} vs meta",
                              delta_color="inverse")
                    st.metric("IS del caso", f"{is_caso:.3f}")

            with col_r:
                st.markdown("#### Recomendaciones SST")
                recs = []
                if "Lumbar" in p_naturaleza or "esfuerzo" in p_mecanismo:
                    recs.append("Riesgo ergonomico: revisar levantamiento, faja lumbar.")
                if "Espalda" in p_parte or "Miembros" in p_parte:
                    recs.append("Parte critica: evaluacion medica ocupacional inmediata.")
                if p_antiguedad <= 2:
                    recs.append("Trabajador nuevo: reforzar induccion y asignar tutor.")
                if "Aeronave" in p_agente or "presion" in p_agente:
                    recs.append("Riesgo aeronautico: verificar procedimientos de maniobra.")
                if "Transito" in p_tipo:
                    recs.append("Accidente transito: validar politica de conduccion segura.")
                if "Violencia" in p_tipo:
                    recs.append("Violencia: activar protocolo psicosocial y seguridad corp.")
                if nivel == "Alto":
                    recs.append("NIVEL ALTO: notificar ARL en 24h y activar investigacion AT.")
                if not recs:
                    recs.append("Riesgo moderado. Mantener controles SST habituales.")
                for r in recs:
                    st.markdown(f"- {r}")

                st.markdown("---")
                st.markdown("**Resumen del caso:**")
                for k,v in [
                    ("Cargo", p_cargo), ("Genero", p_genero),
                    ("Antiguedad", f"{p_antiguedad} anos"),
                    ("Naturaleza", p_naturaleza), ("Mecanismo", p_mecanismo),
                    ("Parte", p_parte), ("Tipo", p_tipo), ("Clasificacion", p_clasif)
                ]:
                    st.markdown(f"- **{k}:** {v}")
        else:
            st.markdown(
                '<div style="text-align:center;padding:3rem;color:#aaa;">'
                '<p style="font-size:2.5rem;">🔮</p>'
                '<p>Configura el caso en el panel<br>y pulsa <strong>Calcular riesgo</strong></p>'
                '</div>', unsafe_allow_html=True)

# =============================================================================
# TAB: DATOS
# =============================================================================
if "📋 Datos" in tab_idx:
    with tabs[tab_idx["📋 Datos"]]:
        st.markdown('<p class="section-title">Datos del Tablero y PREVSIS</p>',
                    unsafe_allow_html=True)

        sub1, sub2 = st.tabs(["Tablero Accidentalidad", "Base PREVSIS"])

        with sub1:
            if tablero_data:
                vista = st.radio("Vista", ["GENERAL","AGIL"], horizontal=True)
                df_vista = tablero_data[vista].copy()
                st.dataframe(df_vista, use_container_width=True, height=400)
                buf = BytesIO()
                df_vista.to_excel(buf, index=False)
                st.download_button("Descargar (.xlsx)", buf.getvalue(),
                    file_name=f"tablero_{vista.lower()}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
            else:
                st.info("Carga el Tablero para ver los datos.")

        with sub2:
            if prevsis_bytes:
                df_pv = load_prevsis(prevsis_bytes)
                st.dataframe(df_pv.head(300), use_container_width=True, height=400)
                buf2 = BytesIO()
                df_pv.to_excel(buf2, index=False)
                st.download_button("Descargar PREVSIS (.xlsx)", buf2.getvalue(),
                    file_name="prevsis_datos.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
            else:
                st.info("Carga el archivo PREVSIS para ver los datos.")
