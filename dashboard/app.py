import json
import requests
import pandas as pd, json, urllib.request
from dash import Dash, html, dcc, Input, Output, State
import plotly.graph_objects as go
from math import isnan
from dash import no_update
from dash.dcc import send_data_frame
from joblib import load
import re

# =========================================
# Configuracion LOCAL (Opcion B)
# =========================================
#BASE = "https://aprodigcodata.s3.eu-north-1.amazonaws.com"
API_URL = "http://localhost:8000"  # tu FastAPI local
MASTER_CSV_PATH  = "https://aprodigcodata.s3.eu-north-1.amazonaws.com/maestro_global_variables_municipio.csv"
GEOJSON_MPIO_PATH = "https://aprodigcodata.s3.eu-north-1.amazonaws.com/colombia_municipios.geojson"
GEOJSON_ID_PROP   = "COD_MUNICIPIO" # propiedad del GeoJSON con el código DANE municipal
MODEL_PATH = "model_nivel_piramide.joblib"

# =========================================
#Cargo modelo para predecir el Nivel de piramide y obtener importancia global de features
# =========================================

try:
    trained_pipe = load(MODEL_PATH)  
    clf = trained_pipe.named_steps['clf']

    feature_names = getattr(trained_pipe.named_steps.get('scaler', None), 'feature_names_in_', None)
except Exception as _e:
    trained_pipe, clf, feature_names = None, None, None

def get_global_logreg_importance(clf_obj, feat_names=None, top_k=10):
    """
    Returns a (names, scores) tuple for the most influential features across classes,
    using mean absolute coefficients (features are standardized → magnitudes comparable).
    """
    import numpy as np
    import pandas as pd

    if clf_obj is None or not hasattr(clf_obj, "coef_"):
        return [], []

    coefs = pd.DataFrame(clf_obj.coef_)
    if feat_names is not None and len(feat_names) == coefs.shape[1]:
        coefs.columns = feat_names
    # mean(|coef|) across classes
    abs_mean = coefs.abs().mean(axis=0).sort_values(ascending=False)
    abs_mean = abs_mean.head(top_k)
    return list(abs_mean.index), list(abs_mean.values)

GLOBAL_TOP_FEAT_NAMES, GLOBAL_TOP_FEAT_SCORES = get_global_logreg_importance(clf, feature_names, top_k=10)

# =========================================
# Cargar maestro
# =========================================
maestro = pd.read_csv(MASTER_CSV_PATH)
# Normaliza los codigos de municipio a 5 digitos
maestro["COD_MUNICIPIO"] = (
    maestro["COD_MUNICIPIO"]
    .astype(str)
    .str.strip()
    .str.zfill(5)
)
maestro.columns = maestro.columns.str.strip()

def pick(colnames, candidates):
    for c in candidates:
        if c in colnames:
            return c
    raise KeyError(f"Falta una columna requerida. Probé {candidates} en {list(colnames)}")

# Llaves
COL_DEPTO_COD = pick(maestro.columns, ["COD_DEPARTAMENTO","cod_departamento","DEPTO_COD","DEPARTAMENTO_COD"])
COL_DEPTO_NOM = pick(maestro.columns, ["DEPARTAMENTO","departamento","NOMBRE_DEPARTAMENTO","NOM_DEPARTAMENTO"])
COL_MPIO_COD  = pick(maestro.columns, ["COD_MUNICIPIO","cod_municipio","MUNICIPIO_COD","CODIGO_MUNICIPIO"])
COL_MPIO_NOM  = pick(maestro.columns, ["MUNICIPIO","municipio","NOMBRE_MUNICIPIO","NOM_MUNICIPIO"])

# Indicadores (para mostrar en labels)
COL_INDICE = pick(maestro.columns, ["INDICE","indice","INDICE_FIJO","INDICE_INTERNET"])
COL_IPM    = pick(maestro.columns, ["ipm","ipm_depto","IPM_DEPTO"])
COL_SABER  = pick(maestro.columns, ["saber_punt_global_mean","SABER_PUNT_GLOBAL_MEAN","saber_global"])

# Normaliza tipos/códigos
maestro[COL_DEPTO_COD] = maestro[COL_DEPTO_COD].astype(str).str.strip()
maestro[COL_DEPTO_NOM] = maestro[COL_DEPTO_NOM].astype(str).str.strip()
maestro[COL_MPIO_COD]  = maestro[COL_MPIO_COD].astype(str).str.strip().str.zfill(5)  # asegúrate de 5 dígitos
maestro[COL_MPIO_NOM]  = maestro[COL_MPIO_NOM].astype(str).str.strip()

# Opciones de departamento
deptos_df = maestro[[COL_DEPTO_COD, COL_DEPTO_NOM]].drop_duplicates().sort_values(COL_DEPTO_NOM)
dept_options = [{"label": r[COL_DEPTO_NOM], "value": r[COL_DEPTO_COD]} for _, r in deptos_df.iterrows()]
default_depto = dept_options[0]["value"] if dept_options else None

def municipio_options_for(depto_code: str):
    m = maestro.loc[maestro[COL_DEPTO_COD] == str(depto_code), [COL_MPIO_COD, COL_MPIO_NOM]].drop_duplicates()
    m = m.sort_values(COL_MPIO_NOM)
    return [{"label": row[COL_MPIO_NOM], "value": row[COL_MPIO_COD]} for _, row in m.iterrows()]

COLORBAR_TITLES = {
    "INDICE": "IPI",
    "IPM":    "IPM",
    "SABER":  "Puntaje Saber 11 (promedio)"
}

# =========================================
# Catalogos
# =========================================
AGE_LABELS = {1:"12-17",2:"18-24",3:"25-34",4:"35-44",5:"45-54",6:"55-64",7:"65-75"}
DEFAULT_AGE = 2

P33_OPTIONS = [
    {"label":"Todos los días","value":0},
    {"label":"De 4 a 6 veces a la semana","value":1},
    {"label":"De 2 a 3 veces a la semana","value":2},
    {"label":"Una vez a la semana","value":3},
    {"label":"Una vez cada quince días","value":4},
    {"label":"Una vez al mes","value":5},
    {"label":"Nunca / No usa / NS","value":6},
]
DEFAULT_P33 = 0

def to_float(x):
    if pd.isna(x): return None
    if isinstance(x, str): x = x.replace(",", ".").strip()
    try: return float(x)
    except: return None

# Creo un dataframe de un solo registro con las variables del modelo de prediccion del Nivel de la Piramide 
def build_model_row(municipio_code: str,
                    RANGO_EDAD: float,
                    ESTRATO: float,
                    PB1_bin: float,
                    SEXO_bin: float,
                    P33: float,
                    indicador_value: float):
    """
    Returns (df_row, err) where df_row is a 1-row DataFrame with columns ordered
    as the trained pipeline saw them (using trained_pipe.named_steps['scaler'].feature_names_in_),
    and err is an error string (or None).
    """
    try:
        # Lookups from maestro for this municipio
        code5 = str(municipio_code).zfill(5)
        rowm = maestro.loc[maestro[COL_MPIO_COD] == code5].head(1)
        if rowm.empty:
            return None, f"No se encontró el municipio {code5} en el maestro."

        indice = to_float(rowm.iloc[0][COL_INDICE])
        ipm    = to_float(rowm.iloc[0][COL_IPM])
        saber  = to_float(rowm.iloc[0][COL_SABER])

        # PB = municipio code as int (your original data used DANE-like code)
        PB = float(code5)
       
        feat_dict = {
            "PB": PB,
            "INDICADOR": float(indicador_value),
            "RANGO_EDAD": float(RANGO_EDAD),
            "PB1": float(PB1_bin),
            "SEXO": float(SEXO_bin),
            "ESTRATO": float(ESTRATO),
            "P33": float(P33),
            "ipm_depto": float(ipm) if ipm is not None else None,
            "INDICE": float(indice) if indice is not None else None,
            "saber_punt_global_mean": float(saber) if saber is not None else None,
        }

        # Alinear al modelo
        if trained_pipe and hasattr(trained_pipe.named_steps.get('scaler', None), 'feature_names_in_'):
            cols = list(trained_pipe.named_steps['scaler'].feature_names_in_)
        else:
            cols = list(feat_dict.keys())

        # Revisar que las columnas existan
        missing = [c for c in cols if c not in feat_dict]
        if missing:
            return None, f"Faltan columnas en el ensamblaje del modelo: {missing}"

        import pandas as pd
        X1 = pd.DataFrame([[feat_dict[c] for c in cols]], columns=cols)

        # Revisar valores faltantes
        if X1.isna().any().any():
            na_cols = list(X1.columns[X1.isna().any()])
            return None, f"Valores faltantes para el modelo en columnas: {na_cols}"

        return X1, None
    except Exception as e:
        return None, f"Error armando features del modelo: {e}"


def series_metric(metric_key: str):
    base = maestro.set_index(COL_MPIO_COD)
    if metric_key == "INDICE":
        s = base[COL_INDICE].apply(to_float)
        title = "IPI (Índice de penetración de internet)"
    elif metric_key == "IPM":
        s = base[COL_IPM].apply(to_float)
        title = "IPM (Índice de pobreza multidimensional)"
    else:
        s = base[COL_SABER].apply(to_float)
        title = "Puntaje Saber (promedio)"
    return s, title

def all_geojson_codes(gj, id_prop):
    return [str(f["properties"].get(id_prop, "")).zfill(5) for f in gj.get("features", [])]


def geojson_locations_and_hover(geojson, z_series):
    """
    locations: todos los MpCodigo del geojson (garantiza que se dibuje TODO el país)
    z_values : toma z_series por código, deja None si falta
    customdata: (Depto, Municipio, Valor) para hover
    """
    locs = []
    zvals = []
    custom = []
    base = maestro.set_index(COL_MPIO_COD)

    for feat in geojson.get("features", []):
        props = feat["properties"]
        code = str(props.get(GEOJSON_ID_PROP, "")).zfill(5)
        locs.append(code)

        val = z_series.get(code, None)
        zvals.append(val if pd.notna(val) else None)

        # Nombre desde maestro si existe; si no, desde geojson
        if code in base.index:
            muni = base.loc[code, COL_MPIO_NOM]
            dept = base.loc[code, COL_DEPTO_NOM] if COL_DEPTO_NOM in base.columns else props.get("DEPTO", "")
        else:
            muni = props.get("MPIO_CCDGO", "")
            dept = props.get("DEPTO", "")
        custom.append([dept, muni, val])

    return locs, zvals, custom


# =========================================
# Carga GeoJSON LOCAL
# =========================================

with urllib.request.urlopen(GEOJSON_MPIO_PATH) as r:
    geojson_mpios = json.load(r)



# Normaliza MpCodigo a 5 dígitos
for feat in geojson_mpios.get("features", []):
    props = feat.get("properties", {})

    # 1) Intenta construir COD_MUNICIPIO = DPTO(2) + MPIO(3)
    dpto = re.sub(r"\D", "", str(props.get("DPTO_CCDGO", "")).strip())
    mpio = re.sub(r"\D", "", str(props.get("MPIO_CCDGO", "")).strip())
    if dpto and mpio:
        code = dpto[-2:].zfill(2) + mpio[-3:].zfill(3)
    else:
        # 2) Fallback
        legacy = str(props.get("MpCodigo", props.get("MPIO_CCDGO", ""))).strip()
        legacy = re.sub(r"\D", "", legacy)
        code = legacy[-5:].zfill(5) if legacy else ""

    props["COD_MUNICIPIO"] = code 
# =========================================
# App Dash
# =========================================
app = Dash(__name__)
server = app.server
app.title = "Apropiación Digital en Colombia"

app.layout = html.Div(
    
    style={"maxWidth":"1200px","margin":"0 auto","fontFamily":"system-ui, -apple-system, Segoe UI, Roboto, Arial","padding":"16px"},
    children=[
        dcc.Store(id="pred_mpio"),
        dcc.Store(id="last_result"),
        dcc.Download(id="download_results"),
        html.H2("Apropiación Digital en Colombia", style={"textAlign":"center","background":"#146c94","color":"white","padding":"10px","borderRadius":"8px"}),

        html.Div(style={"display":"grid","gridTemplateColumns":"1fr 1fr 1fr","gap":"16px"}, children=[
            # -------- Columna 1: Vivienda --------
            html.Div(style={"border":"1px solid #ddd","borderRadius":"10px","padding":"14px","background":"#e9f7fb"},
                     children=[
                html.H4("Filtros Vivienda"),
                html.Label("Departamento"),
                dcc.Dropdown(id="depto_dropdown", options=dept_options, value=default_depto, clearable=False),
                html.Br(),
                html.Label("Municipio"),
                dcc.Dropdown(
                    id="municipio_dropdown",
                    options=municipio_options_for(default_depto) if default_depto else [],
                    value=(municipio_options_for(default_depto)[0]["value"] if default_depto and municipio_options_for(default_depto) else None),
                    placeholder="Seleccione un municipio…",
                    clearable=False
                ),
                html.Br(),
                html.Label("Estrato"),
                dcc.Slider(id="ESTRATO", min=1, max=6, step=1, value=3, marks={i:str(i) for i in range(1,7)}),
                html.Br(),
                html.Label("Vivienda (PB1)"),
                dcc.Dropdown(id="PB1_bin", options=[{"label":"Urbana","value":1},{"label":"Rural","value":0}], value=1, clearable=False),
                html.Br(),
                
            ]),

            # -------- Columna 2: Individuo --------
            html.Div(style={"border":"1px solid #ddd","borderRadius":"10px","padding":"14px","background":"#f6fbff"},
                     children=[
                html.H4("Filtros Individuo"),
                html.Label("Rango de edad"),
                dcc.Slider(id="RANGO_EDAD", min=1, max=7, step=1, value=DEFAULT_AGE, marks=AGE_LABELS),
                html.Br(),
                html.Label("Sexo"),
                dcc.Dropdown(id="SEXO_bin", options=[{"label":"Masculino","value":1},{"label":"Femenino","value":0}], value=1, clearable=False),
                html.Br(),
                html.Label("Frecuencia de uso de internet (P33)"),
                dcc.Dropdown(id="P33", options=P33_OPTIONS, value=DEFAULT_P33, clearable=False),
                html.Div(
                    "0: Todos los días · 1: 4-6/sem · 2: 2-3/sem · 3: 1/sem · 4: quincenal · 5: mensual · 6: nunca",
                    id="p33_hint", style={"fontSize":"12px","color":"#666","marginTop":"8px"}
                ),
                html.Br(),
                html.Div(
                [
                    html.Button(
                        "Predecir",
                        id="btn_predict",
                        n_clicks=0,
                        style={"padding":"10px 20px","fontWeight":"600"}
                    ),
                    html.Button(
                        "Descargar resultados",
                        id="btn_download",
                        n_clicks=0,
                        disabled=True,                          # <- deshabilitado hasta predecir
                        style={"marginLeft":"8px","padding":"10px 16px"}
                    ),
                ],
                style={"display":"flex","alignItems":"center","gap":"8px"}
            ),
            ]),

            # -------- Columna 3: Resultados --------
            html.Div(style={"border":"1px solid #ddd","borderRadius":"10px","padding":"14px","background":"#eef8f0"},
                     children=[
                html.H4("Resultados"),
                html.Label("Indicador de apropiación digital"),
                html.Div(id="prediction_area",
                         style={"border":"1px solid #ccc","padding":"14px","borderRadius":"8px","textAlign":"center","fontSize":"20px","background":"white"}),
                html.Br(),
                html.H5("Pirámide de apropiación (predicción)"),
                dcc.Graph(id="pyramid_graph", config={"displayModeBar": False},style={"height":"260px"}),
                html.H5("Variables más relevantes Nivel de la Pirámide"),
                dcc.Graph(id="top_features_graph", config={"displayModeBar": False},style={"height":"300px"}),
                
                html.Br(),
                html.H5("Indicadores del municipio"),
                html.Div([
                    html.Div([
                    html.B("IPI (Índice de penetración de internet):"),
                    html.Br(),
                    html.Span(id="lbl_ipi", style={"fontSize":"18px"})
                ], style={"marginBottom":"10px"}),

                html.Div([
                    html.B("IPM (Índice de pobreza multidimensional):"),
                    html.Br(),
                    html.Span(id="lbl_ipm", style={"fontSize":"18px"})
                ], style={"marginBottom":"10px"}),

                html.Div([
                    html.B("Puntaje Saber (promedio):"),
                    html.Br(),
                    html.Span(id="lbl_saber", style={"fontSize":"18px"})
                ])
                ], style={"lineHeight":"1.6"}),
                html.Br(),
                html.Div(id="error_area", style={"color":"#b00020","display":"none"}),
                html.Div(id="debug_area", style={"color":"#888","fontSize":"12px","marginTop":"12px"}),
            ]),
        ]),
        html.Div([
            
            html.H4("Mapa de Colombia", style={"margin":"8px 0"}),
            html.Div([
                html.Div([
                    html.Label("Métrica del mapa", style={"fontWeight":"600"}),
                    dcc.Dropdown(
                        id="metric_dropdown",
                        options=[
                            {"label":"IPI (Índice de penetración de internet)","value":"INDICE"},
                            {"label":"IPM (Índice de pobreza multidimensional)","value":"IPM"},
                            {"label":"Puntaje Saber (promedio)","value":"SABER"},
                        ],
                        value="INDICE",
                        clearable=False,
                        style={"maxWidth":"480px"}
                    )
                ], style={"marginBottom":"8px"}),
                html.Div(
                    dcc.Graph(id="mapa_colombia_full", style={"height":"700px"}),
                    style={"overflow":"auto","maxWidth":"100%"}
                )
                
            ], style={"border":"1px solid #ddd","borderRadius":"10px","padding":"12px","background":"#f5fbff"})
        ], style={"gridColumn":"1 / -1","marginTop":"16px"}),  # ocupa todo el ancho
    ]
)

# =========================================
# Callbacks
# =========================================
@app.callback(
    Output("municipio_dropdown","options"),
    Output("municipio_dropdown","value"),
    Input("depto_dropdown","value"),
    State("municipio_dropdown","value"),
)
def _filter_mpios(depto_value, current_mpio):
    if not depto_value:
        return [], None
    opts = municipio_options_for(depto_value)
    valid = {o["value"] for o in opts}
    new_val = current_mpio if current_mpio in valid else (opts[0]["value"] if opts else None)
    return opts, new_val

@app.callback(
    Output("download_results","data"),
    Input("btn_download","n_clicks"),
    State("last_result","data"),
    prevent_initial_call=True
)
def _download_results(n_clicks, data):
    if not data:
        return no_update
    df = pd.DataFrame([data])
    return send_data_frame(df.to_csv, "resultados_nivel_piramide.csv", index=False)

@app.callback(
    Output("mapa_colombia_full","figure"),
    Output("lbl_ipi","children"),
    Output("lbl_ipm","children"),
    Output("lbl_saber","children"),
    Input("municipio_dropdown","value"),
    Input("metric_dropdown","value"),
    Input("pred_mpio","data"),  # ⬅️ prefer highlight of last prediction
)
def _update_map_and_labels(mpio_from_dropdown, metric_key, mpio_pred):
    # labels (from maestro)
    row = maestro.loc[maestro[COL_MPIO_COD] == str(mpio_from_dropdown).zfill(5)].head(1)
    ipi   = to_float(row.iloc[0][COL_INDICE]) if not row.empty else None
    ipm   = to_float(row.iloc[0][COL_IPM]) if not row.empty else None
    saber = to_float(row.iloc[0][COL_SABER]) if not row.empty else None

    # metric series
    z_series, z_title = series_metric(metric_key)

    # locations: all polygons → full country drawn
    locations_all = all_geojson_codes(geojson_mpios, GEOJSON_ID_PROP)

    # z aligned to all polygons
    z_map = []
    custom = []
    base = maestro.set_index(COL_MPIO_COD)
    for code in locations_all:
        val = z_series.get(code, None)
        z_map.append(val if pd.notna(val) else None)

        if code in base.index:
            muni = base.loc[code, COL_MPIO_NOM]
            dept = base.loc[code, COL_DEPTO_NOM] if COL_DEPTO_NOM in base.columns else ""
        else:
            # fallback to geojson props
            feat = next((f for f in geojson_mpios["features"] if str(f["properties"][GEOJSON_ID_PROP]) == code), None)
            props = feat["properties"] if feat else {}
            muni = props.get("MPIO_CCDGO", "")
            dept = props.get("DEPTO", "")
        custom.append([dept, muni, val])

    fig = go.Figure()

    # 1) Base light-gray: makes borders clear everywhere
    fig.add_trace(go.Choropleth(
        geojson=geojson_mpios,
        locations=locations_all,
        z=[0]*len(locations_all),
        featureidkey=f"properties.{GEOJSON_ID_PROP}",
        colorscale=[[0, "#f0f0f0"], [1, "#f0f0f0"]],
        marker_line_color="#aaaaaa",
        marker_line_width=0.35,
        showscale=False,
        hoverinfo="skip",
    ))

    # 2) Metric layer
    vals = [v for v in z_map if v is not None]
    zmin = min(vals) if vals else 0
    zmax = max(vals) if vals else 1

    fig.add_trace(go.Choropleth(
        geojson=geojson_mpios,
        locations=locations_all,
        z=z_map,
        featureidkey=f"properties.{GEOJSON_ID_PROP}",
        colorscale="Blues",
        zmin=zmin, zmax=zmax,
        marker_line_color="#999999",
        marker_line_width=0.25,
        showscale=True,
        colorbar=dict(
            title=dict(
                text=COLORBAR_TITLES.get(metric_key, z_title),  # ← título según la métrica
                side="top"                                       # opcional: título encima de la barra
            ),
            len=0.7,          # un poco más alta
            thickness=14,
            x=1.05,           # mueve la barra un poquito a la derecha
            xanchor="left",
            y=0.5,
            yanchor="middle",
            xpad=10
        ),
        customdata=custom,
        hovertemplate="<b>%{customdata[1]}</b><br>Depto: %{customdata[0]}<br>Valor: %{customdata[2]:.2f}<extra></extra>",
    ))

    # 3) Highlight municipio (prefer predicted; if not, dropdown)
    hl_code = (str(mpio_pred).zfill(5) if mpio_pred else str(mpio_from_dropdown).zfill(5)) if (mpio_pred or mpio_from_dropdown) else None
    if hl_code:
        fig.add_trace(go.Choropleth(
            geojson=geojson_mpios,
            locations=[hl_code],
            z=[0],
            featureidkey=f"properties.{GEOJSON_ID_PROP}",
            colorscale=[[0, "rgba(0,0,0,0)"], [1, "rgba(0,0,0,0)"]],
            marker_line_color="#dabc26",
            marker_line_width=3.0,  # thick border
            showscale=False,
            hoverinfo="skip",
        ))

    # view / size
    fig.update_geos(
        fitbounds="geojson",
        visible=False,
        projection_type="mercator"
    )
    fig.update_layout(
        margin=dict(l=0, r=130, t=0, b=0),  # antes tenías r=0
        height=720,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )

    def fmt(x): 
        return "-" if x is None else (f"{x:.2f}" if isinstance(x, (int,float)) else str(x))

    return fig, fmt(ipi), fmt(ipm), fmt(saber)

def build_pyramid_figure(pred_class: int = None, class_probs: list = None):
    import plotly.graph_objects as go

    levels = [0, 1, 2, 3]  # 0 = nivel más alto, 3 = más bajo
    labels = [f"Nivel {lvl}" for lvl in levels]
    widths = [0.4, 0.65, 0.85, 1.0]

    # y es [0,1,2,3]
    y = list(range(len(levels)))

    def level_color(lvl):
        return "#146c94" if pred_class == lvl else "#cfe8f3"

    fig = go.Figure()
    for lvl in levels:
        txt = labels[lvl]
        if class_probs and len(class_probs) == 4:
            pct = f" ({class_probs[lvl]*100:.1f}%)"
            txt += pct
        fig.add_trace(go.Bar(
            x=[widths[lvl]],
            y=[lvl],                      # usa el mismo índice
            orientation='h',
            marker=dict(
                color=level_color(lvl),
                line=dict(color="#2a4d69", width=1 if pred_class == lvl else 0.5)
            ),
            hovertemplate=txt + "<extra></extra>",
            text=[txt],
            textposition="inside",
            insidetextanchor="middle",
            showlegend=False
        ))

    # Ticks exactamente con y = [0,1,2,3]
    fig.update_yaxes(
        tickvals=levels,
        ticktext=[f"Nivel {lvl}" for lvl in levels],
        autorange="reversed",   # <- para que el 0 quede arriba y 3 abajo como pirámide
        showgrid=False,
        zeroline=False
    )
    fig.update_xaxes(visible=False, range=[0, 1.05])
    fig.update_layout(
        height=260,
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)"
    )
    return fig


def build_top_features_bar(names, scores, title=""):
    import plotly.graph_objects as go
    if not names or not scores:
        return go.Figure()
    fig = go.Figure(go.Bar(
        x=scores[::-1],
        y=names[::-1],
        orientation='h',
        marker=dict(line=dict(width=0))
    ))
    fig.update_layout(
        height=300,
        margin=dict(l=10, r=10, t=30, b=10),
        title=title,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis_title="Imp. (|coef| medio)",
        yaxis_title=""
    )
    return fig

# La API enriquece IPI/IPM/Saber internamente
@app.callback(
    Output("prediction_area","children"),
    Output("pyramid_graph","figure"),            
    Output("top_features_graph","figure"),       
    Output("error_area","children"),
    Output("error_area","style"),
    Output("debug_area","children"),
    Output("pred_mpio","data"), 
    Output("btn_download","disabled"),          
    Output("last_result","data"),                    
    Input("btn_predict","n_clicks"),
    State("depto_dropdown","value"),
    State("municipio_dropdown","value"),
    State("RANGO_EDAD","value"),
    State("ESTRATO","value"),
    State("PB1_bin","value"),
    State("SEXO_bin","value"),
    State("P33","value"),
    prevent_initial_call=True
)

def _predict(n_clicks, depto_code, municipio_code, RANGO_EDAD, ESTRATO, PB1_bin, SEXO_bin, P33):
    if not municipio_code:
        empty_fig = build_pyramid_figure(None, None), build_top_features_bar([], [])
        return ("Seleccione un municipio.", empty_fig[0], empty_fig[1], "", {"display":"none"}, "", no_update, True, None)
    

    payload = {
        "municipio_code": str(municipio_code),
        "RANGO_EDAD": float(RANGO_EDAD),
        "ESTRATO": float(ESTRATO),
        "PB1_bin": float(PB1_bin),
        "SEXO_bin": float(SEXO_bin),
        "P33": float(P33),
        "dept_code": str(depto_code).strip() if depto_code else None
    }

    url = f"{API_URL}/predict"
    try:
    # 1) API -> INDICADOR
        payload_api = {
            "municipio_code": str(municipio_code),
            "RANGO_EDAD": float(RANGO_EDAD),
            "ESTRATO": float(ESTRATO),
            "PB1_bin": float(PB1_bin),
            "SEXO_bin": float(SEXO_bin),
            "P33": float(P33),
            "dept_code": str(depto_code).strip() if depto_code else None
        }
        resp = requests.post(url, json=payload_api, timeout=20)

        if resp.status_code != 200:
            empty_py = build_pyramid_figure(None, None)
            empty_tf = build_top_features_bar([], [])
            return ("", empty_py, empty_tf,
                    f"Error {resp.status_code} al llamar a la API: {resp.text}",
                    {"color":"#b00020","display":"block"},
                    "",
                    no_update,
                    True,
                    None)  

        data = resp.json()
        indicador_val = data.get("indicador") or data.get("INDICADOR") or data.get("prediction")
        if indicador_val is None:
            empty_py = build_pyramid_figure(None, None)
            empty_tf = build_top_features_bar([], [])
            return ("", empty_py, empty_tf,
                    f"La API no devolvió 'indicador'. Respuesta: {data}",
                    {"color":"#b00020","display":"block"},
                    "",
                    no_update,
                    True,
                    None)
        
        indicador_val = float(str(indicador_val).replace(",", "."))
        indicador_text = f"{indicador_val:.4f}"

        # 2) Ensamblar entrada modelo
        if trained_pipe is None:
            empty_py = build_pyramid_figure(None, None)
            empty_tf = build_top_features_bar([], [])
            return ("", empty_py, empty_tf,
                    "Modelo no cargado en el servidor (model_nivel_piramide.joblib).",
                    {"color":"#b00020","display":"block"},
                    "",
                    no_update,
                    True,
                    None)

        SEXO_orig = 1.0 if float(SEXO_bin) == 1.0 else 2.0


        X1, err = build_model_row(
            municipio_code=municipio_code,
            RANGO_EDAD=RANGO_EDAD,
            ESTRATO=ESTRATO,
            PB1_bin=PB1_bin,
            SEXO_bin=SEXO_orig,
            P33=P33,
            indicador_value=indicador_val
        )
        if err:
            empty_py = build_pyramid_figure(None, None)
            empty_tf = build_top_features_bar([], [])
            return ("", empty_py, empty_tf,
                    err,
                    {"color":"#b00020","display":"block"},
                    "",
                    no_update,
                    True,
                    None)

        # 3) Predicccón
        pred_class = int(trained_pipe.predict(X1)[0])
        proba = trained_pipe.predict_proba(X1)[0].tolist() if hasattr(trained_pipe, "predict_proba") else None
        classes = list(trained_pipe.named_steps['clf'].classes_) if hasattr(trained_pipe.named_steps['clf'], "classes_") else []

        pyramid_fig = build_pyramid_figure(pred_class=pred_class, class_probs=proba if (proba and len(proba)==4) else None)
        top_feats_fig = build_top_features_bar(GLOBAL_TOP_FEAT_NAMES, GLOBAL_TOP_FEAT_SCORES) if (GLOBAL_TOP_FEAT_NAMES and GLOBAL_TOP_FEAT_SCORES) else build_top_features_bar([], [])

        # 4) Armar payload para CSV

        row_dict = X1.to_dict(orient='records')[0]
        row_dict.update({
            "municipio_code": str(municipio_code).zfill(5),
            "depto_code": str(depto_code).strip() if depto_code else None,
            "nivel_piramide_pred": int(pred_class),
        })
        if proba and classes:
            for c, p in zip(classes, proba):
                row_dict[f"p_{c}"] = p

        return (indicador_text,
                pyramid_fig,
                top_feats_fig,
                "",
                {"display":"none"},
                "",
                str(municipio_code).zfill(5),
                False,          # habilita botón descargar
                row_dict)       # datos para CSV
    except Exception as e:
        empty_py = build_pyramid_figure(None, None)
        empty_tf = build_top_features_bar([], [])
        return (
            "", empty_py, empty_tf,
            "mensaje de error...",
            {"color":"#b00020","display":"block"},
            "debug opcional o ''",
            no_update,
            True, 
            None 
        )

if __name__ == "__main__":
    # Dash >= 2.16
    app.run(host="localhost", port=8050, debug=True)
