"""
Simulador Interactivo de Indicadores de Concentración de Mercado
Método de Monte Carlo con cuotas de mercado generadas por una distribución
de Dirichlet(1, ..., 1), que es uniforme sobre el simplex y garantiza
que la suma de cuotas sea exactamente 1 en cada iteración.
"""

import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# 1. FUNCIONES DE CÁLCULO DE INDICADORES
#    Todas reciben cuotas en proporción (0 a 1). Aceptan un vector (N,)
#    o una matriz (iteraciones, N).
# ---------------------------------------------------------------------------


def calcular_cr(s, k):
    """Ratio de concentración CRk: suma de las k mayores cuotas, en %."""
    s = np.asarray(s, dtype=float)
    ordenado = np.sort(s, axis=-1)[..., ::-1]
    return ordenado[..., :k].sum(axis=-1) * 100


def calcular_ihh(s):
    """Índice de Herfindahl-Hirschman: suma de cuotas^2 (cuotas en %), rango 10000/N a 10000."""
    s = np.asarray(s, dtype=float)
    return (s ** 2).sum(axis=-1) * 10000


def calcular_id(s):
    """Índice de Dominancia: ID = sum_i (s_i^2 / IHH)^2, con IHH en proporciones. Rango [1/N, 1]."""
    s = np.asarray(s, dtype=float)
    h = (s ** 2).sum(axis=-1, keepdims=True)
    return ((s ** 2 / h) ** 2).sum(axis=-1)


def calcular_ie(s):
    """Índice de Entropía de Shannon: IE = -sum s_i ln(s_i). Rango [0, ln N]. Mayor IE = menor concentración."""
    s = np.asarray(s, dtype=float)
    seguro = np.clip(s, 1e-300, None)  # evita log(0); 0*ln(0) se toma como 0
    return -(s * np.log(seguro)).sum(axis=-1)


def valor_indicador(nombre, s, k=4):
    if nombre == "CRk":
        return calcular_cr(s, k)
    if nombre == "IHH":
        return calcular_ihh(s)
    if nombre == "ID":
        return calcular_id(s)
    return calcular_ie(s)


UNIDADES = {
    "CRk": "Ratio de concentración CRk (%)",
    "IHH": "IHH (puntos, escala 0 a 10.000)",
    "ID": "Índice de Dominancia (0 a 1)",
    "IE": "Entropía de Shannon (nats)",
}

# ---------------------------------------------------------------------------
# 2. CLASIFICACIÓN SEGÚN UMBRALES TEÓRICOS
# ---------------------------------------------------------------------------


def clasificar(nombre, valor, n):
    """Devuelve 'Baja', 'Moderada' o 'Alta' y el texto de los umbrales usados."""
    if nombre == "CRk":
        # Umbrales convencionales (referidos a CR4)
        txt = "CRk < 40 % → baja; 40 % a 70 % → moderada; > 70 % → alta"
        nivel = "Baja" if valor < 40 else ("Moderada" if valor <= 70 else "Alta")
    elif nombre == "IHH":
        # Umbrales clásicos del Departamento de Justicia de EE.UU. (guía 2010)
        txt = "IHH < 1.500 → baja; 1.500 a 2.500 → moderada; > 2.500 → alta"
        nivel = "Baja" if valor < 1500 else ("Moderada" if valor <= 2500 else "Alta")
    elif nombre == "ID":
        txt = "ID < 0,25 → baja; 0,25 a 0,50 → moderada; > 0,50 → alta (umbrales didácticos)"
        nivel = "Baja" if valor < 0.25 else ("Moderada" if valor <= 0.5 else "Alta")
    else:
        # Entropía normalizada E = IE / ln(N): 1 = reparto perfectamente parejo
        e_norm = valor / np.log(n)
        txt = (f"Entropía normalizada IE/ln(N) = {e_norm:.3f}; "
               "> 0,80 → baja; 0,50 a 0,80 → moderada; < 0,50 → alta (umbrales didácticos)")
        nivel = "Baja" if e_norm > 0.8 else ("Moderada" if e_norm >= 0.5 else "Alta")
    return nivel, txt


# ---------------------------------------------------------------------------
# 3. MOTOR DE SIMULACIÓN MONTE CARLO
# ---------------------------------------------------------------------------


@st.cache_data(show_spinner=False)
def simular(n, iteraciones, semilla):
    """Genera 'iteraciones' vectores de cuotas con Dirichlet(1,...,1).
    Cada fila suma 1 por construcción (se normaliza otra vez para blindar errores de redondeo)."""
    rng = np.random.default_rng(semilla)
    cuotas = rng.dirichlet(np.ones(n), size=iteraciones)
    cuotas = cuotas / cuotas.sum(axis=1, keepdims=True)
    return cuotas


# ---------------------------------------------------------------------------
# 4. INTERFAZ
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Simulador de Concentración", layout="wide")
st.title("Simulador de Concentración de Mercado (Monte Carlo)")
st.caption("Organización Industrial: comparación de un caso particular contra una distribución estocástica de cuotas.")

# ---- Barra lateral: parámetros ----
st.sidebar.header("Parámetros")

indicador = st.sidebar.selectbox(
    "Indicador de concentración",
    ["CRk", "IHH", "ID", "IE"],
    format_func=lambda x: {
        "CRk": "Ratio de Concentración (CRk)",
        "IHH": "Índice de Herfindahl-Hirschman (IHH)",
        "ID": "Índice de Dominancia (ID)",
        "IE": "Índice de Entropía (IE)",
    }[x],
)

N = st.sidebar.number_input("Número de empresas (N)", min_value=2, max_value=100, value=10, step=1)
N = int(N)

k = 4
if indicador == "CRk":
    k = st.sidebar.slider("k (empresas más grandes)", min_value=1, max_value=N, value=min(4, N))

ITER_MIN, ITER_MAX = 100, 50000
iteraciones = st.sidebar.number_input(
    "Iteraciones de Monte Carlo",
    min_value=ITER_MIN, max_value=ITER_MAX, value=1000, step=100,
    help=(f"Mínimo {ITER_MIN:,}: con menos, el histograma es inestable. "
          f"Máximo {ITER_MAX:,}: más allá, la memoria y la latencia del servidor web "
          "crecen sin mejorar visiblemente la precisión."),
)
iteraciones = int(iteraciones)

st.sidebar.warning(
    "⚠️ Carga computacional: la simulación crea una matriz de iteraciones × N números. "
    "Al subir las iteraciones (y/o N) aumentan el tiempo de respuesta, el uso de memoria y CPU. "
    f"Sobre 10.000 iteraciones puede notarse lentitud en servidores gratuitos; el tope es {ITER_MAX:,}."
)

semilla = st.sidebar.number_input("Semilla aleatoria (reproducibilidad)", min_value=0, value=42, step=1)

if iteraciones * N > 1_000_000:
    st.sidebar.info(f"Se simularán {iteraciones * N:,} valores; puede tardar unos segundos.")

# ---- Caso particular ----
st.header("1. Caso particular")
modo = st.radio("¿Cómo definir las cuotas?", ["Entrada manual", "Generación aleatoria"], horizontal=True)

if "caso_aleatorio" not in st.session_state:
    st.session_state.caso_aleatorio = None

cuotas_caso = None

if modo == "Entrada manual":
    por_defecto = ", ".join([f"{100 / N:.2f}"] * N)
    texto = st.text_area(
        f"Ingresa {N} cuotas en % separadas por coma (deben sumar 100)",
        value=por_defecto,
        key=f"manual_{N}",
    )
    try:
        valores = [float(x.strip().replace(",", ".")) for x in texto.replace(";", ",").split(",") if x.strip()]
        if len(valores) != N:
            st.error(f"Ingresaste {len(valores)} valores, pero N = {N}.")
        elif any(v < 0 or v > 100 for v in valores):
            st.error("Cada cuota debe estar entre 0 % y 100 %.")
        elif abs(sum(valores) - 100) > 0.01:
            st.error(f"Las cuotas suman {sum(valores):.2f} %; deben sumar 100 %.")
        else:
            cuotas_caso = np.array(valores) / 100
    except ValueError:
        st.error("Formato inválido: usa solo números separados por comas (ej. 40, 30, 20, 10).")
else:
    if st.button("Generar caso aleatorio") or st.session_state.caso_aleatorio is None \
            or len(st.session_state.caso_aleatorio) != N:
        st.session_state.caso_aleatorio = np.random.default_rng().dirichlet(np.ones(N))
    cuotas_caso = st.session_state.caso_aleatorio

if cuotas_caso is None:
    st.stop()

st.write("Cuotas del caso (%):", np.round(np.sort(cuotas_caso)[::-1] * 100, 2).tolist())

# ---- Simulación ----
st.header("2. Distribución de Monte Carlo")
with st.spinner("Simulando..."):
    matriz = simular(N, iteraciones, int(semilla))

suma_max_error = float(np.abs(matriz.sum(axis=1) - 1.0).max())
st.caption(f"Verificación de cierre: máximo |Σsᵢ − 1| = {suma_max_error:.2e} en {iteraciones:,} iteraciones.")

dist = valor_indicador(indicador, matriz, k)
valor_caso = float(valor_indicador(indicador, cuotas_caso, k))
percentil = float((dist <= valor_caso).mean() * 100)

fig, ax = plt.subplots(figsize=(9, 4.5))
ax.hist(dist, bins=40, color="#4C78A8", alpha=0.85, edgecolor="white", density=True)
ax.axvline(valor_caso, color="crimson", linewidth=2.5,
           label=f"Caso particular = {valor_caso:,.3f} (percentil {percentil:.1f})")
ax.set_xlabel(UNIDADES[indicador] + (f" con k = {k}" if indicador == "CRk" else ""))
ax.set_ylabel("Densidad (frecuencia relativa)")
ax.set_title(f"Distribución empírica de {indicador}: N = {N}, {iteraciones:,} iteraciones")
ax.legend()
ax.grid(alpha=0.3)
st.pyplot(fig)

c1, c2, c3 = st.columns(3)
c1.metric("Valor del caso", f"{valor_caso:,.3f}")
c2.metric("Percentil en la simulación", f"{percentil:.1f} %")
c3.metric("Media simulada", f"{dist.mean():,.3f}")

# ---- Evaluador ----
st.header("3. Evaluador")
st.write(f"Según el indicador **{indicador}**, ¿cómo clasificarías la concentración del caso particular?")
respuesta = st.radio("Tu respuesta", ["Baja", "Moderada", "Alta"], horizontal=True, index=None)

if st.button("Validar respuesta"):
    if respuesta is None:
        st.warning("Selecciona una opción primero.")
    else:
        correcta, umbrales = clasificar(indicador, valor_caso, N)
        explicacion = (
            f"**Valor calculado:** {valor_caso:,.3f}\n\n"
            f"**Umbrales usados:** {umbrales}\n\n"
            f"**Posición relativa:** el caso supera al {percentil:.1f} % de los mercados simulados "
            f"con {N} empresas (percentil {percentil:.1f}).\n\n"
        )
        if respuesta == correcta:
            st.success(f"✅ ¡Correcto! La concentración es **{correcta}**.")
        else:
            st.error(f"❌ Incorrecto. La concentración correcta es **{correcta}** (tú respondiste {respuesta}).")
        st.markdown(explicacion)
        st.info(
            "Nota: un mismo caso puede ser 'alto' según el umbral absoluto y estar en un percentil bajo "
            "(o al revés) respecto de la distribución simulada; ambas lecturas se complementan. "
            "Para el IE, un valor más alto significa MENOR concentración."
        )

with st.expander("Fórmulas utilizadas"):
    st.latex(r"CR_k=\sum_{i=1}^{k}s_{(i)}\qquad IHH=\sum_{i=1}^{N}(100\,s_i)^2")
    st.latex(r"ID=\sum_{i=1}^{N}\left(\frac{s_i^2}{\sum_j s_j^2}\right)^2\qquad IE=-\sum_{i=1}^{N}s_i\ln s_i")
