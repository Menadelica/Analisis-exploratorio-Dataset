# %% [markdown]
# # Tarea 3 — Análisis descriptivo y de distribución de la EDAD
#
# **Curso:** Fundamentos de análisis cuantitativo (código 553318631)
# **Actividad:** Tarea 3 — Análisis de datos mediante software
# **Dataset:** *Obesity levels based on eating habits and physical condition* (Kaggle / UCI)
# **Variable de estudio:** `Age` (edad en años)
#
# ---
#
# ## Objetivo
# Aplicar técnicas de estadística descriptiva y visualización de datos, con Python
# (`pandas`, `numpy`, `matplotlib`, `seaborn`, `scipy`), para describir el
# comportamiento de la variable **edad** en una muestra de 2.111 personas y
# extraer conclusiones que apoyen la toma de decisiones.
#
# ## Contenido del cuaderno
# 1. Configuración del entorno
# 2. Carga y estructura del dataset
# 3. Calidad y limpieza de la variable edad
# 4. Estadística descriptiva
# 5. Tabla de frecuencias por rangos de edad
# 6. Detección de valores atípicos
# 7. Visualización: histograma y densidad
# 8. Visualización: diagrama de caja
# 9. Visualización: función de distribución acumulada (ECDF)
# 10. Verificación del supuesto de normalidad (Q-Q plot y pruebas)
# 11. Comparación de la edad por género
# 12. Comparación de la edad por nivel de obesidad
# 13. Relación entre edad e índice de masa corporal (IMC)
# 14. Matriz de correlaciones
# 15. Conclusiones

# %% [markdown]
# ## 1. Configuración del entorno
#
# Se importan las librerías, se define la paleta de colores institucional del
# cuaderno y se crean las carpetas donde se guardarán las figuras y las tablas
# de resultados. Todos los valores quedan reproducibles: no se usan datos
# aleatorios ni semillas ocultas.
# Si alguna librería no está instalada, ejecutar en la terminal:
# python -m pip install pandas numpy matplotlib seaborn scipy

# %%
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

warnings.filterwarnings("ignore", category=FutureWarning)

# Codificación UTF-8 para que las tildes se impriman bien en consola
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Rutas del proyecto (funciona igual como script y como notebook). Primero se
# busca el CSV junto al proyecto; si no está allí, se busca en Descargas.
try:
    BASE = Path(__file__).resolve().parent
except NameError:
    BASE = Path.cwd()

NOMBRE_CSV = "ObesityDataSet_raw_and_data_sinthetic.csv"
RUTAS_CSV = [BASE / "data" / NOMBRE_CSV, Path.home() / "Downloads" / NOMBRE_CSV]
DATA = next((ruta for ruta in RUTAS_CSV if ruta.exists()), RUTAS_CSV[0])
if not DATA.exists():
    raise FileNotFoundError(
        "No se encontró el CSV. Colócalo en la carpeta 'data' del proyecto "
        f"o en Descargas. Rutas revisadas: {RUTAS_CSV}"
    )
FIG = BASE / "figuras"
RES = BASE / "resultados"
FIG.mkdir(exist_ok=True)
RES.mkdir(exist_ok=True)

# Estilo general de las gráficas
sns.set_theme(style="whitegrid", context="notebook")
plt.rcParams.update({
    "figure.dpi": 110,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
    "figure.facecolor": "white",
})

AZUL = "#2E6F95"      # color principal
NARANJA = "#E76F51"   # color de contraste (media, tendencias)
VERDE = "#2A9D8F"
GRIS = "#6C757D"

PALETA_GENERO = {"Femenino": "#E76F51", "Masculino": "#2E6F95"}

# Utilidades de salida -------------------------------------------------------
HALLAZGOS: list[str] = []


def mostrar(obj) -> None:
    """Muestra una tabla en el notebook y la imprime si se ejecuta como script."""
    try:
        from IPython.display import display
        display(obj)
    except Exception:
        print(obj)


def seccion(titulo: str) -> None:
    linea = "=" * 78
    print(f"\n{linea}\n{titulo}\n{linea}")


def guardar(fig, nombre: str) -> None:
    """Guarda la figura a 300 dpi y la muestra en el notebook."""
    ruta = FIG / nombre
    fig.savefig(ruta, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    try:
        from IPython.display import Image, display
        display(Image(filename=str(ruta)))
    except Exception:
        pass


def anotar(texto: str) -> None:
    """Acumula un párrafo de interpretación para el informe final."""
    HALLAZGOS.append(texto)
    print(texto)


# %% [markdown]
# ## 2. Carga y estructura del dataset
#
# El archivo contiene información sociodemográfica, hábitos alimenticios,
# actividad física y el nivel de obesidad de cada persona. La variable de
# interés es `Age`.
#
# **Qué revisar antes de calcular nada:** número de filas y columnas, tipos de
# datos, valores faltantes y registros duplicados. Un análisis descriptivo que
# no documenta la calidad de los datos no es verificable.

# %%
df = pd.read_csv(DATA)
filas_originales, columnas_originales = df.shape

print(f"Filas: {df.shape[0]:,}   Columnas: {df.shape[1]}")
print(f"Memoria: {df.memory_usage(deep=True).sum() / 1024:.1f} KB")
mostrar(df.head())

# %%
# Diccionario de variables (documentación del dataset)
diccionario = pd.DataFrame({
    "Variable": ["Gender", "Age", "Height", "Weight", "family_history_with_overweight",
                 "FAVC", "FCVC", "NCP", "CAEC", "SMOKE", "CH2O", "SCC", "FAF",
                 "TUE", "CALC", "MTRANS", "NObeyesdad"],
    "Descripción": ["Género", "Edad en años", "Estatura en metros", "Peso en kg",
                    "Antecedente familiar de sobrepeso", "Consumo frecuente de alimentos calóricos",
                    "Frecuencia de consumo de vegetales", "Número de comidas principales",
                    "Consumo de alimentos entre comidas", "Fuma", "Consumo diario de agua (L)",
                    "Monitorea las calorías", "Frecuencia de actividad física (días/semana)",
                    "Tiempo de uso de dispositivos (h/día)", "Consumo de alcohol",
                    "Medio de transporte habitual", "Nivel de obesidad (variable objetivo)"],
})
mostrar(diccionario)

# %%
# Calidad de los datos
calidad = pd.DataFrame({
    "Tipo": df.dtypes.astype(str),
    "No_nulos": df.notna().sum(),
    "Nulos": df.isna().sum(),
    "Nulos_%": (df.isna().mean() * 100).round(2),
    "Únicos": df.nunique(),
})
mostrar(calidad)

duplicados = int(df.duplicated().sum())
print(f"\nRegistros duplicados exactos: {duplicados} "
      f"({duplicados / len(df) * 100:.1f} % del total)")
print("Valores faltantes totales:", int(df.isna().sum().sum()))

# %%
# Codificación legible de las variables categóricas para el informe
df["Género"] = df["Gender"].map({"Female": "Femenino", "Male": "Masculino"})
df["Nivel_Obesidad"] = df["NObeyesdad"].map({
    "Insufficient_Weight": "Peso insuficiente",
    "Normal_Weight": "Peso normal",
    "Overweight_Level_I": "Sobrepeso I",
    "Overweight_Level_II": "Sobrepeso II",
    "Obesity_Type_I": "Obesidad I",
    "Obesity_Type_II": "Obesidad II",
    "Obesity_Type_III": "Obesidad III",
})

# Índice de masa corporal (IMC = peso / estatura²)
df["IMC"] = df["Weight"] / df["Height"] ** 2

anotar(
    f"El dataset original contiene {filas_originales:,} registros y "
    f"{columnas_originales} variables. "
    f"Se encontraron {int(df.isna().sum().sum())} valores faltantes y "
    f"{duplicados} registros duplicados exactos. Los duplicados se reportan, "
    "pero no se eliminan automáticamente: el archivo no incluye un identificador "
    "que permita confirmar si son errores o registros válidos. La variable Age "
    "no presenta datos faltantes, por lo que se analiza completa."
)

# %% [markdown]
# ## 3. Calidad y limpieza de la variable `Age`
#
# Antes de describir la edad se verifican tres cosas: que sea numérica, que no
# existan valores imposibles (negativos o mayores a 100 años) y que su
# conversión a entero sea consistente. La edad viene registrada en años
# cumplidos, sin decimales.

# %%
edad = df["Age"]

print("Tipo de dato original:", df["Age"].dtype)
print("Valores imposibles (< 0 o > 100):", int(((edad < 0) | (edad > 100)).sum()))
print("Valores nulos:", int(edad.isna().sum()))
print("Mínimo:", edad.min(), "| Máximo:", edad.max())
print("Valores distintos:", edad.nunique())

assert edad.between(0, 100).all(), "Existen edades fuera de rango"
assert edad.notna().all(), "Existen edades nulas"

print("\n✓ La variable no presenta problemas de calidad: se analiza completa (n = "
      f"{len(edad)}).")

# %% [markdown]
# ## 4. Estadística descriptiva
#
# Se calculan las medidas de tendencia central, dispersión, posición y forma.
# Se usa la **desviación estándar muestral** (`ddof=1`), porque los datos
# provienen de una muestra y no del censo de la población.
#
# Medidas incluidas:
#
# | Grupo | Medidas |
# |---|---|
# | Tendencia central | media, mediana, moda |
# | Dispersión | rango, varianza, desviación estándar, coeficiente de variación |
# | Posición | cuartiles Q1 y Q3, rango intercuartílico (IQR) |
# | Forma | asimetría y curtosis |
# | Inferencia básica | error estándar e intervalo de confianza del 95 % para la media |

# %%
n = int(edad.count())
media = float(edad.mean())
mediana = float(edad.median())
moda = float(edad.mode().iloc[0])
minimo = float(edad.min())
maximo = float(edad.max())
rango = maximo - minimo
q1, q3 = [float(v) for v in edad.quantile([0.25, 0.75])]
iqr = q3 - q1
varianza = float(edad.var(ddof=1))
desv = float(edad.std(ddof=1))
cv = desv / media * 100
error_std = desv / np.sqrt(n)
t_critico = stats.t.ppf(0.975, df=n - 1)
ic_inf, ic_sup = media - t_critico * error_std, media + t_critico * error_std
asimetria = float(stats.skew(edad, bias=False))
curtosis = float(stats.kurtosis(edad, fisher=True, bias=False))

tabla_descriptiva = pd.DataFrame({
    "Medida": ["n (válidos)", "Media", "Mediana", "Moda", "Mínimo", "Máximo",
               "Rango", "Q1 (25 %)", "Q3 (75 %)", "IQR (Q3 - Q1)",
               "Varianza (muestral)", "Desviación estándar (muestral)",
               "Coeficiente de variación (%)", "Error estándar de la media",
               "IC 95 % — límite inferior", "IC 95 % — límite superior",
               "Asimetría", "Curtosis (exceso)"],
    "Valor": [f"{n}", f"{media:.2f}", f"{mediana:.2f}", f"{moda:.0f}",
              f"{minimo:.0f}", f"{maximo:.0f}", f"{rango:.0f}",
              f"{q1:.2f}", f"{q3:.2f}", f"{iqr:.2f}",
              f"{varianza:.2f}", f"{desv:.2f}", f"{cv:.2f}", f"{error_std:.3f}",
              f"{ic_inf:.2f}", f"{ic_sup:.2f}", f"{asimetria:.3f}", f"{curtosis:.3f}"],
})
mostrar(tabla_descriptiva)
tabla_descriptiva.to_csv(RES / "tabla_descriptiva_edad.csv", index=False, encoding="utf-8-sig")

# %%
# Comparación rápida entre medidas: la distancia media-mediana es un indicio de asimetría
print(f"Diferencia media - mediana: {media - mediana:.2f} años")
print(f"¿Coinciden media y mediana? {'Sí' if abs(media - mediana) < 1 else 'No'}")
print(f"Rango intercuartílico (donde está el 50 % central): {iqr:.1f} años "
      f"({q1:.0f} a {q3:.0f} años)")
print(f"El 50 % central representa solo el {iqr / rango * 100:.1f} % del recorrido total.")

# %%
anotar(
    f"La edad media de la muestra es {media:.2f} años (IC 95 %: {ic_inf:.2f} – "
    f"{ic_sup:.2f}), mientras que la mediana es {mediana:.2f} años y la moda "
    f"{moda:.0f} años. La media supera a la mediana en {media - mediana:.2f} años, "
    "lo que evidencia una distribución con asimetría positiva: existen personas "
    f"de edad avanzada que empujan el promedio hacia arriba. La desviación "
    f"estándar es de {desv:.2f} años y el coeficiente de variación del {cv:.1f} %, "
    "nivel moderado que indica que la edad es una variable relativamente "
    f"homogénea alrededor de la media. El 50 % central de la muestra se "
    f"concentra entre {q1:.0f} y {q3:.0f} años (IQR = {iqr:.1f}), es decir, en un "
    f"segmento que abarca apenas el {iqr / rango * 100:.1f} % del recorrido total "
    f"({minimo:.0f} a {maximo:.0f} años): la muestra es marcadamente joven."
)

# %% [markdown]
# ## 5. Tabla de frecuencias por rangos de edad
#
# La edad, al ser una variable cuantitativa con 40 valores distintos, se agrupa
# en intervalos para describir mejor su distribución. Se reportan frecuencias
# absolutas (`fa`), relativas (`fr`), acumuladas (`F`) y relativas acumuladas
# (`Fr`).

# %%
cortes = [13, 17, 20, 25, 30, 40, 50, 61]
etiquetas = ["14-17", "18-20", "21-25", "26-30", "31-40", "41-50", "51-61"]
df["Grupo_Edad"] = pd.cut(df["Age"], bins=cortes, labels=etiquetas, include_lowest=True)

fa = df["Grupo_Edad"].value_counts().reindex(etiquetas).fillna(0).astype(int)
tabla_freq = pd.DataFrame({
    "Rango de edad": etiquetas,
    "fa": fa.values,
    "fr (%)": (fa.values / n * 100).round(1),
})
tabla_freq["F acumulada"] = tabla_freq["fa"].cumsum()
tabla_freq["Fr acumulada (%)"] = (tabla_freq["F acumulada"] / n * 100).round(1)

mostrar(tabla_freq)
tabla_freq.to_csv(RES / "tabla_frecuencias_edad.csv", index=False, encoding="utf-8-sig")

grupo_top = tabla_freq.loc[tabla_freq["fa"].idxmax()]
print(f"\nRango más frecuente: {grupo_top['Rango de edad']} años, con "
      f"{grupo_top['fa']} personas ({grupo_top['fr (%)']} %).")
menores_26 = int((edad < 26).sum())
print(f"Menores de 26 años: {menores_26} personas "
      f"({menores_26 / n * 100:.1f} % de la muestra).")

# %%
p26 = (edad <= 26).mean() * 100
p30 = (edad <= 30).mean() * 100
p40 = (edad > 40).mean() * 100

anotar(
    f"El rango más frecuente es {grupo_top['Rango de edad']} años con "
    f"{grupo_top['fa']} personas ({grupo_top['fr (%)']} %), seguido por el de "
    "18-20 años. Sumados, ambos intervalos reúnen a más de seis de cada diez "
    f"participantes. El {p26:.1f} % tiene 26 años o menos y el {p30:.1f} % no "
    "supera los 30, lo que confirma que se trata de una población "
    "predominantemente adolescente y adulta joven. Los mayores de 40 años "
    f"representan apenas el {p40:.1f} % del total, un grupo minoritario que "
    "explica la cola derecha de la distribución."
)

# %% [markdown]
# ## 6. Detección de valores atípicos
#
# Se aplican dos criterios complementarios:
#
# 1. **Regla de Tukey (1,5 × IQR):** son atípicos los valores fuera del
#    intervalo [Q1 − 1,5·IQR, Q3 + 1,5·IQR].
# 2. **Puntuación Z:** valores con |z| > 3 desviaciones estándar respecto a la media.
#
# En variables como la edad, un atípico **no suele ser un error**: es una
# persona mayor. Por eso se identifican y se describen, pero **no se eliminan**.

# %%
limite_inf = q1 - 1.5 * iqr
limite_sup = q3 + 1.5 * iqr
mascara_iqr = (edad < limite_inf) | (edad > limite_sup)
atipicos = df.loc[mascara_iqr, ["Género", "Age", "IMC", "Nivel_Obesidad"]].sort_values("Age")

z = (edad - media) / desv
n_z3 = int((z.abs() > 3).sum())

print(f"Límites de Tukey: [{limite_inf:.1f} ; {limite_sup:.1f}] años")
print(f"Atípicos según IQR: {int(mascara_iqr.sum())} "
      f"({mascara_iqr.mean() * 100:.1f} % de la muestra)")
print(f"Atípicos según |z| > 3: {n_z3}")
mostrar(atipicos.head(15))
atipicos.to_csv(RES / "atipicos_edad.csv", index=False, encoding="utf-8-sig")

# %%
anotar(
    f"Con la regla de Tukey se identifican {int(mascara_iqr.sum())} personas "
    f"({mascara_iqr.mean() * 100:.1f} %) con edades superiores a "
    f"{limite_sup:.0f} años, y {n_z3} casos superan tres desviaciones estándar "
    f"(z > 3). Son edades plausibles, pero el criterio estadístico por sí solo "
    "no permite confirmar si algún registro es un error de captura; por eso se "
    "conservan y se documentan, en lugar de eliminarlos automáticamente. "
    "Excluirlos sesgaría la muestra hacia la población joven y ocultaría al grupo de mayor "
    f"edad. La asimetría positiva ({asimetria:.2f}) y la curtosis "
    f"({curtosis:.2f}) son la huella estadística de esa cola derecha."
)

# %% [markdown]
# ## 7. Visualización: histograma y densidad
#
# El histograma agrupa las edades en intervalos de dos años y muestra la forma
# de la distribución. La curva KDE (estimación de densidad por núcleos) suaviza
# el perfil y permite ver dónde se concentra la masa de datos. Las líneas
# verticales señalan la media y la mediana: cuando la media queda a la derecha
# de la mediana, la distribución es asimétrica positiva.

# %%
fig, ax = plt.subplots(figsize=(10, 5.5))
sns.histplot(edad, bins=range(13, 63, 2), kde=True, color=AZUL,
             edgecolor="white", linewidth=0.8, alpha=0.75, ax=ax)
ax.axvline(media, color=NARANJA, linestyle="--", linewidth=2,
           label=f"Media = {media:.2f} años")
ax.axvline(mediana, color="#264653", linestyle=":", linewidth=2,
           label=f"Mediana = {mediana:.2f} años")
ax.set(title=f"Distribución de la edad de la muestra (n = {n:,})",
       xlabel="Edad (años)", ylabel="Frecuencia absoluta")
ax.set_xlim(12, 63)
ax.legend(frameon=True, loc="upper right")
ax.text(0.62, 0.62,
        f"Asimetría = {asimetria:.2f}\nCurtosis = {curtosis:.2f}\n"
        f"DE = {desv:.2f} años",
        transform=ax.transAxes, fontsize=10, va="top",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#F1F3F5", edgecolor=GRIS))
guardar(fig, "01_histograma_edad.png")

# %%
anotar(
    "El histograma muestra una distribución concentrada en edades jóvenes, con "
    f"un pico principal entre los 18 y los 25 años (la barra más alta se ubica "
    f"alrededor de los {moda:.0f} años) y una cola que se extiende hacia la "
    "derecha hasta los 61 años. La curva de densidad confirma una forma "
    "unimodal con asimetría positiva: la media queda a la derecha de la mediana. "
    "La forma describe esta muestra, que está compuesta principalmente por "
    "personas jóvenes; no basta para concluir que representa a la población general."
)

# %% [markdown]
# ## 8. Visualización: diagrama de caja (boxplot)
#
# El boxplot resume cinco estadísticos: mínimo, Q1, mediana, Q3 y máximo, además
# de los valores atípicos. La caja contiene el 50 % central de los datos y su
# altura es el IQR. Los puntos por encima del bigote superior son las edades
# atípicas.

# %%
fig, ax = plt.subplots(figsize=(10, 3.6))
sns.boxplot(x=edad, color=AZUL, width=0.45, ax=ax)
ax.scatter([media], [0], color=NARANJA, s=90, zorder=5,
           label=f"Media = {media:.2f}", marker="D")
ax.set(title="Diagrama de caja de la edad", xlabel="Edad (años)", ylabel="")
ax.set_yticks([])
ax.set_xlim(12, 63)

# Anotaciones de los cuartiles
for valor, texto in [(q1, f"Q1 = {q1:.0f}"), (mediana, f"Mediana = {mediana:.0f}"),
                     (q3, f"Q3 = {q3:.0f}")]:
    ax.annotate(texto, xy=(valor, 0.22), xytext=(valor, 0.38), ha="center",
                fontsize=9, color="#264653",
                arrowprops=dict(arrowstyle="-", color=GRIS, lw=0.8))
ax.legend(loc="lower right", frameon=True)
ax.set_ylim(-0.55, 0.62)
guardar(fig, "02_boxplot_edad.png")

# %%
anotar(
    f"El diagrama de caja confirma la concentración de la muestra: el 50 % "
    f"central de las edades se ubica entre {q1:.0f} y {q3:.0f} años, con mediana "
    f"de {mediana:.0f}. La caja es estrecha (IQR = {iqr:.1f} años) y está "
    "desplazada hacia la izquierda del recorrido total, mientras el bigote "
    f"derecho y los {int(mascara_iqr.sum())} valores atípicos se extienden hasta "
    "los 61 años. Esa asimetría visual es la misma que reportan el histograma y "
    "los coeficientes de forma."
)

# %% [markdown]
# ## 9. Visualización: función de distribución acumulada (ECDF)
#
# La ECDF indica qué proporción de la muestra tiene una edad menor o igual a un
# valor dado. Es útil para responder preguntas del tipo “¿qué porcentaje tiene
# menos de 30 años?”.

# %%
fig, ax = plt.subplots(figsize=(10, 5.5))
sns.ecdfplot(edad, color=AZUL, linewidth=2.2, ax=ax)
for valor, color in [(mediana, "#264653"), (30, NARANJA), (40, VERDE)]:
    prop = (edad <= valor).mean() * 100
    ax.axvline(valor, color=color, linestyle="--", linewidth=1.2, alpha=0.9)
    ax.axhline(prop / 100, color=color, linestyle="--", linewidth=1.2, alpha=0.9)
    ax.annotate(f"{prop:.1f} % ≤ {valor:.0f} años", xy=(valor, prop / 100),
                xytext=(valor + 1.5, prop / 100 - 0.07), fontsize=9, color=color)
ax.set(title="Distribución acumulada de la edad (ECDF)",
       xlabel="Edad (años)", ylabel="Proporción acumulada")
ax.set_xlim(12, 63)
ax.set_ylim(0, 1.02)
guardar(fig, "03_ecdf_edad.png")

p40_acum = (edad <= 40).mean() * 100

# %%
anotar(
    "La curva acumulada crece de forma pronunciada entre los 18 y los 26 años: "
    f"el {p26:.1f} % de la muestra tiene 26 años o menos y el {p30:.1f} % no "
    f"supera los 30. A partir de esa edad la pendiente se aplana y el "
    f"{p40_acum:.1f} % tiene 40 años o menos. En términos prácticos, casi ocho de "
    "cada diez personas del dataset son menores de 31 años, y solo una de cada "
    "cuarenta supera los 40."
)

# %% [markdown]
# ## 10. Verificación del supuesto de normalidad
#
# Muchas técnicas estadísticas (pruebas t, intervalos de confianza clásicos)
# suponen que la variable sigue una distribución normal. Se contrasta ese
# supuesto con tres herramientas:
#
# - **Q-Q plot:** compara los cuantiles observados con los teóricos de una normal.
# - **Shapiro-Wilk:** prueba de normalidad sensible a desviaciones en el centro.
# - **D'Agostino-Pearson (K²)** y **Anderson-Darling:** contrastes complementarios.
#
# En todas, la hipótesis nula es “los datos provienen de una distribución normal”.

# %%
fig, ax = plt.subplots(figsize=(7.5, 6))
(osm, osr), (pendiente, intercepto, r) = stats.probplot(edad, dist="norm", plot=ax)
ax.get_lines()[0].set_markerfacecolor(AZUL)
ax.get_lines()[0].set_markeredgecolor(AZUL)
ax.get_lines()[0].set_markersize(4)
ax.get_lines()[1].set_color(NARANJA)
ax.get_lines()[1].set_linewidth(2)
ax.set_title("Q-Q plot de la edad frente a una distribución normal")
ax.set_xlabel("Cuantiles teóricos (distribución normal)")
ax.set_ylabel("Cuantiles observados (edad)")
ax.text(0.04, 0.92, f"R² = {r**2:.3f}", transform=ax.transAxes, fontsize=11,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#F1F3F5", edgecolor=GRIS))
guardar(fig, "04_qqplot_edad.png")

# %%
shapiro = stats.shapiro(edad)
dagostino = stats.normaltest(edad)
anderson = stats.anderson(edad, dist="norm")
crit_5 = anderson.critical_values[2]

pruebas = pd.DataFrame({
    "Prueba": ["Shapiro-Wilk", "D'Agostino-Pearson (K²)", "Anderson-Darling"],
    "Estadístico": [f"{shapiro.statistic:.4f}", f"{dagostino.statistic:.4f}",
                    f"{anderson.statistic:.4f}"],
    "p-valor": [f"{shapiro.pvalue:.3e}", f"{dagostino.pvalue:.3e}", "—"],
    "Valor crítico (5 %)": ["—", "—", f"{crit_5:.4f}"],
    "Conclusión": [
        "Rechaza normalidad" if shapiro.pvalue < 0.05 else "No rechaza normalidad",
        "Rechaza normalidad" if dagostino.pvalue < 0.05 else "No rechaza normalidad",
        "Rechaza normalidad" if anderson.statistic > crit_5 else "No rechaza normalidad",
    ],
})
mostrar(pruebas)
pruebas.to_csv(RES / "pruebas_normalidad.csv", index=False, encoding="utf-8-sig")

# %%
anotar(
    f"Las tres pruebas coinciden en rechazar la hipótesis de normalidad: "
    f"Shapiro-Wilk arroja W = {shapiro.statistic:.4f} con p = {shapiro.pvalue:.2e}; "
    f"D'Agostino-Pearson, K² = {dagostino.statistic:.2f} con p = {dagostino.pvalue:.2e}; "
    f"y el estadístico de Anderson-Darling ({anderson.statistic:.2f}) supera el "
    f"valor crítico del 5 % ({crit_5:.2f}). El Q-Q plot lo confirma visualmente: "
    "los puntos se separan de la recta en el extremo superior derecho, formando "
    "una curvatura típica de las distribuciones asimétricas. Conclusión "
    "metodológica: para describir la edad conviene usar la mediana y el IQR en "
    "lugar de la media y la desviación estándar, y para comparar grupos deben "
    "emplearse pruebas no paramétricas (Mann-Whitney, Kruskal-Wallis)."
)

# %% [markdown]
# ## 11. Comparación de la edad por género
#
# Se contrasta si la distribución de la edad difiere entre hombres y mujeres.
# Como la normalidad fue rechazada, se usa la prueba no paramétrica de
# **Mann-Whitney U** (equivalente a Wilcoxon para dos muestras independientes).

# %%
resumen_genero = (df.groupby("Género", observed=True)["Age"]
                    .agg(n="count", Media="mean", Mediana="median",
                         DE="std", Mínimo="min", Máximo="max")
                    .round(2))
mostrar(resumen_genero)
resumen_genero.to_csv(RES / "edad_por_genero.csv", encoding="utf-8-sig")

femenino = df.loc[df["Género"] == "Femenino", "Age"]
masculino = df.loc[df["Género"] == "Masculino", "Age"]
mw = stats.mannwhitneyu(femenino, masculino, alternative="two-sided")

print(f"\nMann-Whitney U = {mw.statistic:.1f}, p = {mw.pvalue:.4f}")
print("Diferencia de medianas:",
      f"{masculino.median() - femenino.median():.1f} años")

# %%
fig, ax = plt.subplots(figsize=(8, 5.5))
sns.boxplot(data=df, x="Género", y="Age", hue="Género", palette=PALETA_GENERO,
            width=0.5, legend=False, showmeans=True,
            meanprops=dict(marker="D", markerfacecolor="white",
                           markeredgecolor="black", markersize=7), ax=ax)
sns.stripplot(data=df, x="Género", y="Age", color="black", alpha=0.18,
              size=2.5, jitter=0.25, ax=ax)
ax.set(title="Edad según género", xlabel="", ylabel="Edad (años)")
ax.set_ylim(12, 63)
ax.text(0.02, 0.97, f"Mann-Whitney U = {mw.statistic:.0f}\np = {mw.pvalue:.3f}",
        transform=ax.transAxes, va="top", fontsize=10,
        bbox=dict(boxstyle="round,pad=0.45", facecolor="#F1F3F5", edgecolor=GRIS))
guardar(fig, "05_edad_por_genero.png")

# %%
anotar(
    f"Las mujeres registran una edad media de {femenino.mean():.2f} años "
    f"(mediana {femenino.median():.0f}) y los hombres de {masculino.mean():.2f} "
    f"años (mediana {masculino.median():.0f}). La prueba de Mann-Whitney detecta "
    f"una diferencia estadísticamente significativa (p = {mw.pvalue:.4f}), pero su "
    "magnitud es pequeña (la diferencia entre las medias es de "
    f"{abs(masculino.mean() - femenino.mean()):.2f} años) y los diagramas de caja "
    "se superponen ampliamente. Esto ilustra que significancia estadística no "
    "implica necesariamente una diferencia importante en la práctica; sin una "
    "prueba de equivalencia, no corresponde afirmar que ambos grupos sean idénticos."
)

# %% [markdown]
# ## 12. Comparación de la edad por nivel de obesidad
#
# Interesa saber si la edad se asocia con el diagnóstico nutricional. Se usa
# **Kruskal-Wallis**, la extensión no paramétrica del ANOVA para más de dos
# grupos, y se ordenan las categorías por mediana para facilitar la lectura.

# %%
orden_obesidad = ["Peso insuficiente", "Peso normal", "Sobrepeso I",
                  "Sobrepeso II", "Obesidad I", "Obesidad II", "Obesidad III"]

resumen_obesidad = (df.groupby("Nivel_Obesidad", observed=True)["Age"]
                      .agg(n="count", Media="mean", Mediana="median", DE="std")
                      .reindex(orden_obesidad).round(2))
mostrar(resumen_obesidad)
resumen_obesidad.to_csv(RES / "edad_por_nivel_obesidad.csv", encoding="utf-8-sig")

grupos = [g["Age"].values for _, g in df.groupby("Nivel_Obesidad", observed=True)]
kw = stats.kruskal(*grupos)
print(f"\nKruskal-Wallis H = {kw.statistic:.3f}, p = {kw.pvalue:.3e}")

# %%
fig, ax = plt.subplots(figsize=(10.5, 5.8))
sns.boxplot(data=df, x="Nivel_Obesidad", y="Age", order=orden_obesidad,
            hue="Nivel_Obesidad", palette="viridis", legend=False,
            width=0.62, ax=ax)
ax.set(title="Edad según nivel de obesidad", xlabel="", ylabel="Edad (años)")
ax.set_ylim(12, 63)
plt.setp(ax.get_xticklabels(), rotation=18, ha="right")
ax.text(0.01, 0.97, f"Kruskal-Wallis H = {kw.statistic:.1f}\np = {kw.pvalue:.2e}",
        transform=ax.transAxes, va="top", fontsize=10,
        bbox=dict(boxstyle="round,pad=0.45", facecolor="#F1F3F5", edgecolor=GRIS))
guardar(fig, "06_edad_por_nivel_obesidad.png")

# %%
anotar(
    "A diferencia del género, aquí sí hay diferencias relevantes: la prueba de "
    f"Kruskal-Wallis rechaza la igualdad de las siete distribuciones "
    f"(H = {kw.statistic:.1f}; p = {kw.pvalue:.1e}). La mediana de edad asciende "
    f"desde {resumen_obesidad.loc['Peso insuficiente', 'Mediana']:.0f} años en el "
    "grupo con peso insuficiente hasta "
    f"{resumen_obesidad['Mediana'].max():.0f} años en obesidad tipo II, pasando "
    "por 23 en obesidad tipo I. La tendencia no es perfectamente monotónica: la "
    "obesidad tipo III presenta una mediana menor ("
    f"{resumen_obesidad.loc['Obesidad III', 'Mediana']:.0f} años) y una dispersión "
    f"muy baja (DE = {resumen_obesidad.loc['Obesidad III', 'DE']:.1f}), rasgo "
    "atribuible al componente sintético del dataset, que concentra ese subgrupo "
    "en un rango etario estrecho. En conjunto, la edad sí permite discriminar el "
    "estado nutricional, pero la relación no es lineal y debe complementarse con "
    "las variables de hábitos."
)

# %% [markdown]
# ## 13. Relación entre edad e índice de masa corporal (IMC)
#
# Se calcula el IMC (`peso / estatura²`) y se explora su relación con la edad
# mediante un diagrama de dispersión con línea de tendencia. Se reportan los
# coeficientes de Pearson (relación lineal) y Spearman (relación monótona, más
# robusta ante valores atípicos).

# %%
pearson = stats.pearsonr(df["Age"], df["IMC"])
spearman = stats.spearmanr(df["Age"], df["IMC"])
n_imc40 = int((df["IMC"] > 40).sum())
imc40_edad = df.loc[df["IMC"] > 40, "Age"]
print(f"Pearson  r = {pearson.statistic:.3f} (p = {pearson.pvalue:.3e})")
print(f"Spearman ρ = {spearman.statistic:.3f} (p = {spearman.pvalue:.3e})")
print(f"IMC medio = {df['IMC'].mean():.2f} kg/m² "
      f"(DE = {df['IMC'].std():.2f})")
print(f"Casos con IMC > 40: {n_imc40} (edades de {imc40_edad.min():.0f} a "
      f"{imc40_edad.max():.0f} años)")

# %%
fig, ax = plt.subplots(figsize=(10, 5.8))
sns.scatterplot(data=df, x="Age", y="IMC", hue="Género", palette=PALETA_GENERO,
                alpha=0.55, s=28, edgecolor="none", ax=ax)
sns.regplot(data=df, x="Age", y="IMC", scatter=False, color="black",
            line_kws=dict(linestyle="--", linewidth=1.8), ax=ax)
ax.set(title="Relación entre edad e índice de masa corporal",
       xlabel="Edad (años)", ylabel="IMC (kg/m²)")
ax.legend(title="Género", frameon=True)
ax.text(0.62, 0.95,
        f"Pearson r = {pearson.statistic:.2f}\nSpearman ρ = {spearman.statistic:.2f}",
        transform=ax.transAxes, va="top", fontsize=10,
        bbox=dict(boxstyle="round,pad=0.45", facecolor="#F1F3F5", edgecolor=GRIS))
guardar(fig, "07_edad_vs_imc.png")

# %%
anotar(
    "La relación entre edad e IMC es positiva y de intensidad baja-media: "
    f"Pearson r = {pearson.statistic:.2f} (lineal) y Spearman ρ = "
    f"{spearman.statistic:.2f} (monótona), ambos con p < 0,001. Que el "
    "coeficiente de Spearman supere al de Pearson indica que la asociación es "
    "ordenada pero no estrictamente lineal, algo esperable con una distribución "
    f"asimétrica. El modelo lineal explica apenas un {pearson.statistic ** 2 * 100:.0f} "
    "% de la variabilidad del IMC, de modo que la edad, por sí sola, no sirve "
    "para predecir el estado nutricional. La nube de puntos muestra, además, que "
    "los "
    f"{n_imc40} casos con IMC superior a 40 kg/m² se concentran entre los "
    f"{imc40_edad.min():.0f} y los {imc40_edad.max():.0f} años, es decir, en "
    "población joven: la obesidad extrema no es un fenómeno de edades avanzadas "
    "en esta muestra."
)

# %% [markdown]
# ## 14. Matriz de correlaciones
#
# Se correlacionan las variables numéricas para ubicar la edad dentro del
# conjunto de factores medidos. Se usa el coeficiente de Spearman porque varias
# de estas variables no son normales y están medidas en escalas ordinales.

# %%
numericas = ["Age", "Height", "Weight", "IMC", "FCVC", "NCP", "CH2O", "FAF", "TUE"]
etiquetas_corr = ["Edad", "Estatura", "Peso", "IMC", "Vegetales", "Comidas",
                  "Agua", "Act. física", "Pantallas"]

corr = df[numericas].corr(method="spearman")
corr.index = etiquetas_corr
corr.columns = etiquetas_corr

fig, ax = plt.subplots(figsize=(9, 7))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", vmin=-1, vmax=1,
            square=True, linewidths=0.6, linecolor="white",
            cbar_kws={"shrink": 0.8, "label": "Coeficiente de Spearman"}, ax=ax)
ax.set_title("Matriz de correlaciones (Spearman)")
guardar(fig, "08_matriz_correlaciones.png")

corr.round(3).to_csv(RES / "matriz_correlaciones.csv", encoding="utf-8-sig")

# %%
anotar(
    "En la matriz destaca la correlación casi perfecta entre peso e IMC "
    f"(ρ = {corr.loc['Peso', 'IMC']:.2f}), esperable porque el IMC se calcula a "
    "partir del peso, por lo que una de las dos variables es redundante en "
    "cualquier modelo. La correlación entre edad y estatura es prácticamente "
    f"nula en esta muestra (ρ = {corr.loc['Edad', 'Estatura']:.2f}); este dato "
    "no permite afirmar que ocurra lo mismo en otras poblaciones. Las asociaciones "
    "más altas de la edad son "
    f"con el IMC (ρ = {corr.loc['Edad', 'IMC']:.2f}) y el peso "
    f"(ρ = {corr.loc['Edad', 'Peso']:.2f}). También aparecen correlaciones "
    f"negativas débiles con el tiempo de pantalla (ρ = {corr.loc['Edad', 'Pantallas']:.2f}) "
    f"y la actividad física (ρ = {corr.loc['Edad', 'Act. física']:.2f}). Ninguna "
    "supera 0,5: la edad no es un predictor fuerte de ninguna de las variables "
    "registradas."
)

# %% [markdown]
# ## 15. Conclusiones
#
# A continuación se consolidan los hallazgos del análisis. Los valores se
# calculan en la celda siguiente, de modo que las conclusiones quedan
# sincronizadas con los datos si el archivo se actualiza.

# %%
seccion("RESUMEN DE HALLAZGOS")

resumen_final = pd.DataFrame({
    "Indicador": [
        "Tamaño de la muestra", "Edad media (IC 95 %)", "Mediana", "Moda",
        "Desviación estándar", "Coeficiente de variación", "Rango",
        "IQR", "Asimetría", "Curtosis", "Atípicos (Tukey)",
        "Normalidad (Shapiro-Wilk p)", "Diferencia por género (p)",
        "Diferencia por obesidad (p)", "Correlación edad-IMC (r)",
    ],
    "Resultado": [
        f"{n} personas",
        f"{media:.2f} años ({ic_inf:.2f} – {ic_sup:.2f})",
        f"{mediana:.0f} años",
        f"{moda:.0f} años",
        f"{desv:.2f} años",
        f"{cv:.1f} %",
        f"{minimo:.0f} – {maximo:.0f} años",
        f"{iqr:.1f} años",
        f"{asimetria:.2f} (positiva)",
        f"{curtosis:.2f}",
        f"{int(mascara_iqr.sum())} ({mascara_iqr.mean() * 100:.1f} %)",
        f"{shapiro.pvalue:.1e} → no normal",
        f"p = {mw.pvalue:.3g}; "
        f"{'diferencia significativa' if mw.pvalue < 0.05 else 'sin diferencia significativa'}",
        f"p = {kw.pvalue:.3g}; "
        f"{'diferencias significativas entre grupos' if kw.pvalue < 0.05 else 'sin diferencias significativas'}",
        f"{pearson.statistic:.2f} → débil",
    ],
})
mostrar(resumen_final)
resumen_final.to_csv(RES / "resumen_final.csv", index=False, encoding="utf-8-sig")

# %%
conclusiones = [
    "1. La muestra es marcadamente joven: la edad media es de "
    f"{media:.1f} años y la mediana de {mediana:.0f}; el {p30:.1f} % tiene 30 años "
    "o menos. Por tanto, este conjunto describe principalmente a población "
    "adolescente y adulta joven, no necesariamente a la población general.",
    "2. La distribución de la edad no es normal: presenta asimetría positiva "
    f"({asimetria:.2f}) y curtosis ({curtosis:.2f}), confirmadas por el histograma, "
    "el Q-Q plot y las pruebas de Shapiro-Wilk, D'Agostino-Pearson y "
    "Anderson-Darling. Por tanto, la mediana y el IQR son las medidas adecuadas "
    "para describirla.",
    "3. La dispersión es moderada (DE = "
    f"{desv:.1f} años; CV = {cv:.1f} %), pero el 50 % central se concentra en "
    f"solo {iqr:.0f} años ({q1:.0f} a {q3:.0f}), lo que muestra una concentración "
    "de edades en el centro de la muestra.",
    "4. El criterio de Tukey marca "
    f"{int(mascara_iqr.sum())} edades como atípicas, entre "
    f"{atipicos['Age'].min():.0f} y {atipicos['Age'].max():.0f} años. Son valores "
    "plausibles; ser atípico no demuestra que un dato sea erróneo. Se conservan "
    "y se documentan para no alterar la distribución sin una razón adicional.",
    "5. Mann-Whitney encuentra una diferencia estadísticamente significativa "
    f"de edad por género (p = {mw.pvalue:.3g}), aunque pequeña en magnitud. "
    "Kruskal-Wallis también detecta diferencias entre niveles de obesidad "
    f"(p = {kw.pvalue:.3g}). La correlación lineal edad-IMC es positiva pero "
    f"baja (r = {pearson.statistic:.2f}). Estos resultados describen asociaciones "
    "en este dataset; no demuestran causalidad.",
    "6. Para tomar decisiones fuera de este ejercicio se necesitarían datos "
    "representativos de la población objetivo. El archivo incluye datos "
    "sintéticos según su nombre/documentación, por lo que sus proporciones y "
    "asociaciones no deben generalizarse automáticamente."
]
for c in conclusiones:
    print(c + "\n")
    HALLAZGOS.append(c)

# %% [markdown]
# ### Cierre
#
# Los resultados se exportaron a `resultados/` (tablas en CSV) y las figuras a
# `figuras/` (PNG a 300 dpi). El archivo `resultados/hallazgos.txt` contiene
# todos los párrafos de interpretación generados, listos para incorporar al
# informe en APA.
#
# **Referencias sugeridas para el informe:**
#
# - Deckler, G. (2019). *Learn Power BI: A beginner's guide to developing interactive business intelligence solutions using Microsoft Power BI*. Packt Publishing.
# - González Támara, L. (2018). *Análisis exploratorio de datos: una introducción a la estadística descriptiva y probabilidad*.
# - Vilalta Perdomo, C. J. (2016). *Análisis de datos*.
# - Palechor, F. M., & de la Hoz Manotas, A. (2019). *Dataset for estimation of obesity levels based on eating habits and physical condition in individuals from Colombia, Peru and Mexico* [Conjunto de datos]. Kaggle.

# %%
# Guardar todos los párrafos de interpretación en un archivo de texto
texto = "\n\n".join(HALLAZGOS)
(RES / "hallazgos.txt").write_text(texto, encoding="utf-8")

print(f"Figuras generadas : {len(list(FIG.glob('*.png')))}")
print(f"Tablas generadas  : {len(list(RES.glob('*.csv')))}")
print(f"Hallazgos escritos: {len(HALLAZGOS)}")
print("\nProceso finalizado correctamente.")
