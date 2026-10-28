# Tarea 3 — Análisis de datos mediante software

**Curso:** Fundamentos de análisis cuantitativo  
**Actividad:** Tarea 3  
**Estudiante:** Leidy Jimena Noreña Quiceno
## Presentación

Este proyecto realiza un análisis descriptivo de la variable **edad** usando el conjunto de datos sobre niveles de obesidad, hábitos alimenticios y condición física. El análisis se desarrolló en Python y presenta tablas, gráficos e interpretaciones de los resultados.

## Contenido

- `analisis_edad.py`: código del análisis.
- `data/ObesityDataSet_raw_and_data_sinthetic.csv`: archivo de datos.
- `figuras/`: gráficos generados por el código.
- `resultados/`: tablas estadísticas y resumen de hallazgos.

El análisis incluye medidas descriptivas, frecuencias por grupos de edad, revisión de valores atípicos, distribución de la edad y comparaciones con otras variables del conjunto.

## Cómo ejecutarlo

Desde la carpeta del proyecto, instala las bibliotecas necesarias y ejecuta:

```bash
python -m pip install pandas numpy matplotlib seaborn scipy
python analisis_edad.py
```

El código busca el archivo CSV en `data/` o en la carpeta Descargas. Al finalizar, genera las gráficas en `figuras/` y las tablas e interpretaciones en `resultados/`.

## Consideraciones

El archivo incluye datos sintéticos. Por ello, los resultados describen este conjunto de datos y no deben interpretarse como prueba de causalidad ni generalizarse automáticamente a toda la población.

**Fuente de los datos:** conjunto *Dataset for estimation of obesity levels based on eating habits and physical condition in individuals from Colombia, Peru and Mexico*, disponible en Kaggle: 

https://www.kaggle.com/datasets/adeniranstephen/obesity-prediction-dataset
