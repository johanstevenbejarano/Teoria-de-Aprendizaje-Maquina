import streamlit as st
import pandas as pd
from pathlib import Path
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go

# ==== CONFIGURACIÓN GENERAL ====
st.set_page_config(page_title="Comparativa de Modelos TAM 2025-1", layout="wide")

# ==== ESTILO PERSONALIZADO (CSS) ====
st.markdown("""
    <style>
    html, body, [class*="css"]  {
        font-family: 'Segoe UI', sans-serif;
        font-size: 16px;
        color: #1f2937;
    }
    h1 {
        font-size: 2.5rem;
        font-weight: bold;
        color: #0d6efd;
    }
    h2, h3 {
        color: #1a1a1a;
    }
    section[data-testid="stSidebar"] {
        background-color: #20232a;
        color: #ffffff;
    }
    .stRadio label {
        font-size: 16px;
        color: #ffffff;
    }
    .dataframe thead tr th {
        background-color: #f1f1f1;
        color: #333;
    }
    .dataframe tbody tr td {
        background-color: #fafafa;
        color: #111;
    }
    </style>
""", unsafe_allow_html=True)

# ==== RUTAS BASE ====
root_path = Path(__file__).resolve().parent.parent
data_path = root_path / "data"
image_path = Path("C:/Users/PC/Downloads/imagen.jpg")

# ==== ENCABEZADO ====
st.title("Dashboard de Evaluación de Modelos - TAM 2025-1")
st.markdown("""
Bienvenido al panel interactivo de evaluación y comparación de modelos de regresión aplicados al conjunto de datos **Ames Housing**.  
Aquí podrás visualizar métricas clave, distribuciones de errores, gráficos de predicción y mucho más para cada modelo entrenado.

Este tablero busca facilitar la **selección informada** de los **3 mejores modelos**, con base en criterios estadísticos y visuales.
""")

# ==== SIDEBAR ====
st.sidebar.title("🔎 Navegación")
opcion = st.sidebar.radio(
    "Selecciona una sección",
    ["Introducción", 
     "Comparativa de métricas", 
     "Predicciones vs. Reales",
     "Curvas de aprendizaje",
     "Conclusión"]
)

# ==== INTRODUCCIÓN ====
if opcion == "Introducción":
    st.subheader("📘 Bienvenida al Dashboard Interactivo")

    st.image(image_path, use_container_width=True)

    st.markdown("""
    Este tablero interactivo te permite explorar, comparar e interpretar el rendimiento de distintos modelos de **regresión** aplicados al conjunto de datos **Ames Housing** — un registro detallado de propiedades residenciales que incluye más de 70 características por casa.

    ### 🎯 ¿Qué busca responder este dashboard?
    - ¿Qué modelo predice mejor el **precio de una vivienda**?
    - ¿Cuál es más **estable** (menos variable)?
    - ¿Cómo se comportan frente a **nuevos datos**?

    ### 📏 ¿Qué métricas se utilizan?
    Cada modelo se evaluó con las siguientes medidas:

    | Métrica | Qué mide | Interpretación |
    |--------|----------|----------------|
    | `MAE` (Error Absoluto Medio) | Promedio del error absoluto | Más bajo es mejor |
    | `RMSE` (Raíz del Error Cuadrático Medio) | Penaliza errores grandes | Más bajo es mejor |
    | `R²` (Coeficiente de determinación) | Qué tanto explica el modelo la variabilidad | Se espera cercano a 1 |
    | `MAPE` (%) | Error medio porcentual | Indica error relativo al valor real |

    Las unidades de `MAE` y `RMSE` están expresadas en **dólares estadounidenses (USD)**.  
    El `MAPE` está expresado como **porcentaje de error promedio**.

    ---

    ### 👀 ¿Qué puedes hacer aquí?
    - Visualizar gráficas de desempeño por modelo.
    - Comparar resultados estadísticos.
    - Ver predicciones vs. valores reales.
    - Evaluar consistencia mediante curvas de aprendizaje.
    
    ---
    **Este tablero está diseñado para ayudarte a tomar decisiones informadas** sobre qué modelo aplicar, ajustar o desplegar.  
    No necesitas ser experto en Machine Learning para interpretarlo 😊.
    """)




# ==== COMPARATIVA DE MÉTRICAS ====
elif opcion == "Comparativa de métricas":
    st.subheader("📊 Comparación de métricas por modelo")

    resumen_path = data_path / "resumen_mejores_modelos.csv"

    # Leer correctamente la segunda fila como encabezado
    resumen_df = pd.read_csv(resumen_path, header=[1], index_col=0)

    # Renombrar columnas para que tengan formato métrica_estadística
    resumen_df.columns = [
        'MAE_mean', 'MAE_std',
        'RMSE_mean', 'RMSE_std',
        'R2_mean', 'R2_std',
        'MAPE_mean', 'MAPE_std'
    ]

    # Convertir a formato largo
    resumen_long = resumen_df.stack().reset_index()
    resumen_long.columns = ['Modelo', 'Metrica_Completa', 'Valor']
    resumen_long[['Métrica', 'Estadística']] = resumen_long['Metrica_Completa'].str.split('_', expand=True)

    # Pivot a formato ancho con columnas mean y std
    resumen_pivot = resumen_long.pivot(index=['Modelo', 'Métrica'], columns='Estadística', values='Valor').reset_index()

    # Gráficos para cada métrica
    for metrica in ['MAE', 'RMSE', 'MAPE', 'R2']:
        metrica_df = resumen_pivot[resumen_pivot['Métrica'] == metrica]
        fig = px.bar(
            metrica_df,
            x='Modelo',
            y='mean',
            error_y='std',
            color='Modelo',
            text='mean',
            title=f'{metrica} (media ± std)',
            color_discrete_sequence=px.colors.qualitative.Vivid
        )
        fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
        fig.update_layout(
            yaxis_title=metrica,
            xaxis_title="Modelo",
            showlegend=False,
            bargap=0.3,
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)

    # Tabla estilizada
    st.subheader("📋 Tabla resumen de métricas")
    st.dataframe(
        resumen_df.style.format("{:.3f}").highlight_max(axis=0, color="lightgreen"),
        use_container_width=True
    )

    # === Comentario resumen final ===
    st.markdown("---")
    st.markdown("### 📌 Observaciones clave sobre las métricas:")

    st.markdown("""
    - Los modelos evaluados muestran diferencias relevantes en las métricas de desempeño.
     **BayesianRidge** y **GaussianProcess** destacan por su **bajo MAE y MAPE**, lo que indica alta precisión en las predicciones.
    - **KernelRidge** mantiene un buen equilibrio entre error medio y estabilidad (baja desviación estándar).
    - Las barras de error revelan que algunos modelos tienen **rendimientos más consistentes** que otros.
    - La métrica **R²** refuerza el buen ajuste global de los modelos principales.

    > ✅ Estos resultados sugieren que **BayesianRidge** y **GaussianProcess** son opciones sólidas por su rendimiento promedio y estabilidad.
    """)



# ==== PREDICCIONES VS. REALES ====
elif opcion == "Predicciones vs. Reales":
    st.subheader("📈 Comparación: Predicciones vs. Valores reales")
    st.markdown("""
    Observa a continuación cómo se comparan las predicciones frente a los valores reales para los **3 mejores modelos**.
    Cada gráfico incluye una línea de referencia perfecta (_Y = X_) para evaluar visualmente la calidad de las predicciones.
    """)

    # Diccionario con los archivos de predicción
    pred_files = {
        "GaussianProcess": "predicciones_test_gaussian.csv",
        "BayesianRidge": "predicciones_test_bayesianridge.csv",
        "KernelRidge": "predicciones_test_kernelridge.csv"
    }

    # Columnas esperadas en los CSV
    expected_cols = {'Actual', 'Predicho'}

    for model, file_name in pred_files.items():
        file_path = data_path / file_name

        st.markdown(f"---\n### 🔹 {model}")

        if not file_path.exists():
            st.error(f"❌ Archivo no encontrado: `{file_name}`")
            continue

        try:
            df_pred = pd.read_csv(file_path)

            if not expected_cols.issubset(df_pred.columns):
                st.warning("⚠️ El archivo no contiene las columnas esperadas: `Actual` y `Predicho`")
                st.dataframe(df_pred.head())  # Mostrar preview si tiene otras columnas
                continue

            # Calcular errores para resumen (opcional)
            df_pred['Error'] = df_pred['Actual'] - df_pred['Predicho']

            # Gráfico Real vs Predicho
            fig = go.Figure()

            fig.add_trace(go.Scatter(
                x=df_pred['Actual'],
                y=df_pred['Predicho'],
                mode='markers',
                marker=dict(color='dodgerblue', opacity=0.6, size=6),
                name='Predicciones'
            ))

            fig.add_trace(go.Scatter(
                x=df_pred['Actual'],
                y=df_pred['Actual'],
                mode='lines',
                line=dict(color='crimson', dash='dash'),
                name='Línea ideal'
            ))

            fig.update_layout(
                title=f"{model}: Valor Real vs. Predicción",
                xaxis_title="Valor real",
                yaxis_title="Predicción",
                height=450,
                margin=dict(l=40, r=40, t=60, b=40)
            )

            st.plotly_chart(fig, use_container_width=True)

            # Estadísticas básicas del error
            st.caption("📌 Estadísticas de error:")
            col1, col2, col3 = st.columns(3)
            col1.metric("MAE", f"{abs(df_pred['Error']).mean():,.2f}")
            col2.metric("RMSE", f"{(df_pred['Error']**2).mean()**0.5:,.2f}")
            col3.metric("R²", f"{df_pred[['Actual', 'Predicho']].corr().iloc[0,1]**2:.3f}")

        except Exception as e:
            st.error(f"❌ Error procesando el archivo `{file_name}`: {str(e)}")
    # === Comentario resumen final ===
    st.markdown("---")
    st.markdown("### 🧠 Observaciones finales:")

    st.markdown("""
    - Un modelo ideal tendría sus puntos alineados perfectamente sobre la línea roja (`Y = X`).
    - **GaussianProcess** y **BayesianRidge** muestran buena concentración alrededor de la línea ideal.
    - **KernelRidge** también presenta un desempeño competitivo, aunque con ligeros errores para valores extremos.
    - Las métricas presentadas (MAE, RMSE, R²) refuerzan la calidad general de estos modelos.

    > 💡 Esta visualización es clave para evaluar **precisión individual por instancia**, no solo promedios agregados.
    """)







# ==== CURVAS DE APRENDIZAJE ====
elif opcion == "Curvas de aprendizaje":
    st.subheader("📈 Curvas de Aprendizaje por Modelo")

    st.markdown("""
    Las **curvas de aprendizaje** muestran cómo evoluciona el desempeño del modelo a medida que aumentan las muestras de entrenamiento.  
    Ayudan a identificar posibles problemas de *underfitting* (modelo muy simple) o *overfitting* (modelo demasiado ajustado).

    **Interpretación esperada:**
    - 🟢 **Underfitting:** Las curvas se mantienen bajas → no aprende suficiente.
    - 🟠 **Overfitting:** La curva de entrenamiento es alta, pero la de validación no.
    - ✅ **Buen aprendizaje:** Ambas curvas convergen con valores altos.

    **Métrica utilizada:** Error cuadrático medio (MSE) en escala log-transformada.
    """)

    curve_images = {
        "GaussianProcess": "curva_aprendizaje_gpr.png",
        "BayesianRidge": "curva_aprendizaje_bayesianridge.png",
        "KernelRidge": "curva_aprendizaje_kernelridge.png"
    }

    for model, file_name in curve_images.items():
        image_path = data_path / file_name

        if image_path.exists():
            st.markdown(f"### 🔹 {model}")
            st.image(image_path, use_container_width=True, caption=f"Curva de aprendizaje – {model}")
            st.caption(f"📊 Observa el equilibrio entre aprendizaje y generalización del modelo **{model}** a medida que aumenta el tamaño del set de entrenamiento.")
        else:
            st.warning(f"⚠️ No se encontró la imagen: `{file_name}` para el modelo {model}")




# ==== CONCLUSIÓN ====
elif opcion == "Conclusión":
    st.subheader("Conclusión final")

    st.markdown("""
    ### 🧠 Hallazgos clave del análisis:

    - Se evaluaron diversos modelos de regresión, incluyendo **Bayesian Ridge**, **Gaussian Process Regressor (GPR)** y **Kernel Ridge**.
    - A través de **curvas de aprendizaje**, se observó cómo cada modelo generaliza con distintas cantidades de datos de entrenamiento:
        - El modelo **Bayesian Ridge** mostró un comportamiento estable y generalización progresiva, con errores de validación decrecientes al aumentar los datos.
        - El **GPR**, aunque más costoso computacionalmente, logró errores de validación bajos, especialmente útil en datasets de tamaño moderado.
        - El análisis de sensibilidad con **Kernel Ridge** reveló que la elección de hiperparámetros como `alpha` y `gamma` tiene un impacto considerable sobre el rendimiento, visualizado mediante mapas de calor.

    ### ✅ Recomendaciones generales:

    - Si se prioriza interpretabilidad y eficiencia, **Bayesian Ridge** es una excelente elección.
    - Para tareas donde se requiera mayor flexibilidad y se cuente con capacidad de cómputo, **GPR** puede ofrecer mejor precisión.
    - Se recomienda realizar búsqueda bayesiana o mapas de calor como apoyo visual para elegir hiperparámetros en modelos sensibles como **Kernel Ridge**.

    ### 📌 Sugerencia final:

    - Considerar el **trade-off entre complejidad, precisión y coste computacional** al seleccionar el modelo ideal.
    - Para futuras mejoras, se podría aplicar **selección de características**, **ingeniería de atributos**, y comparar con modelos no lineales como **XGBoost** o **Redes Neuronales**.

    """)

    st.success("La combinación de análisis visual, ajuste de hiperparámetros y evaluación sistemática permitió identificar modelos robustos y adecuados para este conjunto de datos.")


