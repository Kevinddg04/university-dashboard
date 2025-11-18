# ============================================================================
#  PROYECTO FINAL – CLASIFICACIÓN DE ESPECIES IRIS
# ============================================================================
# Integrantes:
# - Kevin David Gallardo
# - Mauricio Carrillo
#
# Curso: Minería de Datos
# Profesor: José Escorcia-Gutiérrez, PhD.
#
# En este proyecto desarrollamos un flujo completo de Machine Learning:
# Exploración de datos, entrenamiento, evaluación y predicción.
# Todo integrado en un dashboard interactivo con Streamlit.
# ============================================================================

import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import pickle
from io import BytesIO

# ============================================================================
# ⚙️ CONFIGURACIÓN INICIAL DE LA PÁGINA
# ============================================================================

st.set_page_config(
    page_title="Iris Classification",
    layout="wide",
    page_icon="🌸"
)

st.title("🌸 Iris Species Classification Dashboard")
st.write("""
Proyecto final del curso **Minería de Datos**, donde aplicamos técnicas de 
Machine Learning para clasificar flores Iris según sus características.
""")

st.info("Integrantes: **Kevin David Gallardo** – **Mauricio Carrillo**")

st.write("---")

# ============================================================================
# 📥 CARGA DE DATOS
# ============================================================================

@st.cache_data
def load_data():
    # Usamos el dataset de Seaborn para simplificar
    return sns.load_dataset("iris")

df = load_data()

# Opciones de navegación del dashboard
st.sidebar.title("📌 Navegación")
section = st.sidebar.radio(
    "Selecciona una sección:",
    ["Vista del Dataset", "Análisis Exploratorio", "Entrenamiento del Modelo",
     "Evaluación del Modelo", "Predicción"]
)

# ============================================================================
# 🧾 SECCIÓN 1: VISTA DEL DATASET
# ============================================================================

if section == "Vista del Dataset":
    st.header("📊 Vista general del dataset")

    st.write("El dataset contiene 150 muestras de flores Iris con 4 características numéricas.")
    st.dataframe(df)

    st.subheader("🔎 Estadísticas básicas")
    st.write(df.describe())

    st.subheader("📌 Distribución de especies")
    st.bar_chart(df["species"].value_counts())

# ============================================================================
# 🔬 SECCIÓN 2: ANÁLISIS EXPLORATORIO
# ============================================================================

elif section == "Análisis Exploratorio":
    st.header("🔬 Exploración de datos")

    st.subheader("📈 Scatter Matrix")
    fig = sns.pairplot(df, hue="species")
    st.pyplot(fig)

    st.subheader("📊 Histograma general")
    fig2, ax2 = plt.subplots(figsize=(7,4))
    df.hist(ax=ax2)
    st.pyplot(fig2)

    st.subheader("🌐 Gráfico 3D interactivo")
    fig3 = px.scatter_3d(
        df,
        x="sepal_length",
        y="sepal_width",
        z="petal_length",
        color="species",
        title="Gráfico 3D de las flores Iris"
    )
    st.plotly_chart(fig3)

# ============================================================================
# 🤖 SECCIÓN 3: ENTRENAMIENTO DEL MODELO
# ============================================================================

elif section == "Entrenamiento del Modelo":
    st.header("🤖 Entrenamiento del modelo")

    X = df.drop("species", axis=1)
    y = df["species"]

    # Dividir los datos
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # Escalado
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # Crear el modelo
    model = RandomForestClassifier(n_estimators=120, random_state=42)
    model.fit(X_train_scaled, y_train)

    st.success("Modelo entrenado correctamente 🎉")

    # Descargar modelo
    buffer = BytesIO()
    pickle.dump(model, buffer)
    st.download_button("💾 Descargar modelo entrenado", data=buffer.getvalue(), file_name="iris_model.pkl")

# ============================================================================
# 📈 SECCIÓN 4: EVALUACIÓN DEL MODELO
# ============================================================================

elif section == "Evaluación del Modelo":
    st.header("📈 Evaluación del modelo")

    X = df.drop("species", axis=1)
    y = df["species"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = RandomForestClassifier(n_estimators=120, random_state=42)
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)

    # Métricas
    st.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.3f}")
    st.metric("Precision", f"{precision_score(y_test, y_pred, average='weighted'):.3f}")
    st.metric("Recall", f"{recall_score(y_test, y_pred, average='weighted'):.3f}")
    st.metric("F1-Score", f"{f1_score(y_test, y_pred, average='weighted'):.3f}")

    st.subheader("📊 Matriz de Confusión")
    cm = confusion_matrix(y_test, y_pred)
    st.write(cm)

# ============================================================================
# 🌼 SECCIÓN 5: PREDICCIÓN
# ============================================================================

elif section == "Predicción":
    st.header("🌼 Predicción de especie")

    st.write("Ajusta los valores usando los sliders:")

    sepal_length = st.slider("Sepal Length", 4.0, 8.0, 5.0)
    sepal_width  = st.slider("Sepal Width",  2.0, 4.5, 3.0)
    petal_length = st.slider("Petal Length", 1.0, 7.0, 4.0)
    petal_width  = st.slider("Petal Width",  0.1, 2.5, 1.3)

    # Entrenar el modelo para predicción
    X = df.drop("species", axis=1)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = RandomForestClassifier(n_estimators=120, random_state=42)
    model.fit(X_scaled, df["species"])

    new_data = [[sepal_length, sepal_width, petal_length, petal_width]]
    scaled_new = scaler.transform(new_data)

    prediction = model.predict(scaled_new)[0]

    st.success(f"🌸 La especie predicha es: **{prediction}**")

    st.subheader("📌 Ubicación de la muestra en 3D")
    fig = px.scatter_3d(
        df,
        x="sepal_length",
        y="sepal_width",
        z="petal_length",
        color="species"
    )
    fig.add_scatter3d(
        x=[sepal_length], y=[sepal_width], z=[petal_length],
        mode="markers", marker=dict(size=8, color="black"),
        name="Nueva muestra"
    )
    st.plotly_chart(fig)

