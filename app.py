# ==============================================================
# 🎓 UNIVERSITY DATA DASHBOARD - STREAMLIT APP
# ==============================================================

import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Configuración general
st.set_page_config(
    page_title="University Data Dashboard",
    layout="wide",
    page_icon="🎓"
)

# ==============================================================
# 1️⃣ Cargar datos
# ==============================================================

@st.cache_data
def load_data():
    df = pd.read_csv("university_student_data.csv")
    return df

df = load_data()

st.title("🎓 University Student Data Dashboard")
st.markdown("### Data Mining - Visualization & Dashboard Deployment")

# ==============================================================
# 2️⃣ Filtros interactivos
# ==============================================================

col1, col2, col3 = st.columns(3)
years = sorted(df["Year"].unique())
terms = df["Term"].unique()

with col1:
    selected_year = st.selectbox("Select Year", options=["All"] + list(map(str, years)))
with col2:
    selected_term = st.selectbox("Select Term", options=["All"] + list(terms))
with col3:
    st.markdown("")

# Aplicar filtros
filtered_df = df.copy()

if selected_year != "All":
    filtered_df = filtered_df[filtered_df["Year"] == int(selected_year)]
if selected_term != "All":
    filtered_df = filtered_df[filtered_df["Term"] == selected_term]

# ==============================================================
# 3️⃣ KPIs principales
# ==============================================================

avg_retention = filtered_df["Retention Rate (%)"].mean()
avg_satisfaction = filtered_df["Student Satisfaction (%)"].mean()
total_enrollment = filtered_df["Enrolled"].sum()

col1, col2, col3 = st.columns(3)
col1.metric("📈 Average Retention Rate", f"{avg_retention:.1f}%")
col2.metric("😊 Student Satisfaction", f"{avg_satisfaction:.1f}%")
col3.metric("👩‍🎓 Total Enrolled Students", f"{total_enrollment}")

st.markdown("---")

# ==============================================================
# 4️⃣ Visualizaciones
# ==============================================================

sns.set_style("whitegrid")

# --- Retention Trend ---
st.subheader("Retention Rate Trends Over Time")
fig1, ax1 = plt.subplots(figsize=(7,4))
sns.lineplot(data=filtered_df, x="Year", y="Retention Rate (%)", marker="o", ax=ax1)
ax1.set_title("Retention Rate Over Time")
st.pyplot(fig1)

# --- Satisfaction by Year ---
st.subheader("Student Satisfaction by Year")
fig2, ax2 = plt.subplots(figsize=(7,4))
sns.barplot(data=filtered_df, x="Year", y="Student Satisfaction (%)", ax=ax2, ci=None)
ax2.set_title("Student Satisfaction by Year")
st.pyplot(fig2)

# --- Enrollment Comparison ---
st.subheader("Enrollment Comparison (Spring vs Fall)")
fig3, ax3 = plt.subplots(figsize=(6,4))
sns.barplot(data=filtered_df, x="Term", y="Enrolled", ci=None, ax=ax3)
ax3.set_title("Enrollment by Term")
st.pyplot(fig3)

# --- Department Enrollment ---
st.subheader("Department Enrollment Distribution")
dept_cols = ["Engineering Enrolled", "Business Enrolled", "Arts Enrolled", "Science Enrolled"]
dept_totals = filtered_df[dept_cols].sum().sort_values(ascending=False)

fig4, ax4 = plt.subplots(figsize=(7,4))
sns.barplot(x=dept_totals.index, y=dept_totals.values, ax=ax4)
ax4.set_title("Departmental Enrollment")
ax4.set_ylabel("Students")
ax4.set_xlabel("Department")
st.pyplot(fig4)

# ==============================================================
# 5️⃣ Conclusión
# ==============================================================

st.markdown("---")
st.markdown("""
**🧠 Insights principales:**
- La tasa de retención ha mostrado una tendencia positiva a lo largo de los años.
- La satisfacción estudiantil también ha mejorado consistentemente.
- Las matrículas se distribuyen de manera equilibrada entre *Spring* y *Fall*.
- Ingeniería sigue siendo el departamento con mayor número de estudiantes matriculados.

---
Desarrollado con ❤️ usando **Streamlit**, **Pandas** y **Seaborn**.
""")
