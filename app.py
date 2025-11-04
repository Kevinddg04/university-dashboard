# ==============================================================
# 🎓 UNIVERSITY DATA DASHBOARD - STREAMLIT APP
# ==============================================================

import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Configuración general del dashboard
st.set_page_config(
    page_title="University Data Dashboard",
    layout="wide",
    page_icon="🎓"
)

# ==============================================================
# 🧱 Encabezado
# ==============================================================

st.title("🎓 University Student Data Dashboard")
st.markdown("### Data Visualization and Dashboard Deployment – Activity 1")
st.markdown("""
**📚 Course:** Mineria de Datos  
**👨‍🏫 Professor:** José Escorcia-Gutiérrez, Ph.D.  
**👥 Team Members:**  
- Kevin David Gallardo  
- Mauricio Carrillo
""")

st.markdown("---")

# ==============================================================
# 📂 1️⃣ Cargar datos
# ==============================================================

@st.cache_data
def load_data():
    df = pd.read_csv("university_student_data.csv")
    return df

df = load_data()

st.sidebar.header("🔎 Data Filters")

# ==============================================================
# 🎛️ 2️⃣ Filtros interactivos
# ==============================================================

years = sorted(df["Year"].unique())
terms = sorted(df["Term"].unique())

selected_year = st.sidebar.selectbox("Select Year", options=["All"] + [str(y) for y in years])
selected_term = st.sidebar.selectbox("Select Term", options=["All"] + list(terms))

filtered_df = df.copy()

if selected_year != "All":
    filtered_df = filtered_df[filtered_df["Year"] == int(selected_year)]
if selected_term != "All":
    filtered_df = filtered_df[filtered_df["Term"] == selected_term]

# ==============================================================
# 🔢 3️⃣ KPIs principales
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
# 📊 4️⃣ Visualizaciones
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
# 🧠 5️⃣ Conclusiones e insights
# ==============================================================

st.markdown("---")
st.markdown("""
### 🧠 Key Insights
- The university’s **retention rate** has shown a gradual improvement over time.
- **Student satisfaction** levels have increased steadily, suggesting positive institutional changes.
- Enrollment levels between **Spring and Fall** are relatively balanced.
- The **Engineering department** consistently leads in total enrollment.

### 💡 Suggested Actions
- Keep strengthening academic support programs to maintain retention growth.
- Conduct satisfaction surveys focused on departments with lower retention.
- Use this dashboard as a tool for yearly academic planning and performance tracking.

---
Developed by **Kevin David Gallardo** and **Mauricio Carrillo** using **Streamlit**, **Pandas**, and **Seaborn**.
""")
