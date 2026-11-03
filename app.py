from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# -------------------- Page setup --------------------
st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #071426 0%, #0b1f38 55%, #102d4b 100%);
    color: #eef6ff;
}
[data-testid="stSidebar"] {
    background: #091a2d;
    border-right: 1px solid rgba(110, 190, 255, .18);
}
h1, h2, h3, p, label, .stMarkdown { color: #eef6ff; }
.hero {
    padding: 1.35rem 1.6rem;
    border: 1px solid rgba(91, 192, 255, .25);
    border-radius: 18px;
    background: linear-gradient(110deg, rgba(17, 57, 91, .92), rgba(12, 36, 62, .92));
    margin-bottom: 1.2rem;
}
.hero p { color: #bdd7ef; margin-bottom: 0; }
.metric-card {
    background: linear-gradient(145deg, #123453, #0d2742);
    border: 1px solid rgba(92, 190, 255, .22);
    border-radius: 15px;
    padding: 16px 18px;
    min-height: 105px;
}
.metric-label { color: #a9c9e5; font-size: .9rem; }
.metric-value { color: #ffffff; font-size: 1.75rem; font-weight: 700; }
div.stButton > button {
    background: linear-gradient(90deg, #168bd2, #21b6c7);
    color: white; border: 0; border-radius: 10px;
    font-weight: 700; padding: .65rem 1rem;
}
div.stButton > button:hover { color: white; border: 1px solid #8bdcff; }
div[data-testid="stDataFrame"] { border: 1px solid rgba(110,190,255,.18); border-radius: 12px; }
.small-note { color: #a9c5df; font-size: .85rem; }
</style>
""", unsafe_allow_html=True)

DATA_PATH = Path(__file__).parent / "Student_Performance.csv"

@st.cache_data
def load_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            "Student_Performance.csv was not found. Keep it in the same folder as app.py."
        )
    data = pd.read_csv(DATA_PATH)
    required = {"G1", "G2", "G3"}
    if not required.issubset(data.columns):
        raise ValueError("Dataset must contain G1, G2, and G3 columns.")
    return data.drop_duplicates().copy()

@st.cache_resource
def train_model():
    data = load_data()
    X = data[["G1", "G2"]]
    y = data["G3"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = LinearRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    metrics = {
        "r2": r2_score(y_test, predictions),
        "mae": mean_absolute_error(y_test, predictions),
        "mse": mean_squared_error(y_test, predictions),
    }
    results = pd.DataFrame({
        "Actual Grade (G3)": y_test.values,
        "Predicted Grade": predictions.round(2),
    })
    return model, metrics, results, X_test, y_test, predictions

st.markdown("""
<div class="hero">
  <h1>🎓 Student Performance Prediction</h1>
  <p>Predict a student's final grade (G3) using the first-period grade (G1)
  and second-period grade (G2), with a Linear Regression model.</p>
</div>
""", unsafe_allow_html=True)

try:
    data = load_data()
    model, metrics, results, X_test, y_test, predictions = train_model()
except Exception as exc:
    st.error(f"Could not load or prepare the project: {exc}")
    st.stop()

# Sidebar inputs
st.sidebar.header("🔎 Prediction Input")
st.sidebar.write("Enter the student's previous grades.")
g1 = st.sidebar.slider("First-period grade (G1)", 0, 20, 12)
g2 = st.sidebar.slider("Second-period grade (G2)", 0, 20, 14)
predict_clicked = st.sidebar.button("Predict Final Grade", use_container_width=True)
st.sidebar.markdown("---")
st.sidebar.caption("Model features: G1 and G2")
st.sidebar.caption("Target: G3 (final grade)")

# Dataset overview
c1, c2, c3, c4 = st.columns(4)
overview = [
    ("Students in dataset", f"{len(data):,}"),
    ("Average final grade", f"{data['G3'].mean():.2f} / 20"),
    ("Average G1", f"{data['G1'].mean():.2f}"),
    ("Average G2", f"{data['G2'].mean():.2f}"),
]
for col, (label, value) in zip([c1, c2, c3, c4], overview):
    col.markdown(
        f'<div class="metric-card"><div class="metric-label">{label}</div>'
        f'<div class="metric-value">{value}</div></div>',
        unsafe_allow_html=True,
    )

st.markdown("## 📌 Model Performance")
m1, m2, m3 = st.columns(3)
m1.metric("R² Score", f"{metrics['r2']:.3f}", help="Closer to 1 generally indicates a better fit.")
m2.metric("Mean Absolute Error", f"{metrics['mae']:.2f}")
m3.metric("Mean Squared Error", f"{metrics['mse']:.2f}")

st.markdown("## 🧮 Predict a Student's Final Grade")
if predict_clicked:
    prediction = float(model.predict(pd.DataFrame({"G1": [g1], "G2": [g2]}))[0])
    prediction = max(0.0, min(20.0, prediction))
    st.success(f"Estimated final grade (G3): **{prediction:.2f} / 20**")
    st.caption("This is an estimate from the trained model, not a guaranteed exam result.")
else:
    st.info("Choose G1 and G2 in the sidebar, then click **Predict Final Grade**.")

st.markdown("## 📊 Actual vs Predicted Grades")
chart_data = pd.DataFrame({
    "Actual Grade": y_test.values,
    "Predicted Grade": predictions,
})
fig = px.scatter(
    chart_data, x="Actual Grade", y="Predicted Grade",
    title="Actual vs Predicted Final Grades",
    opacity=0.75,
    color_discrete_sequence=["#35c4e8"],
)
fig.add_shape(
    type="line", x0=0, y0=0, x1=20, y1=20,
    line=dict(color="#f7b955", dash="dash", width=2),
)
fig.update_layout(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(9,26,45,.55)",
    font_color="#eaf5ff",
    title_font_color="#ffffff",
)
st.plotly_chart(fig, use_container_width=True)

left, right = st.columns(2)
with left:
    grade_counts = data["G3"].value_counts().sort_index().rename_axis("Final Grade").reset_index(name="Students")
    bar = px.bar(
        grade_counts, x="Final Grade", y="Students",
        title="Distribution of Final Grades (G3)",
        color="Students", color_continuous_scale="Teal",
    )
    bar.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(9,26,45,.55)")
    st.plotly_chart(bar, use_container_width=True)
with right:
    trend = data.groupby("G2", as_index=False)["G3"].mean()
    line = px.line(
        trend, x="G2", y="G3", markers=True,
        title="Average Final Grade by G2",
        labels={"G2": "Second-period grade (G2)", "G3": "Average final grade (G3)"},
    )
    line.update_traces(line_color="#35c4e8", marker_color="#f7b955")
    line.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(9,26,45,.55)")
    st.plotly_chart(line, use_container_width=True)

with st.expander("View sample prediction results"):
    st.dataframe(results.head(15), use_container_width=True)

with st.expander("View dataset"):
    st.dataframe(data, use_container_width=True)

csv_bytes = results.to_csv(index=False).encode("utf-8")
st.download_button(
    "⬇️ Download test prediction results",
    data=csv_bytes,
    file_name="student_performance_predictions.csv",
    mime="text/csv",
)

st.markdown("---")
st.markdown(
    '<p class="small-note">Educational project. This model follows the notebook approach '
    'and uses only G1 and G2 to predict G3. Predictions should not be used as the sole '
    'basis for academic decisions.</p>',
    unsafe_allow_html=True,
)
