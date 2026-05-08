import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Telecom Network Anomaly Detection",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded"
)




# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data" / "processed"
FIGURES_DIR = BASE_DIR / "figures"

DATA_PATH = DATA_DIR / "telecom_network_app_sample.csv"

POSSIBLE_MODEL_RESULTS = [
    DATA_DIR / "model_comparison_results.csv",
    DATA_DIR / "model_results.csv",
    BASE_DIR / "model_results.csv",
]

POSSIBLE_FINAL_SUMMARY = [
    DATA_DIR / "final_model_summary.csv",
    DATA_DIR / "model_summary.csv",
    BASE_DIR / "final_model_summary.csv",
]

POSSIBLE_FEATURE_IMPORTANCE = [
    DATA_DIR / "feature_importance_final.csv",
    DATA_DIR / "feature_importance_results.csv",
    BASE_DIR / "feature_importance_final.csv",
]

POSSIBLE_CV_RESULTS = [
    DATA_DIR / "cross_validation_results.csv",
    BASE_DIR / "cross_validation_results.csv",
]

POSSIBLE_THRESHOLD_RESULTS = [
    DATA_DIR / "threshold_optimisation_results.csv",
    BASE_DIR / "threshold_optimisation_results.csv",
]

# ============================================================
# OPERATOR THEMES
# ============================================================

THEMES = {
    "Djezzy": {
        "primary": "#E30613",
        "secondary": "#FFFFFF",
        "accent": "#FFD200",
        "dark": "#171717",
        "soft": "#FFF4F4",
        "gradient": "linear-gradient(135deg, #E30613 0%, #B8000D 60%, #FFD200 100%)",
        "logo_text": "DJEZZY",
        "subtitle": "Red, white and yellow telecom operations theme"
    },
    "Mobilis": {
        "primary": "#36A936",
        "secondary": "#FFFFFF",
        "accent": "#E30613",
        "dark": "#0F3D1E",
        "soft": "#F0FFF4",
        "gradient": "linear-gradient(135deg, #36A936 0%, #15803D 65%, #E30613 100%)",
        "logo_text": "mobilis",
        "subtitle": "Green, white and red network intelligence theme"
    },
    "Ooredoo": {
        "primary": "#ED1C24",
        "secondary": "#FFFFFF",
        "accent": "#F3F4F6",
        "dark": "#7F1D1D",
        "soft": "#FFF1F2",
        "gradient": "linear-gradient(135deg, #ED1C24 0%, #B91C1C 70%, #F9FAFB 100%)",
        "logo_text": "ooredoo",
        "subtitle": "Red, white and clean telecom analytics theme"
    },
}


# ============================================================
# DATA LOADING
# ============================================================

def read_first_existing(paths):
    for path in paths:
        if path.exists():
            return pd.read_csv(path)
    return pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_dataset():
    if DATA_PATH.exists():
        return pd.read_csv(DATA_PATH)
    return pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_model_results():
    return read_first_existing(POSSIBLE_MODEL_RESULTS)


@st.cache_data(show_spinner=False)
def load_final_summary():
    return read_first_existing(POSSIBLE_FINAL_SUMMARY)


@st.cache_data(show_spinner=False)
def load_feature_importance():
    return read_first_existing(POSSIBLE_FEATURE_IMPORTANCE)


data = load_dataset()
model_results = load_model_results()
final_summary = load_final_summary()
feature_importance = load_feature_importance()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.markdown("## 📡 Telecom Operations Control")

operator = st.sidebar.selectbox(
    "Choose operator theme",
    ["Djezzy", "Mobilis", "Ooredoo"],
    index=0
)

theme = THEMES[operator]

st.sidebar.markdown(
    f"""
    <div style="
        padding: 18px;
        border-radius: 18px;
        background: {theme['gradient']};
        color: white;
        margin-bottom: 18px;
        box-shadow: 0 14px 35px rgba(0,0,0,0.16);
    ">
        <div style="font-size: 28px; font-weight: 900;">{theme['logo_text']}</div>
        <div style="font-size: 13px; opacity: 0.95;">{theme['subtitle']}</div>
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# CSS
# ============================================================

st.markdown(
    f"""
    <style>
    .stApp {{
        background: #F8FAFC;
    }}

    section[data-testid="stSidebar"] {{
        background: linear-gradient(180deg, {theme['soft']} 0%, #FFFFFF 100%);
        border-right: 1px solid #E5E7EB;
    }}

    .main-header {{
        background: {theme['gradient']};
        padding: 42px 46px;
        border-radius: 30px;
        color: white;
        margin-bottom: 28px;
        box-shadow: 0 22px 55px rgba(15, 23, 42, 0.20);
        position: relative;
        overflow: hidden;
    }}

    .main-header::after {{
        content: "📡";
        position: absolute;
        right: 38px;
        top: 22px;
        font-size: 90px;
        opacity: 0.18;
    }}

    .header-badge {{
        display: inline-block;
        padding: 8px 16px;
        border-radius: 999px;
        background: rgba(255,255,255,0.18);
        border: 1px solid rgba(255,255,255,0.3);
        font-weight: 700;
        margin-bottom: 20px;
    }}

    .main-title {{
        font-size: 44px;
        font-weight: 900;
        margin-bottom: 10px;
        letter-spacing: -0.8px;
    }}

    .main-subtitle {{
        font-size: 18px;
        max-width: 920px;
        line-height: 1.7;
        opacity: 0.98;
    }}

    .section-title {{
        border-left: 8px solid {theme['primary']};
        padding-left: 16px;
        margin-top: 18px;
        margin-bottom: 18px;
        font-size: 27px;
        font-weight: 900;
        color: {theme['dark']};
    }}

    .info-card {{
        background: white;
        padding: 24px;
        border-radius: 22px;
        border: 1px solid #E5E7EB;
        box-shadow: 0 12px 28px rgba(15,23,42,0.08);
        margin-bottom: 18px;
    }}

    .kpi-card {{
        background: white;
        padding: 22px;
        border-radius: 22px;
        border: 1px solid #E5E7EB;
        box-shadow: 0 12px 30px rgba(15,23,42,0.08);
        position: relative;
        overflow: hidden;
    }}

    .kpi-card::before {{
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        height: 6px;
        width: 100%;
        background: {theme['primary']};
    }}

    .kpi-label {{
        font-size: 14px;
        color: #6B7280;
        font-weight: 700;
        margin-bottom: 8px;
    }}

    .kpi-value {{
        font-size: 32px;
        font-weight: 900;
        color: {theme['dark']};
    }}

    .kpi-note {{
        font-size: 13px;
        color: #6B7280;
        margin-top: 6px;
    }}

    .business-card {{
        background: {theme['soft']};
        border-left: 6px solid {theme['primary']};
        padding: 20px 24px;
        border-radius: 18px;
        margin-bottom: 14px;
    }}

    .footer {{
        text-align: center;
        color: #6B7280;
        font-size: 13px;
        margin-top: 40px;
        padding: 24px;
    }}

    div[data-testid="stMetric"] {{
        background: white;
        padding: 18px;
        border-radius: 18px;
        border: 1px solid #E5E7EB;
        box-shadow: 0 10px 24px rgba(15,23,42,0.06);
    }}
    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def render_header():
    st.markdown(
        f"""
        <div class="main-header">
            <div class="header-badge">{operator} Inspired Telecom Intelligence</div>
            <div class="main-title">Telecom Network Anomaly Detection Dashboard</div>
            <div class="main-subtitle">
                Interactive decision-support dashboard for telecom network monitoring,
                anomaly detection, cybersecurity analytics and NOC/SOC operational intelligence.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


def section_title(title):
    st.markdown(f'<div class="section-title">{title}</div>', unsafe_allow_html=True)


def kpi_card(label, value, note=""):
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
            <div class="kpi-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


def display_figure(file_name, caption=None):
    fig_path = FIGURES_DIR / file_name
    if fig_path.exists():
        st.image(str(fig_path), caption=caption, use_container_width=True)
    else:
        st.warning(f"Figure not found: {file_name}")


def format_pct(x):
    return f"{x * 100:.2f}%"


# ============================================================
# EMPTY DATA CHECK
# ============================================================

render_header()

if data.empty:
    st.error(
        "Dataset not found. Please make sure `data/processed/telecom_network_modeling_dataset.csv` exists."
    )
    st.stop()


# ============================================================
# SIDEBAR FILTERS
# ============================================================

st.sidebar.markdown("### Filters")

if "Network_Status" in data.columns:
    selected_status = st.sidebar.multiselect(
        "Network status",
        sorted(data["Network_Status"].dropna().unique().tolist()),
        default=sorted(data["Network_Status"].dropna().unique().tolist())
    )
else:
    selected_status = []

if "Label" in data.columns:
    selected_labels = st.sidebar.multiselect(
        "Traffic / attack labels",
        sorted(data["Label"].dropna().unique().tolist()),
        default=sorted(data["Label"].dropna().unique().tolist())
    )
else:
    selected_labels = []

filtered_data = data.copy()

if selected_status and "Network_Status" in filtered_data.columns:
    filtered_data = filtered_data[filtered_data["Network_Status"].isin(selected_status)]

if selected_labels and "Label" in filtered_data.columns:
    filtered_data = filtered_data[filtered_data["Label"].isin(selected_labels)]


# ============================================================
# TABS
# ============================================================

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "Executive Overview",
        "Network Traffic Analysis",
        "Machine Learning Performance",
        "Risk Drivers",
        "Data Explorer"
    ]
)


# ============================================================
# TAB 1 — EXECUTIVE OVERVIEW
# ============================================================

with tab1:
    section_title("Executive Operations Overview")

    total_flows = len(filtered_data)

    anomaly_count = (
        filtered_data["Network_Status"].eq("Anomaly").sum()
        if "Network_Status" in filtered_data.columns else 0
    )

    normal_count = (
        filtered_data["Network_Status"].eq("Normal").sum()
        if "Network_Status" in filtered_data.columns else 0
    )

    anomaly_rate = anomaly_count / total_flows if total_flows > 0 else 0

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        kpi_card("Total network flows", f"{total_flows:,}", "Filtered records")
    with c2:
        kpi_card("Anomalous flows", f"{anomaly_count:,}", "Detected attack/suspicious records")
    with c3:
        kpi_card("Normal flows", f"{normal_count:,}", "Legitimate traffic")
    with c4:
        kpi_card("Anomaly rate", format_pct(anomaly_rate), "Operational risk indicator")

    st.markdown(
        """
        <div class="info-card">
        This dashboard presents a telecom network anomaly detection project based on CICIDS2017 network-flow data.
        The objective is to support telecom operators in identifying suspicious flows, monitoring attack patterns
        and improving cybersecurity decision-making for NOC/SOC teams.
        </div>
        """,
        unsafe_allow_html=True
    )

    c1, c2 = st.columns([1.1, 1])

    with c1:
        if "Network_Status" in filtered_data.columns:
            status_counts = filtered_data["Network_Status"].value_counts().reset_index()
            status_counts.columns = ["Network_Status", "Count"]

            fig = px.pie(
                status_counts,
                names="Network_Status",
                values="Count",
                hole=0.45,
                title="Normal vs Anomaly Traffic",
                color="Network_Status",
                color_discrete_map={
                    "Normal": "#2563EB",
                    "Anomaly": theme["primary"]
                }
            )
            fig.update_layout(height=430)
            st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.markdown(
            f"""
            <div class="business-card">
            <b>Final supervised model:</b><br>
            HistGradientBoosting was selected as the final model because it achieved the best
            balance between accuracy, recall, F1-score and ROC-AUC under attack-stratified validation.
            </div>
            <div class="business-card">
            <b>Important limitation:</b><br>
            The source-file validation shows that supervised models may struggle with attack families
            that were not represented during training. For zero-day attacks, an unsupervised detection layer is recommended.
            </div>
            """,
            unsafe_allow_html=True
        )


# ============================================================
# TAB 2 — NETWORK TRAFFIC ANALYSIS
# ============================================================

with tab2:
    section_title("Network Traffic Analysis")

    c1, c2 = st.columns([1.2, 1])

    with c1:
        if "Label" in filtered_data.columns:
            label_counts = filtered_data["Label"].value_counts().reset_index()
            label_counts.columns = ["Label", "Count"]

            fig = px.bar(
                label_counts.head(15),
                x="Label",
                y="Count",
                title="Traffic Label Distribution",
                color="Label",
                color_discrete_sequence=[theme["primary"]]
            )
            fig.update_layout(xaxis_tickangle=-45, height=520, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.markdown("#### Saved EDA Figures")
        display_figure("01_global_label_distribution.png", "Global label distribution")

    st.markdown("---")

    c3, c4 = st.columns(2)

    with c3:
        display_figure("02_feature_comparison_boxplots.png", "Key feature comparison")
    with c4:
        display_figure("03_correlation_heatmap.png", "Correlation heatmap")


# ============================================================
# TAB 3 — MACHINE LEARNING PERFORMANCE
# ============================================================

with tab3:
    section_title("Machine Learning Performance")

    if not model_results.empty:
        st.markdown("### Model Comparison")
        st.dataframe(model_results, use_container_width=True)

        metric_cols = [col for col in ["Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC"] if col in model_results.columns]

        if "Model" in model_results.columns and metric_cols:
            model_plot = model_results.melt(
                id_vars="Model",
                value_vars=metric_cols,
                var_name="Metric",
                value_name="Score"
            )

            fig = px.bar(
                model_plot,
                x="Metric",
                y="Score",
                color="Model",
                barmode="group",
                title="Model Performance Comparison",
            )
            fig.update_layout(height=480, yaxis_tickformat=".1%")
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Model results file not found.")

    c1, c2 = st.columns(2)

    with c1:
        display_figure("04_model_comparison.png", "Model comparison")
    with c2:
        display_figure("05_overfitting_audit.png", "Overfitting audit")

    st.markdown("---")

    c3, c4 = st.columns(2)

    with c3:
        display_figure("06_confusion_matrix.png", "Final confusion matrix")
    with c4:
        display_figure("07_roc_pr_curves.png", "ROC and Precision-Recall curves")

    st.markdown("### Final Model Summary")

    if not final_summary.empty:
        st.dataframe(final_summary, use_container_width=True)
    else:
        st.info("Final model summary file not found. The notebook summary can still be shown through saved figures.")


# ============================================================
# TAB 4 — RISK DRIVERS
# ============================================================

with tab4:
    section_title("Risk Drivers & Feature Importance")

    st.markdown(
        """
        <div class="info-card">
        Feature importance analysis helps identify which network-flow variables contribute most to anomaly detection.
        These indicators can support telecom NOC/SOC teams in understanding suspicious traffic behavior.
        </div>
        """,
        unsafe_allow_html=True
    )

    c1, c2 = st.columns(2)

    with c1:
        display_figure("08_feature_importance_rf.png", "Random Forest feature importance")
    with c2:
        display_figure("09_permutation_importance.png", "Permutation importance — HistGradientBoosting")

    if not feature_importance.empty:
        st.markdown("### Feature Importance Table")
        st.dataframe(feature_importance.head(30), use_container_width=True)
    else:
        st.info("Feature importance CSV not found. Saved figures are displayed instead.")

    st.markdown("### Business Interpretation")

    st.markdown(
        f"""
        <div class="business-card">
        <b>Destination Port</b> appears as a key driver because several network attacks target specific ports or services.
        </div>
        <div class="business-card">
        <b>Packet length and byte asymmetry features</b> help identify abnormal communication patterns between forward and backward traffic.
        </div>
        <div class="business-card">
        <b>TCP window and flag-based features</b> provide useful signals for distinguishing normal sessions from suspicious flows.
        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# TAB 5 — DATA EXPLORER
# ============================================================

with tab5:
    section_title("Filtered Network Traffic Data")

    st.markdown(
        f"""
        <div class="info-card">
        The table below shows the filtered network-flow dataset used in the dashboard.
        The full dataset can be explored using the sidebar filters.
        </div>
        """,
        unsafe_allow_html=True
    )

    st.dataframe(filtered_data.head(1000), use_container_width=True)

    csv = filtered_data.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Download filtered data as CSV",
        data=csv,
        file_name="filtered_telecom_network_data.csv",
        mime="text/csv"
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    f"""
    <div class="footer">
    {operator} Inspired Telecom Network Anomaly Detection Dashboard ·
    Streamlit · Machine Learning · Cybersecurity Analytics · NOC/SOC Intelligence
    </div>
    """,
    unsafe_allow_html=True
)