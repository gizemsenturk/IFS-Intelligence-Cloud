import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import shap
import streamlit as st
import pandas as pd
import numpy as np
from st_aggrid import AgGrid, GridOptionsBuilder
from src.explain import explain
import matplotlib.pyplot as plt
import joblib

from openai import OpenAI

@st.cache_resource
def load_model():
    return joblib.load("models/regression_model.pkl")

model = load_model()

@st.cache_resource
def get_client():
    return OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

client = get_client()

def generate_gpt_email(row):

    prompt = f"""
You are a B2B sales expert working for IFS ERP.

Write a personalized outreach email for the company below.

Company Name: {row['company']}
Lead Score: {row['predicted_score']}
Segment: {row['segment']}
Multi-site: {row['multi_site']}
Legacy dependency: {row['legacy_dependency']}
ERP hiring signal: {row['erp_hiring_signal']}

Instructions:
- Keep it professional but natural
- Focus on business value
- Mention relevant signals (if any)
- Keep it concise
- End with a clear call-to-action

Generate only the email.
"""

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": "You are an expert B2B sales assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7
    )

    return response.choices[0].message.content

# =========================
# HELPERS
# =========================
def generate_insight(row):
    insights = []

    # =========================
    # COMPLEXITY SIGNALS
    # =========================
    if row.get("multi_site", 0) == 1:
        insights.append("🏭 Multi-site operations increase operational complexity")

    # =========================
    # TECH DEBT SIGNALS
    # =========================
    if row.get("legacy_dependency", 0) > 0.7:
        insights.append("🧩 High dependency on legacy systems")
    
    # =========================
    # TRANSFORMATION SIGNALS
    # =========================
    if row.get("erp_hiring_signal", 0) > 0.6:
        insights.append("📈 Active ERP hiring suggests digital transformation")

    # =========================
    # DEFAULT FALLBACK
    # =========================
    if not insights:
        insights.append("✅ Standard ERP opportunity profile")

    return " | ".join(insights)

def recommend_action(row):
    score = row["predicted_score"]

    if score >= 85:
        return "📞 Call immediately & schedule demo (within 24h)"
    
    elif score >= 70:
        return "📅 Schedule product demo"
    
    elif score >= 50:
        return "📧 Send personalized email"
    
    elif score >= 30:
        return "📨 Add to nurture campaign"
    
    else:
        return "❌ Low priority"

def explain_reason(row):
    reasons = []

    if row["multi_site"] == 1:
        reasons.append("Multi-site operations increase ERP complexity")

    if row["legacy_dependency"] > 0.7:
        reasons.append("Legacy system dependency is high")

    if row["erp_hiring_signal"] > 0.6:
        reasons.append("ERP hiring activity detected")

    if not reasons:
        reasons.append("Standard ERP opportunity profile with no strong risk or urgency signals")

    return " • ".join(reasons)


def sales_pitch(row):
    pitch = []

    if row["multi_site"] == 1:
        pitch.append("highlight multi-site efficiency")

    if row["legacy_dependency"] > 0.7:
        pitch.append("focus on legacy system replacement")

    if row["erp_hiring_signal"] > 0.6:
        pitch.append("emphasize digital transformation")

    if not pitch:
        pitch.append("introduce ERP modernization")

    return " + ".join(pitch)


# =========================
# CONFIG
# =========================
st.set_page_config(
    page_title="IFS Intelligence Cloud",
    page_icon="🏭",
    layout="wide"
)

# =========================
# THEME (NOTION + SAAS MIX)
# =========================
st.markdown("""
<style>

/* background */
.stApp {
    background-color: #0b0f1a;
    color: white;
}

/* sidebar = Notion style */
[data-testid="stSidebar"] {
    background-color: #0f172a;
    border-right: 1px solid #1f2937;
}

/* metrics = Salesforce cards */
[data-testid="metric-container"] {
    background-color: #111827;
    border-radius: 12px;
    padding: 14px;
    box-shadow: 0 3px 15px rgba(0,0,0,0.4);
}

/* buttons / tabs minimal */
button {
    border-radius: 8px !important;
}

/* hide streamlit default footer */
footer {visibility: hidden;}

</style>
""", unsafe_allow_html=True)

# =========================
# DATA LAYER
# =========================
if not os.path.exists("outputs/scored_leads.csv"):
    st.warning("⚠️ No data found")
    
    if st.button("Run Pipeline Guide"):
        st.code("python pipeline/train_pipeline.py")
    
    st.stop()

@st.cache_data
def load_data():
    return pd.read_csv("outputs/scored_leads.csv")

df = load_data()

# =========================
# NOTION-STYLE SIDEBAR
# =========================
st.sidebar.title("🏭 IFS Intelligence Cloud")

page = st.sidebar.radio(
    "Navigation",
    ["🏠 Home", "📊 Pipeline", "🔥 Deals", "🔍 Company CRM"]
)

st.sidebar.divider()

score_filter = st.sidebar.slider("Lead Score Filter", 0, 100, (0, 100))

filtered = df[
    (df["predicted_score"] >= score_filter[0]) &
    (df["predicted_score"] <= score_filter[1])
].copy()

# =========================
# SALESFORCE-STYLE KPI BAR
# =========================
col1, col2, col3, col4 = st.columns(4)

col1.metric("Accounts", len(df))
col2.metric("Avg Score", round(df["predicted_score"].mean(), 1))
col3.metric("High Intent", len(df[df["predicted_score"] > 75]))
col4.metric("Pipeline Value", "₺18.7M")

st.divider()

# =========================
# PAGE ROUTER
# =========================

# -------------------------
# HOME (Notion-like overview)
# -------------------------
if page == "🏠 Home":

    st.title("Welcome to Intelligence Cloud")
    st.caption("AI-powered ERP opportunity system")

    st.markdown("""
    ### 🧠 What this system does
    - Scores manufacturing companies for ERP opportunity
    - Identifies high-value sales leads
    - Explains AI-driven scoring logic
    """)

# -------------------------
# PIPELINE (TABLE + ANALYTICS)
# -------------------------
elif page == "📊 Pipeline":

    st.title("Lead Pipeline")

    # ---------------------------
    # Column Rename (Business UI)
    # ---------------------------
    display_df = filtered.copy()

    display_df = display_df.rename(columns={
        "company": "Company",
        "company_size": "Size",
        "multi_site": "Multi-Site",
        "legacy_dependency": "Legacy Dependency",
        "erp_hiring_signal": "ERP Hiring Signal",
        "predicted_score": "Score",
        "segment": "Segment"
    })

    # ---------------------------
    # Segment Badge
    # ---------------------------
    def color_segment(val):
        if val == "Hot":
            return "color: #ef4444;"
        elif val == "Warm":
            return "color: #facc15; font-weight: bold;"
        else:
            return "color: #22c55e; font-weight: bold;"

    # ---------------------------
    # Score Bar (visual emphasis)
    # ---------------------------
    styled_df = display_df.style \
        .background_gradient(subset=["Score"], cmap="Blues") \
        .format({
            "Score": "{:.1f}",
            "Legacy Dependency": "{:.2f}",
            "ERP Hiring Signal": "{:.2f}"
        }) \
        .map(color_segment, subset=["Segment"])

    # ---------------------------
    # Sort High → Low
    # ---------------------------
    display_df = display_df.sort_values(by="Score", ascending=False)

    # ---------------------------
    # TABLE (clean view)
    # ---------------------------
    st.dataframe(
        styled_df,
        use_container_width=True,
        height=500
    )

# -------------------------
# DEALS (SALESFORCE CARDS)
# -------------------------
elif page == "🔥 Deals":

    st.title("High Value Opportunities")

    top = df.sort_values("predicted_score", ascending=False).head(5)

    for _, r in top.iterrows():
        st.markdown(f"""
        <div style="
            background:#111827;
            border:1px solid #1f2937;
            padding:16px;
            border-radius:12px;
            margin-bottom:10px;
        ">
        <h3>🥇 {r['company']}</h3>
        <p>Score: <b>{round(r['predicted_score'],1)}</b></p>
        <p>ERP Opportunity: HIGH</p>
        <p style="color:#22c55e;">Action: Schedule Demo</p>
        </div>
        """, unsafe_allow_html=True)

# -------------------------
# CRM (REAL SAAS FEEL)
# -------------------------
elif page == "🔍 Company CRM":

    selected = st.selectbox("Select Account", df["company"])
    c = df[df["company"] == selected].iloc[0]
    # =========================
    # 🏢 HERO HEADER
    # =========================
    st.markdown(f"""
    <div style="
        background:#111827;
        padding:20px;
        border-radius:14px;
        border:1px solid #1f2937;
    ">
        <h2>🏢 {c['company']}</h2>
        <p style="color:#9ca3af;">Manufacturing Account • AI Scored</p>
    </div>
    """, unsafe_allow_html=True)
    st.divider()

    # =========================
    # 📊 KPI CARDS
    # =========================
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Lead Score", round(c["predicted_score"], 1))
        st.progress(c["predicted_score"] / 100)

    with col2:
        st.metric("Segment", c["segment"])

    with col3:
        st.metric("ERP Opportunity", "HIGH" if c["predicted_score"] > 70 else "MEDIUM")

    st.divider()

    # =========================
    # 🎯 ACTION + WHY
    # =========================
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🎯 Recommended Action")
        st.success(recommend_action(c))

    with col2:
        st.subheader("🧠 Why this account?")
        st.info(explain_reason(c))

    st.divider()
    # =========================
    # 🧠 SHAP  Explainability
    # =========================
    st.subheader("🧠 AI Score Breakdown")

    FEATURES = [
        "size_score",
        "multi_site",
        "legacy_dependency",
        "erp_hiring_signal"
    ]

    X = pd.DataFrame([c[FEATURES]])

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    shap_df = pd.DataFrame({
        "Feature": FEATURES,
        "Impact": shap_values[0]
    })

    st.bar_chart(shap_df.set_index("Feature"))

    top_feature = FEATURES[np.argmax(np.abs(shap_values[0]))]
    st.success(f"🚀 Biggest driver: {top_feature}")

    st.markdown("### 🔍 Interpretation")

    for i, feature in enumerate(FEATURES):
      val = shap_values[0][i]
      direction = "increased" if val > 0 else "decreased"
      st.write(f"• **{feature}** → {direction} score by {round(val,2)}")
 
    st.divider()

    # =========================
    # 💬 SALES PLAYBOOK
    # =========================
    st.subheader("💬 Sales Playbook")

    st.markdown(f"""
    **How to approach:**
    - {sales_pitch(c)}
    - focus on operational efficiency
    - position IFS as scalable ERP

    **Next steps:**
    - contact within 48h
    - qualify current ERP usage
    - propose tailored demo
    """)

    st.divider()

    # =========================
    # 📌 INSIGHTS CARD
    # =========================
    st.subheader("🧠 Intelligence Summary")
    st.warning(generate_insight(c))

    # =========================
    # 📧 EMAIL GENERATOR
    # =========================
    st.subheader("📧 AI Generated Email")

    if st.button("✨ Generate Email"):
       with st.spinner("Generating email..."):
        email = generate_gpt_email(c)
        st.code(email)
