# how to run (cmd)
# cd C:\Users\jasqi\OneDrive\Documents\Stevens\aai627 project
# streamlit run app.py

import pandas as pd
import plotly.express as px
import streamlit as st
import numpy as np

# Page Configuration
st.set_page_config(page_title="E-Commerce Analytics Dashboard", layout="wide")
st.title("Customer Segmentation & Churn Risk Analytics")

# 1. Load Precomputed Data
@st.cache_data
def load_data():
    return pd.read_csv("customer_analytics_processed.csv")

df = load_data()

from sklearn.ensemble import RandomForestClassifier

@st.cache_resource
def get_what_if_model(data: pd.DataFrame):
    base_cols = ["Frequency", "Monetary", "Avg_Rating", "Age", "Loyalty_Member", "Cancellation_Rate"]
    available = [c for c in base_cols if c in data.columns]
    X_sim = data[available].fillna(0)
    y_sim = data["Churned"].astype(int)
    clf = RandomForestClassifier(n_estimators=80, max_depth=6, random_state=42)
    clf.fit(X_sim, y_sim)
    return clf, available

sim_model, sim_cols = get_what_if_model(df)

# 2. Executive KPI Header
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Customers", f"{len(df):,}")
col2.metric("Overall Churn Rate", f"{df['Churned'].mean():.1%}")
col3.metric("Avg Customer Spend", f"${df['Monetary'].mean():,.2f}")
col4.metric("Loyalty Enrollment", f"{df['Loyalty_Member'].mean():.1%}")

st.divider()

# 3. Sidebar Filters
st.sidebar.header("Filter Segments")
selected_segments = st.sidebar.multiselect(
    "Select Customer Segments:",
    options=sorted(df["Segment"].unique()),
    default=sorted(df["Segment"].unique())
)

filtered_df = df[df["Segment"].isin(selected_segments)]

# 4. Interactive Visualizations
left_col, right_col = st.columns(2)

with left_col:
    st.subheader("Customer Clusters (PCA Projection)")
    fig_pca = px.scatter(
        filtered_df,
        x="PCA1",
        y="PCA2",
        color="Segment",
        hover_data=["Customer ID", "Monetary", "Frequency", "Churn_Risk_Score"],
        title="2D Projection of Customer Personas"
    )
    st.plotly_chart(fig_pca, use_container_width=True)

with right_col:
    st.subheader("Churn Risk vs. Monetary Spend")
    fig_churn = px.scatter(
        filtered_df,
        x="Monetary",
        y="Churn_Risk_Score",
        color="Segment",
        hover_data=["Customer ID"],
        title="Spending vs. Predicted Churn Probability"
    )
    st.plotly_chart(fig_churn, use_container_width=True)

st.divider()
st.subheader("Revenue-at-Risk Prioritization Matrix & Outreach Capacity")

spend_cutoff = filtered_df["Monetary"].median()
risk_cutoff = 0.50

def assign_quadrant(row):
    high_val = row["Monetary"] >= spend_cutoff
    high_risk = row["Churn_Risk_Score"] >= risk_cutoff
    if high_val and high_risk:
        return "1. Priority Rescue (High Value, High Risk)"
    elif high_val and not high_risk:
        return "2. VIP Loyalty Nurture (High Value, Low Risk)"
    elif not high_val and high_risk:
        return "3. Low-Cost Win-Back (Low Value, High Risk)"
    return "4. Standard Monitoring (Low Value, Low Risk)"

matrix_df = filtered_df.copy()
matrix_df["Quadrant"] = matrix_df.apply(assign_quadrant, axis=1)

fig_quad = px.scatter(
    matrix_df,
    x="Monetary",
    y="Churn_Risk_Score",
    color="Quadrant",
    hover_data=["Customer ID", "Segment", "Frequency", "Avg_Rating"],
    title="Spend vs. Predicted Churn Risk (Quadrant Prioritization)"
)
fig_quad.add_vline(x=spend_cutoff, line_dash="dash", line_color="gray")
fig_quad.add_hline(y=risk_cutoff, line_dash="dash", line_color="gray")
st.plotly_chart(fig_quad, use_container_width=True)

max_n = max(10, min(500, len(matrix_df)))
top_n = st.slider("Outreach Capacity (Top-N Highest Churn Risk Customers):", min_value=10, max_value=max_n, value=min(50, max_n), step=10)

priority_list = matrix_df.sort_values(by=["Churn_Risk_Score", "Monetary"], ascending=[False, False]).head(top_n)
protected_revenue = priority_list["Monetary"].sum()
avg_top_risk = priority_list["Churn_Risk_Score"].mean()

m1, m2, m3 = st.columns(3)
m1.metric("Target Outreach List Size", f"{len(priority_list):,} customers")
m2.metric("Gross Spend in Top-N List", f"${protected_revenue:,.2f}")
m3.metric("Avg Churn Risk (Top-N)", f"{avg_top_risk:.1%}")

st.dataframe(
    priority_list[["Customer ID", "Segment", "Quadrant", "Monetary", "Frequency", "Avg_Rating", "Churn_Risk_Score"]].round({"Monetary": 2, "Avg_Rating": 2, "Churn_Risk_Score": 3}),
    use_container_width=True
)

st.divider()

# 5. Individual Customer Profile Search
st.subheader("Customer Risk Lookup")
customer_list = df["Customer ID"].unique()
selected_cust = st.selectbox("Search / Select a Customer ID:", customer_list)

cust_data = df[df["Customer ID"] == selected_cust].iloc[0]

c1, c2, c3, c4 = st.columns(4)
c1.metric("Assigned Segment", f"Segment {cust_data['Segment']}")
c2.metric("Predicted Churn Risk", f"{cust_data['Churn_Risk_Score']:.1%}")
c3.metric("Recency", f"{int(cust_data['Recency'])} days ago")
c4.metric("Total Spend", f"${cust_data['Monetary']:,.2f}")

st.markdown("#### Why Is This Customer at Risk? (Feature Comparison vs. Dataset Average)")
compare_cols = [c for c in ["Frequency", "Monetary", "Avg_Rating", "Cancellation_Rate", "Loyalty_Member"] if c in df.columns]
pop_means = df[compare_cols].mean()
cust_vals = cust_data[compare_cols]

# Relative % difference from population average
rel_diff_pct = ((cust_vals - pop_means) / pop_means.replace(0, np.nan)).fillna(0) * 100
explain_df = pd.DataFrame({
    "Feature": compare_cols,
    "Customer Value": [round(float(cust_vals[c]), 2) for c in compare_cols],
    "Dataset Avg": [round(float(pop_means[c]), 2) for c in compare_cols],
    "Diff vs Avg (%)": [round(float(rel_diff_pct[c]), 1) for c in compare_cols],
})

fig_exp = px.bar(
    explain_df,
    x="Diff vs Avg (%)",
    y="Feature",
    orientation="h",
    hover_data=["Customer Value", "Dataset Avg"],
    title=f"Customer {selected_cust} Profile Deviation vs. Average Customer (%)"
)
st.plotly_chart(fig_exp, use_container_width=True)

st.markdown("#### 'What-If' Retention Lever Simulator")
sim_c1, sim_c2, sim_c3 = st.columns(3)
sim_loyalty = sim_c1.selectbox(
    "Simulated Loyalty Member Status",
    options=[0, 1],
    index=int(cust_data["Loyalty_Member"]) if "Loyalty_Member" in cust_data else 0,
    format_func=lambda x: "Yes (1)" if x == 1 else "No (0)"
)
sim_rating = sim_c2.slider(
    "Simulated Avg Rating",
    min_value=1.0,
    max_value=5.0,
    value=float(cust_data["Avg_Rating"]) if "Avg_Rating" in cust_data else 3.0,
    step=0.5
)
sim_cancel = sim_c3.slider(
    "Simulated Cancellation Rate",
    min_value=0.0,
    max_value=1.0,
    value=float(cust_data["Cancellation_Rate"]) if "Cancellation_Rate" in cust_data else 0.0,
    step=0.05
)

sim_row = pd.DataFrame([cust_data[sim_cols].to_dict()])
if "Loyalty_Member" in sim_row.columns:
    sim_row["Loyalty_Member"] = sim_loyalty
if "Avg_Rating" in sim_row.columns:
    sim_row["Avg_Rating"] = sim_rating
if "Cancellation_Rate" in sim_row.columns:
    sim_row["Cancellation_Rate"] = sim_cancel

new_prob = float(sim_model.predict_proba(sim_row.fillna(0))[0, 1])
delta_prob = new_prob - float(cust_data["Churn_Risk_Score"])

s_col1, s_col2 = st.columns(2)
s_col1.metric("Baseline Churn Risk", f"{cust_data['Churn_Risk_Score']:.1%}")
s_col2.metric("Simulated Churn Risk", f"{new_prob:.1%}", delta=f"{delta_prob:+.1%}", delta_color="inverse")