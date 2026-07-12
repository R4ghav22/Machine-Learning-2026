
import os

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


# Page config

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# Global styling (visual polish only - no logic here)

st.markdown(
    """
    <style>
    .main .block-container {padding-top: 1.5rem; padding-bottom: 2.5rem;}

    .app-header {
        background: linear-gradient(135deg, #4F46E5 0%, #7C3AED 60%, #DB2777 100%);
        padding: 1.6rem 2rem;
        border-radius: 16px;
        color: white;
        margin-bottom: 1.4rem;
        box-shadow: 0 8px 24px rgba(79, 70, 229, 0.25);
    }
    .app-header h1 {margin: 0; font-size: 1.7rem; font-weight: 700;}
    .app-header p {margin: 0.35rem 0 0 0; opacity: 0.92; font-size: 0.95rem;}

    div[data-testid="stMetric"] {
        background: #FFFFFF;
        border: 1px solid #EEF0F5;
        border-radius: 14px;
        padding: 1rem 1.1rem 0.8rem 1.1rem;
        box-shadow: 0 2px 10px rgba(17, 24, 39, 0.04);
    }
    div[data-testid="stMetricLabel"] {font-weight: 600; color: #6B7280;}

    section[data-testid="stSidebar"] {
        border-right: 1px solid #EEF0F5;
    }

    .section-title {
        font-size: 1.15rem;
        font-weight: 700;
        margin: 0.2rem 0 0.6rem 0;
        color: #1F2937;
    }
    .insight-box {
        background: #F8F9FC;
        border-left: 4px solid #7C3AED;
        border-radius: 8px;
        padding: 0.7rem 1rem;
        font-size: 0.9rem;
        color: #374151;
        margin-bottom: 0.9rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

PLOTLY_TEMPLATE = "plotly_white"
CHURN_COLORS = {"No": "#4F46E5", "Yes": "#DB2777"}


# Constants that mirror the notebook exactly (PREDICTION PIPELINE - DO
# NOT MODIFY)


# Columns that were Label Encoded in the notebook (binary_cols).
# sklearn's LabelEncoder sorts unique string values alphabetically and
# assigns 0, 1, 2 ... in that order. For every one of these columns the
# raw values are simple two-class Yes/No or Male/Female pairs, so the
# alphabetical mapping below is exactly what LabelEncoder produced.
BINARY_LABEL_MAPS = {
    "gender": {"Female": 0, "Male": 1},
    "SeniorCitizen": {"No": 0, "Yes": 1},
    "Partner": {"No": 0, "Yes": 1},
    "Dependents": {"No": 0, "Yes": 1},
    "PhoneService": {"No": 0, "Yes": 1},
    "PaperlessBilling": {"No": 0, "Yes": 1},
}

# Columns that were One-Hot Encoded in the notebook (multi_cols), in the
# exact same order used for pd.get_dummies(..., drop_first=True).
MULTI_COLS = [
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaymentMethod",
    "MonthlySpentCategory",
    "TenureGroup",
]

# Numeric columns that were scaled with StandardScaler (used only for the
# Logistic Regression model, exactly as in the notebook: model_LR was
# fit on X_train_scaled while model_RF and xgb were fit on the raw,
# unscaled X_train / X_test).
NUM_COLS = ["tenure", "MonthlyCharges", "TotalCharges"]

# Raw category options, taken from the source dataset, used to populate
# the dropdowns so the user never has to type one-hot column names.
OPTIONS = {
    "gender": ["Female", "Male"],
    "SeniorCitizen": ["No", "Yes"],
    "Partner": ["No", "Yes"],
    "Dependents": ["No", "Yes"],
    "PhoneService": ["No", "Yes"],
    "PaperlessBilling": ["No", "Yes"],
    "MultipleLines": ["No", "Yes", "No phone service"],
    "InternetService": ["DSL", "Fiber optic", "No"],
    "OnlineSecurity": ["No", "Yes", "No internet service"],
    "OnlineBackup": ["No", "Yes", "No internet service"],
    "DeviceProtection": ["No", "Yes", "No internet service"],
    "TechSupport": ["No", "Yes", "No internet service"],
    "StreamingTV": ["No", "Yes", "No internet service"],
    "StreamingMovies": ["No", "Yes", "No internet service"],
    "Contract": ["Month-to-month", "One year", "Two year"],
    "PaymentMethod": [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ],
}

# Model evaluation results exactly as printed by classification_report /
# accuracy_score in the notebook (weighted avg precision/recall/f1).
# These are static values taken from the notebook's own output cells,
# not recomputed here, since no training/test split is available to the
# app and no retraining is permitted.
METRICS = pd.DataFrame(
    {
        "Model": ["Logistic Regression", "Random Forest", "XGBoost"],
        "Accuracy": [0.7946, 0.7100, 0.7953],
        "Precision": [0.79, 0.71, 0.79],
        "Recall": [0.79, 0.71, 0.80],
        "F1-score": [0.79, 0.71, 0.79],
    }
)


# Feature engineering helpers (copied 1:1 from the notebook's functions)
# Used by BOTH the prediction pipeline and the EDA page, so the two stay
# perfectly consistent with the notebook.

def mc(x):
    """Reproduces the notebook's MonthlySpentCategory bucketing."""
    if x < 40:
        return "Low"
    elif x < 70:
        return "Medium"
    elif x < 90:
        return "High"
    else:
        return "Very High"


def tenure_group(x):
    """Reproduces the notebook's TenureGroup bucketing."""
    if x <= 12:
        return "New"
    elif x <= 24:
        return "Short-term"
    elif x <= 48:
        return "Mid-term"
    elif x <= 60:
        return "Long-term"
    else:
        return "Loyal"



# Cached loaders for the artifacts saved by the notebook (PREDICTION
# PIPELINE - DO NOT MODIFY)

@st.cache_resource
def load_artifacts():
    lr_model = joblib.load("lr_model.pkl")
    rf_model = joblib.load("rf_model.pkl")
    xgb_model = joblib.load("xgb.pkl")
    scaler = joblib.load("scaler.pkl")
    feature_columns = joblib.load("feature_columns.pkl")
    return lr_model, rf_model, xgb_model, scaler, feature_columns


def build_raw_dataframe(inputs: dict) -> pd.DataFrame:
    """Assembles the single-row raw dataframe, exactly the shape/columns
    of `mldata.drop('Churn', axis=1).drop('customerID', axis=1)` before
    any encoding, with the two engineered columns added exactly as the
    notebook engineered them."""
    row = dict(inputs)
    row["MonthlySpentCategory"] = mc(row["MonthlyCharges"])
    row["TenureGroup"] = tenure_group(row["tenure"])
    return pd.DataFrame([row])


def preprocess(raw_df: pd.DataFrame, feature_columns: list) -> pd.DataFrame:
    """Applies, in order: Label Encoding on binary_cols -> One-Hot
    Encoding on multi_cols via pd.get_dummies(drop_first=True) ->
    reindexing to the exact training-time feature column order,
    filling any dummy columns absent from this single row with 0."""
    df = raw_df.copy()

    # 1) Label Encoding (binary_cols)
    for col, mapping in BINARY_LABEL_MAPS.items():
        df[col] = df[col].map(mapping)

    # 2) One-Hot Encoding (multi_cols) - identical call to the notebook
    df = pd.get_dummies(df, columns=MULTI_COLS, drop_first=True, dtype=int)

    # 3) Reindex to the exact training feature order/columns
    df = df.reindex(columns=feature_columns, fill_value=0)

    return df


def predict(model_choice, raw_df, feature_columns, lr_model, rf_model, xgb_model, scaler):
    X = preprocess(raw_df, feature_columns)

    if model_choice == "Logistic Regression":
        # Only Logistic Regression was trained on scaled numeric columns
        X_scaled = X.copy()
        X_scaled[NUM_COLS] = scaler.transform(X_scaled[NUM_COLS])
        model = lr_model
        X_final = X_scaled
    elif model_choice == "Random Forest":
        model = rf_model
        X_final = X
    else:  # XGBoost
        model = xgb_model
        X_final = X

    pred = model.predict(X_final)[0]
    proba = model.predict_proba(X_final)[0]
    return pred, proba, X_final


# EDA data loader - COMPLETELY SEPARATE from the prediction pipeline.
# Loads the raw Telco.csv and reproduces the notebook's cleaning /
# feature-engineering steps ONLY for charts and KPIs. Nothing here
# touches the models, the scaler, or feature_columns.pkl, and nothing
# here is fit/trained.

EDA_DATA_PATH = "Telco.csv"


@st.cache_data
def load_eda_data(path: str = EDA_DATA_PATH):
    df = pd.read_csv(path)

    # Drop the 11 rows with blank TotalCharges, exactly as the notebook
    # does by index (df.drop(index=[488, 753, ...])). Falling back to a
    # value-based drop keeps this robust if the CSV has been re-sorted.
    bad_index = [488, 753, 936, 1082, 1340, 3331, 3826, 4380, 5218, 6670, 6754]
    existing_bad_index = [i for i in bad_index if i in df.index]
    if existing_bad_index:
        df = df.drop(index=existing_bad_index)
    mask = pd.to_numeric(df["TotalCharges"], errors="coerce").isna()
    df = df.loc[~mask].copy()
    df["TotalCharges"] = df["TotalCharges"].astype(float)

    # SeniorCitizen: 0/1 -> No/Yes, exactly as the notebook
    if df["SeniorCitizen"].dtype != object:
        df["SeniorCitizen"] = df["SeniorCitizen"].map({0: "No", 1: "Yes"})

    # Engineered columns, exactly as the notebook
    df["MonthlySpentCategory"] = df["MonthlyCharges"].apply(mc)
    df["TenureGroup"] = df["tenure"].apply(tenure_group)
    df["AvgCharge"] = df["TotalCharges"] / df["tenure"].replace(0, np.nan)

    return df


def grouped_churn_bar(df, col, title, x_title, barmode="group", category_order=None):
    grp = df.groupby([col, "Churn"]).size().reset_index(name="Count")
    fig = px.bar(
        grp,
        x=col,
        y="Count",
        color="Churn",
        barmode=barmode,
        color_discrete_map=CHURN_COLORS,
        title=title,
        template=PLOTLY_TEMPLATE,
        category_orders={col: category_order} if category_order else None,
    )
    fig.update_layout(
        xaxis_title=x_title,
        yaxis_title="Customer Count",
        legend_title="Churn",
        margin=dict(t=60, b=10, l=10, r=10),
        height=380,
    )
    return fig



# SIDEBAR NAVIGATION

st.sidebar.markdown("## 📊 Churn Dashboard")
page = st.sidebar.radio(
    "Navigate",
    ["📊 Dashboard", "📈 EDA & Business Insights", "🤖 Prediction"],
    label_visibility="collapsed",
)
st.sidebar.markdown("---")


# PAGE: DASHBOARD

if page == "📊 Dashboard":
    st.markdown(
        """
        <div class="app-header">
            <h1>📊 Customer Churn Prediction Dashboard</h1>
            <p>An overview of the customer base, churn behavior, and the models trained to predict it.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not os.path.exists(EDA_DATA_PATH):
        st.warning(
            f"Couldn't find `{EDA_DATA_PATH}` in the app folder, so KPIs and charts can't be "
            "computed. Place the original dataset next to `app.py` to enable this page. "
            "The Prediction page works independently of this file."
        )
    else:
        df = load_eda_data()

        total_customers = len(df)
        churn_rate = (df["Churn"] == "Yes").mean()
        avg_monthly = df["MonthlyCharges"].mean()
        avg_tenure = df["tenure"].mean()

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Total Customers", f"{total_customers:,}")
        k2.metric("Churn Rate", f"{churn_rate:.1%}")
        k3.metric("Avg. Monthly Charges", f"${avg_monthly:,.2f}")
        k4.metric("Avg. Tenure", f"{avg_tenure:,.1f} mo")

        st.markdown("<br>", unsafe_allow_html=True)
        col_a, col_b = st.columns([1, 1.2])

        with col_a:
            st.markdown('<div class="section-title">Churn Distribution</div>', unsafe_allow_html=True)
            churn_counts = df["Churn"].value_counts().reset_index()
            churn_counts.columns = ["Churn", "Count"]
            fig = px.pie(
                churn_counts,
                names="Churn",
                values="Count",
                hole=0.55,
                color="Churn",
                color_discrete_map=CHURN_COLORS,
                template=PLOTLY_TEMPLATE,
            )
            fig.update_traces(textinfo="percent+label")
            fig.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=340, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        with col_b:
            st.markdown('<div class="section-title">Model Comparison</div>', unsafe_allow_html=True)
            st.caption("Metrics as reported in the source notebook's evaluation cells.")
            st.dataframe(
                METRICS.style.format(
                    {
                        "Accuracy": "{:.2%}",
                        "Precision": "{:.2f}",
                        "Recall": "{:.2f}",
                        "F1-score": "{:.2f}",
                    }
                ),
                hide_index=True,
                use_container_width=True,
            )
            best_row = METRICS.loc[METRICS["Accuracy"].idxmax()]
            st.metric("Best Accuracy", best_row["Model"], f"{best_row['Accuracy']:.2%}")

        st.markdown(
            """
            <div class="insight-box">
            💡 Head to <b>📈 EDA & Business Insights</b> for a full breakdown of churn drivers,
            or jump to <b>🤖 Prediction</b> to score an individual customer with a trained model.
            </div>
            """,
            unsafe_allow_html=True,
        )


# PAGE: EDA & BUSINESS INSIGHTS

elif page == "📈 EDA & Business Insights":
    st.markdown(
        """
        <div class="app-header">
            <h1>📈 EDA & Business Insights</h1>
            <p>An interactive dashboard for exploring the raw Telco dataset - filter, chart, and drill in on your own terms.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not os.path.exists(EDA_DATA_PATH):
        st.error(
            f"Couldn't find `{EDA_DATA_PATH}`. Place the original Telco dataset (the CSV the "
            "notebook read with `pd.read_csv('Telco.csv')`) in the same folder as `app.py` to "
            "enable this page. This page is used for visualization only - it never retrains or "
            "touches the prediction models."
        )
        st.stop()

    df = load_eda_data()

    categorical_cols = sorted(df.select_dtypes(include="object").columns.tolist())
    numeric_cols = sorted(df.select_dtypes(include=np.number).columns.tolist())
    all_cols = df.columns.tolist()

  
    # Chart builder helper - purely for visualization, built dynamically
    # from whatever the user selects. Never touches the model pipeline.
 
    def build_dynamic_chart(data, graph_type, x_col, y_col, hue_col):
        color = None if hue_col in (None, "None") else hue_col
        y = None if y_col in (None, "None") else y_col

        if graph_type == "Bar":
            grp_cols = [x_col] + ([color] if color else [])
            if y:
                agg = data.groupby(grp_cols, dropna=False)[y].mean().reset_index()
                fig = px.bar(
                    agg, x=x_col, y=y, color=color, barmode="group",
                    template=PLOTLY_TEMPLATE,
                    title=f"Average {y} by {x_col}" + (f" and {color}" if color else ""),
                )
            else:
                agg = data.groupby(grp_cols, dropna=False).size().reset_index(name="Count")
                fig = px.bar(
                    agg, x=x_col, y="Count", color=color, barmode="group",
                    template=PLOTLY_TEMPLATE,
                    title=f"Count by {x_col}" + (f" and {color}" if color else ""),
                )

        elif graph_type == "Pie":
            if y:
                agg = data.groupby(x_col, dropna=False)[y].sum().reset_index()
                fig = px.pie(
                    agg, names=x_col, values=y, hole=0.45, template=PLOTLY_TEMPLATE,
                    title=f"Share of {y} by {x_col}",
                )
            else:
                counts = data[x_col].value_counts(dropna=False).reset_index()
                counts.columns = [x_col, "Count"]
                fig = px.pie(
                    counts, names=x_col, values="Count", hole=0.45, template=PLOTLY_TEMPLATE,
                    title=f"Distribution of {x_col}",
                )
            fig.update_traces(textinfo="percent+label")

        elif graph_type == "Histogram":
            fig = px.histogram(
                data, x=x_col, color=color, nbins=40, opacity=0.85, barmode="overlay",
                template=PLOTLY_TEMPLATE,
                title=f"Distribution of {x_col}" + (f" by {color}" if color else ""),
            )

        elif graph_type == "Box":
            if y:
                fig = px.box(
                    data, x=x_col, y=y, color=color, template=PLOTLY_TEMPLATE,
                    title=f"{y} by {x_col}",
                )
            else:
                fig = px.box(
                    data, y=x_col, color=color, template=PLOTLY_TEMPLATE,
                    title=f"Distribution of {x_col}",
                )

        elif graph_type == "Scatter":
            if not y:
                raise ValueError("Scatter plots need both an X-axis and a Y-axis column.")
            fig = px.scatter(
                data, x=x_col, y=y, color=color, opacity=0.7, template=PLOTLY_TEMPLATE,
                title=f"{y} vs {x_col}",
            )

        else:  # Countplot
            grp_cols = [x_col] + ([color] if color else [])
            agg = data.groupby(grp_cols, dropna=False).size().reset_index(name="Count")
            fig = px.bar(
                agg, x=x_col, y="Count", color=color, barmode="group",
                template=PLOTLY_TEMPLATE,
                title=f"Count of {x_col}" + (f" by {color}" if color else ""),
            )

        fig.update_layout(margin=dict(t=60, b=10, l=10, r=10), height=440)
        return fig

   
    # 1) Filter the dataset
   
    st.markdown('<div class="section-title">🔎 Filter the Dataset</div>', unsafe_allow_html=True)
    st.caption("Pick any categorical columns to filter by (e.g. Contract, Gender, InternetService, Churn), then choose the values to keep.")

    filter_cols = st.multiselect(
        "Select categorical columns to filter by",
        categorical_cols,
        default=[c for c in ["Contract", "Churn"] if c in categorical_cols],
    )

    filtered_df = df.copy()
    if filter_cols:
        filt_columns = st.columns(min(len(filter_cols), 3))
        for i, col in enumerate(filter_cols):
            options = sorted(df[col].dropna().unique().tolist())
            with filt_columns[i % len(filt_columns)]:
                selected_vals = st.multiselect(f"{col}", options, default=options, key=f"filter_{col}")
            if selected_vals:
                filtered_df = filtered_df[filtered_df[col].isin(selected_vals)]
            else:
                filtered_df = filtered_df.iloc[0:0]

    st.caption(f"Showing **{len(filtered_df):,}** of **{len(df):,}** customers after filtering.")

    with st.expander("📋 View filtered dataframe", expanded=False):
        st.dataframe(filtered_df, use_container_width=True)
        csv_bytes = filtered_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download filtered data as CSV",
            data=csv_bytes,
            file_name="filtered_telco_data.csv",
            mime="text/csv",
            use_container_width=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

  
    # 2) KPIs (recomputed live from the filtered data)
   
    st.markdown('<div class="section-title">📌 Key Metrics (Filtered)</div>', unsafe_allow_html=True)
    if filtered_df.empty:
        st.warning("No rows match the current filters - relax a filter above to see metrics and charts.")
    else:
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Customers", f"{len(filtered_df):,}")
        k2.metric("Churn Rate", f"{(filtered_df['Churn'] == 'Yes').mean():.1%}")
        k3.metric("Avg. Monthly Charges", f"${filtered_df['MonthlyCharges'].mean():,.2f}")
        k4.metric("Avg. Tenure", f"{filtered_df['tenure'].mean():,.1f} mo")

    st.markdown("<br>", unsafe_allow_html=True)

 
    # 3) Interactive chart builder
  
    st.markdown('<div class="section-title">🎛️ Chart Builder</div>', unsafe_allow_html=True)
    st.caption("Build any chart you like from the filtered data - pick a graph type, then the columns that feed it.")

    cb1, cb2, cb3, cb4 = st.columns(4)

    with cb1:
        graph_type = st.selectbox(
            "Graph type", ["Bar", "Pie", "Histogram", "Box", "Scatter", "Countplot"]
        )

    y_applicable = graph_type in ("Bar", "Pie", "Box", "Scatter")

    with cb2:
        default_x = "Contract" if "Contract" in all_cols else all_cols[0]
        x_col = st.selectbox("X-axis column", all_cols, index=all_cols.index(default_x))

    with cb3:
        if y_applicable:
            y_options = ["None"] + numeric_cols
            default_y = "MonthlyCharges" if graph_type in ("Box", "Scatter") and "MonthlyCharges" in y_options else "None"
            y_col = st.selectbox("Y-axis column", y_options, index=y_options.index(default_y))
        else:
            y_col = None
            st.selectbox("Y-axis column", ["Not applicable for this chart"], disabled=True)

    with cb4:
        hue_options = ["None"] + categorical_cols
        default_hue = "Churn" if "Churn" in hue_options else "None"
        hue_col = st.selectbox("Hue / color column (optional)", hue_options, index=hue_options.index(default_hue))

    if filtered_df.empty:
        st.info("Adjust the filters above to see a chart here.")
    else:
        try:
            fig = build_dynamic_chart(filtered_df, graph_type, x_col, y_col, hue_col)
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Couldn't build this chart with the selected options: {e}")

    st.markdown("<br>", unsafe_allow_html=True)

    # ------------------------------------------------------------------
    # 4) Dataset summary, missing values, descriptive statistics
    # ------------------------------------------------------------------
    st.markdown('<div class="section-title">🧾 Dataset Summary</div>', unsafe_allow_html=True)
    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Rows (filtered)", f"{filtered_df.shape[0]:,}")
    s2.metric("Columns", f"{filtered_df.shape[1]:,}")
    s3.metric("Numeric Columns", f"{len(numeric_cols)}")
    s4.metric("Categorical Columns", f"{len(categorical_cols)}")

    sum_col1, sum_col2 = st.columns(2)

    with sum_col1:
        st.markdown("**Missing Values**")
        missing = filtered_df.isnull().sum()
        missing = missing[missing > 0].reset_index()
        missing.columns = ["Column", "Missing Count"]
        if missing.empty:
            st.success("No missing values in the filtered data. ✅")
        else:
            st.dataframe(missing, use_container_width=True, hide_index=True)

    with sum_col2:
        st.markdown("**Column Data Types**")
        dtypes_df = filtered_df.dtypes.astype(str).reset_index()
        dtypes_df.columns = ["Column", "Dtype"]
        st.dataframe(dtypes_df, use_container_width=True, hide_index=True, height=220)

    st.markdown("**Descriptive Statistics**")
    if filtered_df.empty:
        st.info("No data to describe - adjust the filters above.")
    else:
        st.dataframe(filtered_df.describe(include="all").transpose(), use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ------------------------------------------------------------------
    # 5) Correlation heatmap (all numeric columns, filtered data)
    # ------------------------------------------------------------------
    st.markdown('<div class="section-title">🔥 Correlation Among Numerical Features</div>', unsafe_allow_html=True)
    numeric_for_corr = [c for c in numeric_cols if c in filtered_df.columns]
    if filtered_df.empty or len(numeric_for_corr) < 2:
        st.info("Not enough numeric data to compute a correlation heatmap.")
    else:
        corr = filtered_df[numeric_for_corr].corr(numeric_only=True)
        fig = px.imshow(
            corr, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
            template=PLOTLY_TEMPLATE, title="Correlation Heatmap",
        )
        fig.update_layout(margin=dict(t=60, b=10, l=10, r=10), height=460)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown(
        """
        <div class="insight-box">
        💡 Use the filters and chart builder above to explore any relationship in the data -
        e.g. set X to <b>Contract</b>, Hue to <b>Churn</b>, and pick <b>Countplot</b> to see
        churn by contract type, or set X/Y to <b>tenure</b>/<b>TotalCharges</b> with
        <b>Scatter</b> to see how spend accumulates over time.
        </div>
        """,
        unsafe_allow_html=True,
    )


# PAGE: PREDICTION  (UNCHANGED PIPELINE)

else:
    st.markdown(
        """
        <div class="app-header">
            <h1>🤖 Customer Churn Prediction</h1>
            <p>Inference-only dashboard for the pre-trained Logistic Regression, Random Forest, and XGBoost churn models.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    try:
        lr_model, rf_model, xgb_model, scaler, feature_columns = load_artifacts()
    except FileNotFoundError as e:
        st.error(
            f"Could not find a required artifact: {e}. "
            "Make sure lr_model.pkl, rf_model.pkl, xgb.pkl, scaler.pkl and "
            "feature_columns.pkl are in the same folder as app.py."
        )
        st.stop()

    # ---------------- Sidebar: user inputs ----------------
    st.sidebar.header("🧾 Customer Details")

    with st.sidebar.expander("Demographics", expanded=True):
        gender = st.selectbox("Gender", OPTIONS["gender"])
        senior = st.selectbox("Senior Citizen", OPTIONS["SeniorCitizen"])
        partner = st.selectbox("Has Partner", OPTIONS["Partner"])
        dependents = st.selectbox("Has Dependents", OPTIONS["Dependents"])

    with st.sidebar.expander("Account Info", expanded=True):
        tenure = st.slider("Tenure (months)", 0, 72, 12)
        contract = st.selectbox("Contract", OPTIONS["Contract"])
        paperless = st.selectbox("Paperless Billing", OPTIONS["PaperlessBilling"])
        payment = st.selectbox("Payment Method", OPTIONS["PaymentMethod"])
        monthly_charges = st.number_input(
            "Monthly Charges ($)", min_value=0.0, max_value=200.0, value=70.0, step=0.5
        )
        default_total = round(monthly_charges * max(tenure, 1), 2)
        total_charges = st.number_input(
            "Total Charges ($)", min_value=0.0, value=default_total, step=1.0
        )

    with st.sidebar.expander("Services", expanded=True):
        phone_service = st.selectbox("Phone Service", OPTIONS["PhoneService"])
        multiple_lines = st.selectbox("Multiple Lines", OPTIONS["MultipleLines"])
        internet_service = st.selectbox("Internet Service", OPTIONS["InternetService"])
        online_security = st.selectbox("Online Security", OPTIONS["OnlineSecurity"])
        online_backup = st.selectbox("Online Backup", OPTIONS["OnlineBackup"])
        device_protection = st.selectbox("Device Protection", OPTIONS["DeviceProtection"])
        tech_support = st.selectbox("Tech Support", OPTIONS["TechSupport"])
        streaming_tv = st.selectbox("Streaming TV", OPTIONS["StreamingTV"])
        streaming_movies = st.selectbox("Streaming Movies", OPTIONS["StreamingMovies"])

    st.sidebar.markdown("---")
    model_choice = st.sidebar.selectbox(
        "🤖 Select Model", ["Logistic Regression", "Random Forest", "XGBoost"]
    )
    predict_clicked = st.sidebar.button("🔮 Predict Churn", use_container_width=True)

    # ---------------- Main area ----------------
    col_left, col_right = st.columns([1.3, 1])

    with col_left:
        st.subheader("Prediction")

        if predict_clicked:
            inputs = {
                "gender": gender,
                "SeniorCitizen": senior,
                "Partner": partner,
                "Dependents": dependents,
                "tenure": tenure,
                "PhoneService": phone_service,
                "MultipleLines": multiple_lines,
                "InternetService": internet_service,
                "OnlineSecurity": online_security,
                "OnlineBackup": online_backup,
                "DeviceProtection": device_protection,
                "TechSupport": tech_support,
                "StreamingTV": streaming_tv,
                "StreamingMovies": streaming_movies,
                "Contract": contract,
                "PaperlessBilling": paperless,
                "PaymentMethod": payment,
                "MonthlyCharges": monthly_charges,
                "TotalCharges": total_charges,
            }

            raw_df = build_raw_dataframe(inputs)
            pred, proba, X_final = predict(
                model_choice, raw_df, feature_columns, lr_model, rf_model, xgb_model, scaler
            )

            churn_prob = proba[1]
            stay_prob = proba[0]

            if pred == 1:
                st.error("### ⚠️ This customer is LIKELY TO CHURN")
            else:
                st.success("### ✅ This customer is LIKELY TO STAY")

            m1, m2, m3 = st.columns(3)
            m1.metric("Model Used", model_choice)
            m2.metric("Churn Probability", f"{churn_prob:.1%}")
            m3.metric("Retention Probability", f"{stay_prob:.1%}")

            st.progress(float(churn_prob), text=f"Churn risk: {churn_prob:.1%}")

            with st.expander("🔍 View engineered features sent to the model"):
                st.dataframe(raw_df)
        else:
            st.info("Fill in the customer details in the sidebar and click **Predict Churn**.")

    with col_right:
        st.subheader("Model Comparison")
        st.caption("Metrics as reported in the source notebook's evaluation cells.")
        st.dataframe(
            METRICS.style.format(
                {"Accuracy": "{:.2%}", "Precision": "{:.2f}", "Recall": "{:.2f}", "F1-score": "{:.2f}"}
            ),
            hide_index=True,
            use_container_width=True,
        )

        best_row = METRICS.loc[METRICS["Accuracy"].idxmax()]
        st.metric("Best Accuracy", best_row["Model"], f"{best_row['Accuracy']:.2%}")

    st.markdown("---")
    with st.expander("ℹ️ About this app"):
        st.write(
            "This dashboard performs inference only. All models, the scaler, "
            "and the feature column order were trained/fitted in the source "
            "notebook and loaded here from `.pkl` files. Preprocessing "
            "(label encoding, one-hot encoding, column reindexing, and "
            "selective scaling) mirrors the notebook exactly so predictions "
            "match what the notebook would produce for the same inputs."
        )
