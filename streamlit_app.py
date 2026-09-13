"""
Life Insurance Analysis Dashboard

This Streamlit application provides a comprehensive analysis of life insurance data
using Bayesian decision models. The application includes data visualization,
interactive filtering, and predictive modeling capabilities.

"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from data_preprocessing import load_and_preprocess_data, calculate_risk_score, create_age_and_salary_bins
from bayesian_model import BayesianLifeInsuranceModel
import os
from sklearn.model_selection import train_test_split

# Configure page settings
st.set_page_config(
    page_title="Life Insurance Analysis Dashboard",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Force light theme and custom CSS for white background and improved styling
st.markdown("""
<style>
    /* Force light theme */
    .stApp {
        background-color: white;
    }

    /* Main background */
    .main {
        background-color: white;
    }

    /* Sidebar styling */
    .css-1d391kg {
        background-color: #f8f9fa;
        color: #2c3e50; /* Darker text for sidebar */
    }

    /* Header styling */
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #2c3e50; /* Darker text for header */
        text-align: center;
        margin-bottom: 2rem;
        padding: 1rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }

    /* Metric cards */
    .metric-card {
        background-color: white;
        padding: 1.5rem;
        border-radius: 10px;
        border: 1px solid #e9ecef;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
        color: #2c3e50; /* Darker text for metric cards */
    }

    /* Sidebar header */
    .sidebar-header {
        font-size: 1.3rem;
        font-weight: bold;
        color: #2c3e50; /* Darker text for sidebar header */
        margin-bottom: 1.5rem;
        padding: 0.5rem;
        background-color: #e9ecef;
        border-radius: 5px;
        text-align: center;
    }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 0.5rem;
    }

    .stTabs [aria-selected="true"] {
        background-color: #007bff;
        color: white;
        border-color: #007bff;
    }

    /* Info box styling */
    .stAlert {
        background-color: #e7f3ff;
        border: 1px solid #b3d9ff;
        border-radius: 8px;
        color: #2c3e50; /* Darker text for info boxes */
    }

    /* Button styling */
    .stButton > button {
        background-color: #007bff;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 0.5rem 1rem;
        font-weight: 500;
        transition: all 0.3s ease;
    }

    .stButton > button:hover {
        background-color: #0056b3;
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }

    /* Selectbox styling */
    .stSelectbox > div > div {
        background-color: white;
        border: 1px solid #ced4da;
        border-radius: 5px;
        color: #2c3e50; /* Darker text for selectbox */
    }

    /* Text input styling */
    .stTextInput > div > div > input {
        background-color: white;
        border: 1px solid #ced4da;
        border-radius: 5px;
        color: #2c3e50; /* Darker text for text input */
    }

    /* Slider styling */
    .stSlider > div > div > div {
        background-color: #007bff;
    }

    /* Chart container */
    .chart-container {
        background-color: white;
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #e9ecef;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        margin-bottom: 1rem;
        color: #2c3e50; /* Darker text for chart containers */
    }

    /* Section headers */
    .section-header {
        color: #2c3e50; /* Darker text for section headers */
        font-size: 1.8rem;
        font-weight: 600;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #007bff;
    }

    /* Subheader styling */
    .stSubheader {
        color: #495057; /* Darker text for subheaders */
        font-weight: 500;
    }

    /* Ensure all text is visible */
    body {
        color: #2c3e50; /* Default text color for the body */
    }

    /* Specific adjustments for text within Streamlit components */
    .stMarkdown, .stText, .stLabel, .stNumberInput, .stDateInput, .stTimeInput {
        color: #2c3e50; /* Darker text for various Streamlit components */
    }

    /* Adjust slider text color */
    .stSlider .st-bh, .stSlider .st-bi {
        color: #2c3e50; /* Darker text for slider values */
    }

    /* Adjust selectbox text color */
    .stSelectbox .st-bd, .stSelectbox .st-be {
        color: #2c3e50; /* Darker text for selectbox selected value */
    }

    /* Adjust text input text color */
    .stTextInput .st-bd, .stTextInput .st-be {
        color: #2c3e50; /* Darker text for text input value */
    }

    /* General text color for all elements */
    * {
        color: #2c3e50 !important;
    }

</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """
    Load and preprocess the life insurance data.

    Returns:
        pd.DataFrame: Preprocessed DataFrame with all necessary columns
    """
    try:
        script_dir = os.path.dirname(__file__)
        file_path = os.path.join(script_dir, "data", "lifeInsurance.txt")
        if not os.path.exists(file_path):
            st.error("lifeInsurance.txt not found. Add your local file to data/lifeInsurance.txt.")
            return None

        df = load_and_preprocess_data(file_path)
        df = calculate_risk_score(df)
        df = create_age_and_salary_bins(df)
        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

@st.cache_resource
def train_model(df):
    """
    Trains the Bayesian Life Insurance Model.

    Args:
        df (pd.DataFrame): The preprocessed DataFrame.

    Returns:
        tuple: Trained model, X_test, y_test, and evaluation metrics.
    """
    # Ensure 'Decision' column exists before dropping
    if "Decision" not in df.columns:
        st.error("Target variable 'Decision' not found in the DataFrame.")
        return None, None, None, None

    X = df.drop("Decision", axis=1)
    y = df["Decision"]

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    model = BayesianLifeInsuranceModel()
    model.fit(X_train, y_train)
    metrics = model.evaluate(X_test, y_test)

    return model, X_test, y_test, metrics

def create_sidebar_filters(df):
    """
    Create sidebar filters for data exploration.

    Args:
        df (pd.DataFrame): The life insurance DataFrame

    Returns:
        dict: Dictionary containing filter values
    """
    st.sidebar.markdown("<div class=\"sidebar-header\">🔍 Data Filters</div>", unsafe_allow_html=True)

    # Age range filter
    st.sidebar.markdown("**Age Range**")
    age_range = st.sidebar.slider(
        "Select age range:",
        min_value=int(df["Age"].min()),
        max_value=int(df["Age"].max()),
        value=(int(df["Age"].min()), int(df["Age"].max())),
        step=1,
        key="age_slider"
    )

    # Salary range filter
    st.sidebar.markdown("**Salary Range (€)**")
    salary_range = st.sidebar.slider(
        "Select salary range:",
        min_value=int(df["MonthlySalary"].min()),
        max_value=int(df["MonthlySalary"].max()),
        value=(int(df["MonthlySalary"].min()), int(df["MonthlySalary"].max())),
        step=100,
        key="salary_slider"
    )

    # Gender filter
    st.sidebar.markdown("**Gender**")
    gender_options = ["All"] + list(df["Gender"].unique())
    selected_gender = st.sidebar.selectbox(
        "Select gender:",
        gender_options,
        key="gender_select"
    )

    # Marital status filter
    st.sidebar.markdown("**Marital Status**")
    marital_options = ["All"] + list(df["MaritalStatus"].unique())
    selected_marital = st.sidebar.selectbox(
        "Select marital status:",
        marital_options,
        key="marital_select"
    )

    # Physical status filter
    st.sidebar.markdown("**Physical Status**")
    physical_options = ["All"] + list(df["PhysicalStatus"].unique())
    selected_physical = st.sidebar.selectbox(
        "Select physical status:",
        physical_options,
        key="physical_select"
    )

    # Chronic diseases filter
    st.sidebar.markdown("**Health Conditions**")
    chronic_options = ["All"] + list(df["ChronicDiseases"].unique())
    selected_chronic = st.sidebar.selectbox(
        "Select health condition:",
        chronic_options,
        key="chronic_select"
    )

    # Reset filters button
    st.sidebar.markdown("---")
    if st.sidebar.button("🔄 Reset All Filters", key="reset_filters"):
        st.experimental_rerun()

    return {
        "age_range": age_range,
        "salary_range": salary_range,
        "gender": selected_gender,
        "marital_status": selected_marital,
        "physical_status": selected_physical,
        "chronic_diseases": selected_chronic
    }

def apply_filters(df, filters):
    """
    Apply filters to the DataFrame.

    Args:
        df (pd.DataFrame): The original DataFrame
        filters (dict): Dictionary containing filter values

    Returns:
        pd.DataFrame: Filtered DataFrame
    """
    filtered_df = df.copy()

    # Apply age filter
    filtered_df = filtered_df[
        (filtered_df["Age"] >= filters["age_range"][0]) &
        (filtered_df["Age"] <= filters["age_range"][1])
    ]

    # Apply salary filter
    filtered_df = filtered_df[
        (filtered_df["MonthlySalary"] >= filters["salary_range"][0]) &
        (filtered_df["MonthlySalary"] <= filters["salary_range"][1])
    ]

    # Apply gender filter
    if filters["gender"] != "All":
        filtered_df = filtered_df[filtered_df["Gender"] == filters["gender"]]

    # Apply marital status filter
    if filters["marital_status"] != "All":
        filtered_df = filtered_df[filtered_df["MaritalStatus"] == filters["marital_status"]]

    # Apply physical status filter
    if filters["physical_status"] != "All":
        filtered_df = filtered_df[filtered_df["PhysicalStatus"] == filters["physical_status"]]

    # Apply chronic diseases filter
    if filters["chronic_diseases"] != "All":
        filtered_df = filtered_df[filtered_df["ChronicDiseases"] == filters["chronic_diseases"]]

    return filtered_df

def create_overview_metrics(df):
    """
    Create overview metrics cards with improved styling.

    Args:
        df (pd.DataFrame): The filtered DataFrame
    """
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown("<div class=\"metric-card\">", unsafe_allow_html=True)
        st.metric(
            label="📊 Total Cases",
            value=f"{len(df):,}",
            delta=None
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        acceptance_rate = (df["Decision"] == "Accept").mean() * 100
        st.markdown("<div class=\"metric-card\">", unsafe_allow_html=True)
        st.metric(
            label="✅ Acceptance Rate",
            value=f"{acceptance_rate:.1f}%",
            delta=None
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with col3:
        avg_age = df["Age"].mean()
        st.markdown("<div class=\"metric-card\">", unsafe_allow_html=True)
        st.metric(
            label="👥 Average Age",
            value=f"{avg_age:.1f} years",
            delta=None
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with col4:
        avg_salary = df["MonthlySalary"].mean()
        st.markdown("<div class=\"metric-card\">", unsafe_allow_html=True)
        st.metric(
            label="💰 Average Salary",
            value=f"€{avg_salary:,.0f}",
            delta=None
        )
        st.markdown("</div>", unsafe_allow_html=True)

def create_decision_distribution_chart(df):
    """
    Create a pie chart showing decision distribution.

    Args:
        df (pd.DataFrame): The filtered DataFrame

    Returns:
        plotly.graph_objects.Figure: The pie chart figure
    """
    decision_counts = df["Decision"].value_counts()

    fig = px.pie(
        values=decision_counts.values,
        names=decision_counts.index,
        title="Insurance Decision Distribution",
        color_discrete_map={"Accept": "#28a745", "Reject": "#dc3545"},
        hole=0.4
    )

    fig.update_traces(
        textposition="inside",
        textinfo="percent+label",
        textfont_size=14,
        marker = dict(line=dict(color='white', width=2))
    )

    fig.update_layout(
        font=dict(size=14),
        title_font_size=18,
        title_x=0.5,
        height=400,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.2,
            xanchor="center",
            x=0.5
        )
    )

    return fig

def create_age_group_chart(df):
    """
    Create a stacked bar chart showing decisions by age group.

    Args:
        df (pd.DataFrame): The filtered DataFrame

    Returns:
        plotly.graph_objects.Figure: The bar chart figure
    """
    age_decision = df.groupby(["AgeGroup", "Decision"]).size().unstack(fill_value=0)

    fig = px.bar(
        age_decision,
        x=age_decision.index,
        y=["Accept", "Reject"],
        title="Insurance Decisions by Age Group",
        labels={"value": "Number of Cases", "variable": "Decision"},
        color_discrete_map={"Accept": "#28a745", "Reject": "#dc3545"},
        barmode="stack"
    )

    fig.update_layout(
        xaxis_title="Age Group",
        yaxis_title="Number of Cases",
        font=dict(size=14),
        title_font_size=18,
        title_x=0.5,
        height=400,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.2,
            xanchor="center",
            x=0.5
        )
    )

    return fig

def create_scatter_plot(df):
    """
    Create a scatter plot of Age vs Monthly Salary colored by Decision.

    Args:
        df (pd.DataFrame): The filtered DataFrame

    Returns:
        plotly.graph_objects.Figure: The scatter plot figure
    """
    fig = px.scatter(
        df,
        x="Age",
        y="MonthlySalary",
        color="Decision",
        title="Age vs. Monthly Salary Analysis",
        labels={"MonthlySalary": "Monthly Salary (€)", "Age": "Age (years)"},
        color_discrete_map={"Accept": "#28a745", "Reject": "#dc3545"},
        opacity=0.7,
        size_max=10
    )

    fig.update_layout(
        font=dict(size=14),
        title_font_size=18,
        title_x=0.5,
        height=500,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.15,
            xanchor="center",
            x=0.5
        )
    )

    return fig

def create_correlation_heatmap(df):
    """
    Create a correlation heatmap for numerical variables.

    Args:
        df (pd.DataFrame): The filtered DataFrame

    Returns:
        plotly.graph_objects.Figure: The heatmap figure
    """
    # Select numerical columns and encode categorical variables
    df_encoded = df.copy()
    df_encoded["Gender_encoded"] = df_encoded["Gender"].map({"Female": 0, "Male": 1})
    df_encoded["MaritalStatus_encoded"] = df_encoded["MaritalStatus"].map({"Single": 0, "Married": 1})
    df_encoded["PhysicalStatus_encoded"] = df_encoded["PhysicalStatus"].map({
        "Sedentary": 0, "Moderately Active": 1, "Active": 2
    })
    df_encoded["ChronicDiseases_encoded"] = df_encoded["ChronicDiseases"].map({
        "No Conditions": 0, "Moderate": 1, "Severe": 2
    })
    df_encoded["Dependents_encoded"] = df_encoded["Dependents"].map({
        "0": 0, "1": 1, "2": 2, ">=3": 3
    })
    df_encoded["Decision_encoded"] = df_encoded["Decision"].map({"Reject": 0, "Accept": 1})

    # Select relevant columns for correlation
    correlation_cols = [
        "Age", "MonthlySalary", "RiskScore",
        "Gender_encoded", "MaritalStatus_encoded",
        "PhysicalStatus_encoded", "ChronicDiseases_encoded",
        "Dependents_encoded", "Decision_encoded"
    ]

    # Ensure all columns exist before calculating correlation
    existing_cols = [col for col in correlation_cols if col in df_encoded.columns]
    correlation_matrix = df_encoded[existing_cols].corr()

    fig = go.Figure(data=go.Heatmap(
        z=correlation_matrix.values,
        x=correlation_matrix.columns,
        y=correlation_matrix.columns,
        colorscale="Viridis",
        text=correlation_matrix.round(2).values,
        texttemplate="%{text}",
        hoverongaps=False
    ))

    fig.update_layout(
        title="Correlation Heatmap of Features",
        font=dict(size=12),
        title_font_size=18,
        title_x=0.5,
        height=600,
        xaxis_showgrid=False,
        yaxis_showgrid=False,
        yaxis_autorange='reversed'
    )

    return fig

def create_risk_score_distribution_chart(df):
    """
    Create a histogram of risk scores colored by decision.

    Args:
        df (pd.DataFrame): The filtered DataFrame

    Returns:
        plotly.graph_objects.Figure: The histogram figure
    """
    fig = px.histogram(
        df,
        x="RiskScore",
        color="Decision",
        title="Risk Score Distribution by Decision",
        labels={"RiskScore": "Risk Score", "count": "Number of Cases"},
        color_discrete_map={"Accept": "#28a745", "Reject": "#dc3545"},
        barmode="overlay",
        opacity=0.7
    )

    fig.update_layout(
        font=dict(size=14),
        title_font_size=18,
        title_x=0.5,
        height=400,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.2,
            xanchor="center",
            x=0.5
        )
    )

    return fig

def create_model_performance_chart(metrics):
    """
    Create a bar chart showing model performance metrics.

    Args:
        metrics (dict): Dictionary of model performance metrics

    Returns:
        plotly.graph_objects.Figure: The bar chart figure
    """
    metric_names = ["Accuracy", "Precision (Accept)", "Recall (Accept)", "F1-Score (Accept)",
                    "Precision (Reject)", "Recall (Reject)", "F1-Score (Reject)"]
    metric_values = [
        metrics.get("accuracy", 0),
        metrics.get("precision_accept", 0),
        metrics.get("recall_accept", 0),
        metrics.get("f1_score_accept", 0),
        metrics.get("precision_reject", 0),
        metrics.get("recall_reject", 0),
        metrics.get("f1_score_reject", 0)
    ]

    fig = go.Figure(data=[go.Bar(
        x=metric_names,
        y=metric_values,
        text=[f"{val:.2f}" for val in metric_values],
        textposition='auto',
        marker_color=["#007bff", "#28a745", "#28a745", "#28a745", "#dc3545", "#dc3545", "#dc3545"]
    )])

    fig.update_layout(
        title="Model Performance Metrics",
        font=dict(size=14),
        title_font_size=18,
        title_x=0.5,
        height=500,
        yaxis_title="Score",
        xaxis_tickangle=-45
    )

    return fig

def main():
    """
    Main function to run the Streamlit application.
    """
    st.markdown("<h1 class=\"main-header\">Life Insurance Analysis Dashboard</h1>", unsafe_allow_html=True)

    # Load data
    df = load_data()
    if df is None:
        st.warning("Data could not be loaded. Please check the data file and try again.")
        return

    # Train model
    model, X_test, y_test, metrics = train_model(df)
    if model is None:
        st.warning("Model could not be trained. Please check the data and model configuration.")
        return

    # Create sidebar filters
    filters = create_sidebar_filters(df)

    # Apply filters
    filtered_df = apply_filters(df, filters)

    # Create tabs for different sections
    tab_titles = [
        "📊 Overview & Demographics",
        "📈 Risk Analysis",
        "🤖 Model Performance",
        "💡 Predictions & Insights"
    ]
    tab1, tab2, tab3, tab4 = st.tabs(tab_titles)

    with tab1:
        st.markdown("<h2 class=\"section-header\">Dashboard Overview</h2>", unsafe_allow_html=True)
        create_overview_metrics(filtered_df)

        st.markdown("<h2 class=\"section-header\">Demographic Analysis</h2>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(create_decision_distribution_chart(filtered_df), use_container_width=True)
        with col2:
            st.plotly_chart(create_age_group_chart(filtered_df), use_container_width=True)

        st.plotly_chart(create_scatter_plot(filtered_df), use_container_width=True)

    with tab2:
        st.markdown("<h2 class=\"section-header\">Risk Factor Analysis</h2>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(create_risk_score_distribution_chart(filtered_df), use_container_width=True)
        with col2:
            st.plotly_chart(create_correlation_heatmap(df), use_container_width=True) # Use original df for full correlation

    with tab3:
        st.markdown("<h2 class=\"section-header\">Bayesian Model Performance</h2>", unsafe_allow_html=True)
        st.plotly_chart(create_model_performance_chart(metrics), use_container_width=True)

        st.markdown("### Detailed Metrics")
        st.json(metrics) # Display metrics as JSON for clarity

    with tab4:
        st.markdown("<h2 class=\"section-header\">Predictive Insights</h2>", unsafe_allow_html=True)
        st.info(
            "Use the form below to get a life insurance decision prediction for a new applicant. "
            "The prediction is based on the trained Bayesian model."
        )

        # Input form for prediction
        with st.form(key="prediction_form"):
            st.markdown("### Applicant Details")
            col1, col2 = st.columns(2)
            with col1:
                age = st.number_input("Age", min_value=18, max_value=100, value=30, step=1)
                gender = st.selectbox("Gender", options=["Female", "Male"])
                marital_status = st.selectbox("Marital Status", options=["Single", "Married"])
            with col2:
                dependents = st.selectbox("Dependents", options=["0", "1", "2", ">=3"])
                physical_status = st.selectbox("Physical Status", options=["Sedentary", "Moderately Active", "Active"])
                chronic_diseases = st.selectbox("Chronic Diseases", options=["No Conditions", "Moderate", "Severe"])

            monthly_salary = st.number_input("Monthly Salary (€)", min_value=0, value=2500, step=100)

            submit_button = st.form_submit_button(label="Get Prediction")

        if submit_button:
            # Create input DataFrame for prediction
            input_data = pd.DataFrame({
                "Age": [age],
                "Gender": [gender],
                "MaritalStatus": [marital_status],
                "Dependents": [dependents],
                "PhysicalStatus": [physical_status],
                "ChronicDiseases": [chronic_diseases],
                "MonthlySalary": [monthly_salary]
            })

            # Make prediction
            prediction, probabilities = model._predict_single(input_data.iloc[0])

            st.markdown("### Prediction Result")
            if prediction == "Accept":
                st.success(f"**Predicted Decision: {prediction}**")
            else:
                st.error(f"**Predicted Decision: {prediction}**")

            st.markdown("#### Prediction Probabilities")
            st.json(probabilities)

            # Display risk score for the input data
            input_data_with_risk = calculate_risk_score(input_data.copy())
            st.markdown(f"**Calculated Risk Score:** {input_data_with_risk['RiskScore'].iloc[0]}")

    # Footer
    st.markdown("---")
    st.markdown("Academic team project | 2025")

if __name__ == "__main__":
    main()
