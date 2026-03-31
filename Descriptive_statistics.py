# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 05:24:50 2026

@author: USMAN
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --- App Configuration ---
st.set_page_config(page_title="Descriptive Statistics Analyzer", page_icon="📊", layout="wide")

# --- Custom Functions ---
@st.cache_data
def load_data(file):
    """Loads the uploaded dataset."""
    try:
        name = file.name
        if name.endswith('.csv'):
            return pd.read_csv(file)
        elif name.endswith('.xlsx') or name.endswith('.xls'):
            # Note: Requires 'openpyxl' installed
            return pd.read_excel(file)
    except Exception as e:
        st.error(f"Error loading file: {e}")
    return None

# --- Main UI ---
st.title("📊 Descriptive Statistics Analyzer")
st.markdown("""
Welcome to the Descriptive Stats Analyzer! Upload your dataset to instantly generate summary statistics, distribution plots, and correlation matrices.
""")

# --- Sidebar: File Upload ---
st.sidebar.header("1. Upload Data")
uploaded_file = st.sidebar.file_uploader("Upload your CSV or Excel file", type=['csv', 'xlsx', 'xls'])

if uploaded_file is not None:
    df = load_data(uploaded_file)
    
    if df is not None:
        st.sidebar.success("File successfully uploaded!")
        
        # --- Tabs for organization ---
        tab1, tab2, tab3, tab4 = st.tabs([
            "🗂️ Data Overview", 
            "📈 Summary Statistics", 
            "📊 Distribution Visualizations", 
            "🔗 Correlations"
        ])
        
        # --- Tab 1: Data Overview ---
        with tab1:
            st.header("Data Overview")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Rows", df.shape[0])
            with col2:
                st.metric("Total Columns", df.shape[1])
            
            st.subheader("First 5 Rows")
            st.dataframe(df.head(), use_container_width=True)
            
            st.subheader("Missing Values")
            null_counts = df.isnull().sum()
            if null_counts.sum() > 0:
                missing_data = pd.DataFrame({
                    'Missing Values': null_counts,
                    'Percentage (%)': (null_counts / len(df)) * 100
                }).sort_values(by='Missing Values', ascending=False)
                st.dataframe(missing_data[missing_data['Missing Values'] > 0], use_container_width=True)
            else:
                st.success("No missing values found in the dataset!")

        # --- Tab 2: Summary Statistics ---
        with tab2:
            st.header("Summary Statistics")
            
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
            categorical_cols = df.select_dtypes(exclude=['number']).columns.tolist()
            
            if numeric_cols:
                st.subheader("Numerical Data")
                desc = df[numeric_cols].describe().T
                # Adding Skewness and Kurtosis
                desc['skewness'] = df[numeric_cols].skew()
                desc['kurtosis'] = df[numeric_cols].kurt()
                st.dataframe(desc, use_container_width=True)
            
            if categorical_cols:
                st.subheader("Categorical Data")
                st.dataframe(df[categorical_cols].describe().T, use_container_width=True)

        # --- Tab 3: Distribution Visualizations ---
        with tab3:
            st.header("Distribution Visualizations")
            if numeric_cols:
                selected_col = st.selectbox("Select a numerical column to visualize:", numeric_cols)
                
                col_plot1, col_plot2 = st.columns(2)
                
                with col_plot1:
                    st.subheader("Histogram & KDE")
                    fig, ax = plt.subplots(figsize=(8, 5))
                    sns.histplot(df[selected_col], kde=True, ax=ax, color='skyblue')
                    ax.set_title(f'Distribution of {selected_col}')
                    st.pyplot(fig)
                
                with col_plot2:
                    st.subheader("Boxplot (Outlier Detection)")
                    fig, ax = plt.subplots(figsize=(8, 5))
                    sns.boxplot(x=df[selected_col], ax=ax, color='lightgreen')
                    ax.set_title(f'Boxplot of {selected_col}')
                    st.pyplot(fig)
            else:
                st.warning("No numerical columns found to visualize.")

        # --- Tab 4: Correlations ---
        with tab4:
            st.header("Correlation Analysis")
            if len(numeric_cols) > 1:
                st.markdown("Pearson correlation coefficients between numerical variables.")
                
                fig, ax = plt.subplots(figsize=(10, 8))
                corr_matrix = df[numeric_cols].corr()
                
                sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5, ax=ax)
                st.pyplot(fig)
            else:
                st.warning("Need at least two numerical columns to generate a correlation heatmap.")
                
    else:
        st.error("Error reading file. Please ensure it is a valid CSV or Excel file.")
else:
    st.info("👈 Please upload a dataset from the sidebar to begin analysis.")
