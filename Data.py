import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import io
import plotly.express as px

# Set page configuration
st.set_page_config(
    page_title="Data Analysis Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #0D47A1;
    }
    .info-text {
        color: #424242;
    }
    .highlight {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 0.5rem;
    }
    .footer {
        text-align: center;
        color: #616161;
        font-size: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)

# Title and introduction
st.markdown('<p class="main-header">Interactive Data Analysis Dashboard</p>', unsafe_allow_html=True)
st.markdown("""
This dashboard allows you to upload, explore, visualize, and analyze your data interactively.
Use the sidebar to navigate between different sections and control various parameters.
""")

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Data Upload", "Data Exploration", "Visualization", "Statistical Analysis", "Machine Learning"])

# Initialize session state for storing data between pages
if 'data' not in st.session_state:
    st.session_state.data = None
if 'filename' not in st.session_state:
    st.session_state.filename = None
if 'ml_model' not in st.session_state:
    st.session_state.ml_model = None

#######################
# DATA UPLOAD PAGE
#######################
if page == "Data Upload":
    st.markdown('<p class="sub-header">Upload Your Data</p>', unsafe_allow_html=True)
    
    # Data source selection
    data_source = st.radio("Select data source:", ["Upload CSV/Excel file", "Use sample dataset"])
    
    if data_source == "Upload CSV/Excel file":
        uploaded_file = st.file_uploader("Choose a CSV or Excel file", type=["csv", "xlsx", "xls"])
        
        if uploaded_file is not None:
            try:
                # Determine file type and read accordingly
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file)
                
                st.session_state.data = df
                st.session_state.filename = uploaded_file.name
                
                st.success(f"Successfully loaded {uploaded_file.name} with {df.shape[0]} rows and {df.shape[1]} columns!")
                
                # Display preview
                with st.expander("Preview data"):
                    st.dataframe(df.head(10))
                    
            except Exception as e:
                st.error(f"Error loading file: {e}")
    
    else:  # Use sample dataset
        dataset_choice = st.selectbox(
            "Choose a sample dataset:", 
            ["Iris Flower Dataset", "Boston Housing", "Diabetes", "Wine Quality"]
        )
        
        if st.button("Load Sample Dataset"):
            if dataset_choice == "Iris Flower Dataset":
                df = sns.load_dataset("iris")
            elif dataset_choice == "Boston Housing":
                from sklearn.datasets import load_boston
                boston = load_boston()
                df = pd.DataFrame(boston.data, columns=boston.feature_names)
                df['PRICE'] = boston.target
            elif dataset_choice == "Diabetes":
                from sklearn.datasets import load_diabetes
                diabetes = load_diabetes()
                df = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
                df['TARGET'] = diabetes.target
            elif dataset_choice == "Wine Quality":
                # Create sample wine quality data
                np.random.seed(42)
                n = 1000
                df = pd.DataFrame({
                    'fixed_acidity': np.random.normal(8.3, 1.7, n),
                    'volatile_acidity': np.random.normal(0.5, 0.2, n),
                    'citric_acid': np.random.normal(0.3, 0.1, n),
                    'residual_sugar': np.random.normal(6.4, 5.0, n),
                    'chlorides': np.random.normal(0.05, 0.02, n),
                    'free_sulfur_dioxide': np.random.normal(35, 15, n),
                    'total_sulfur_dioxide': np.random.normal(138, 42, n),
                    'density': np.random.normal(0.994, 0.003, n),
                    'pH': np.random.normal(3.2, 0.2, n),
                    'sulphates': np.random.normal(0.6, 0.1, n),
                    'alcohol': np.random.normal(10.4, 1.0, n),
                    'quality': np.random.randint(3, 9, n)
                })
            
            st.session_state.data = df
            st.session_state.filename = dataset_choice
            
            st.success(f"Successfully loaded {dataset_choice} with {df.shape[0]} rows and {df.shape[1]} columns!")
            
            # Display preview
            with st.expander("Preview data"):
                st.dataframe(df.head(10))
    
    # Data preprocessing options
    if st.session_state.data is not None:
        st.markdown('<p class="sub-header">Data Preprocessing</p>', unsafe_allow_html=True)
        
        with st.expander("Data Cleaning Options"):
            df = st.session_state.data
            
            # Handle missing values
            if df.isnull().sum().sum() > 0:
                st.write("Missing values detected!")
                missing_action = st.selectbox(
                    "How to handle missing values?",
                    ["Do nothing", "Drop rows", "Fill with mean", "Fill with median", "Fill with zero"]
                )
                
                if missing_action != "Do nothing" and st.button("Apply Missing Value Treatment"):
                    if missing_action == "Drop rows":
                        df = df.dropna()
                    elif missing_action == "Fill with mean":
                        df = df.fillna(df.mean(numeric_only=True))
                    elif missing_action == "Fill with median":
                        df = df.fillna(df.median(numeric_only=True))
                    elif missing_action == "Fill with zero":
                        df = df.fillna(0)
                    
                    st.session_state.data = df
                    st.success("Missing values handled successfully!")
            
            # Filter data
            st.write("Filter rows based on conditions:")
            if len(df.columns) > 0:
                col_to_filter = st.selectbox("Select column to filter:", df.columns)
                
                if df[col_to_filter].dtype.kind in 'ifc':  # If numeric column
                    min_val = float(df[col_to_filter].min())
                    max_val = float(df[col_to_filter].max())
                    filter_range = st.slider(
                        f"Range for {col_to_filter}",
                        min_val, max_val, (min_val, max_val)
                    )
                    
                    if st.button("Apply Numeric Filter"):
                        filtered_df = df[(df[col_to_filter] >= filter_range[0]) & 
                                        (df[col_to_filter] <= filter_range[1])]
                        st.session_state.data = filtered_df
                        st.success(f"Data filtered: {filtered_df.shape[0]} rows remaining")
                
                elif df[col_to_filter].dtype == 'object':  # If categorical column
                    categories = df[col_to_filter].unique().tolist()
                    selected_cats = st.multiselect(
                        f"Select values for {col_to_filter}",
                        categories, default=categories
                    )
                    
                    if st.button("Apply Categorical Filter"):
                        filtered_df = df[df[col_to_filter].isin(selected_cats)]
                        st.session_state.data = filtered_df
                        st.success(f"Data filtered: {filtered_df.shape[0]} rows remaining")

#######################
# DATA EXPLORATION PAGE
#######################
elif page == "Data Exploration":
    st.markdown('<p class="sub-header">Data Exploration</p>', unsafe_allow_html=True)
    
    if st.session_state.data is None:
        st.warning("Please upload or select a dataset first!")
    else:
        df = st.session_state.data
        st.write(f"Exploring: **{st.session_state.filename}**")
        
        # Basic info
        with st.expander("Dataset Information"):
            st.write(f"**Shape**: {df.shape[0]} rows, {df.shape[1]} columns")
            
            # Memory usage
            memory_usage = df.memory_usage(deep=True).sum()
            if memory_usage < 1024:
                st.write(f"**Memory usage**: {memory_usage} bytes")
            elif memory_usage < 1024**2:
                st.write(f"**Memory usage**: {memory_usage/1024:.2f} KB")
            else:
                st.write(f"**Memory usage**: {memory_usage/1024**2:.2f} MB")
            
            # Data types
            st.write("**Data Types**:")
            dtype_df = pd.DataFrame(df.dtypes, columns=['Data Type'])
            dtype_df = dtype_df.reset_index()
            dtype_df.columns = ['Column', 'Data Type']
            st.dataframe(dtype_df)
            
            # Missing values
            st.write("**Missing Values**:")
            missing_df = pd.DataFrame(df.isnull().sum(), columns=['Missing Values'])
            missing_df['Percentage'] = (missing_df['Missing Values'] / df.shape[0]) * 100
            missing_df = missing_df.reset_index()
            missing_df.columns = ['Column', 'Missing Values', 'Percentage']
            st.dataframe(missing_df)
        
        # Data view
        with st.expander("View Data", expanded=True):
            # Choose between head, tail, sample or full
            view_option = st.radio("View option:", ["First rows", "Last rows", "Random sample", "Full dataset"])
            num_rows = st.slider("Number of rows:", 5, 100, 10)
            
            if view_option == "First rows":
                st.dataframe(df.head(num_rows))
            elif view_option == "Last rows":
                st.dataframe(df.tail(num_rows))
            elif view_option == "Random sample":
                st.dataframe(df.sample(min(num_rows, len(df))))
            else:
                st.dataframe(df)
        
        # Summary statistics
        with st.expander("Summary Statistics"):
            # Select numeric or categorical columns
            stats_type = st.radio("Statistics type:", ["Numeric", "Categorical"])
            
            if stats_type == "Numeric":
                numeric_cols = df.select_dtypes(include=['number']).columns
                
                if len(numeric_cols) == 0:
                    st.warning("No numeric columns found in the dataset.")
                else:
                    selected_num_cols = st.multiselect("Select numeric columns:", numeric_cols, default=numeric_cols)
                    
                    if selected_num_cols:
                        st.write("**Descriptive Statistics**:")
                        st.dataframe(df[selected_num_cols].describe().T)
                        
                        # Correlation matrix
                        if len(selected_num_cols) > 1:
                            st.write("**Correlation Matrix**:")
                            corr = df[selected_num_cols].corr()
                            
                            # Visualization options for correlation
                            corr_viz = st.radio("Correlation visualization:", ["Table", "Heatmap"])
                            
                            if corr_viz == "Table":
                                st.dataframe(corr)
                            else:
                                fig, ax = plt.subplots(figsize=(10, 8))
                                sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
                                st.pyplot(fig)
            else:
                cat_cols = df.select_dtypes(include=['object', 'category']).columns
                
                if len(cat_cols) == 0:
                    st.warning("No categorical columns found in the dataset.")
                else:
                    selected_cat_cols = st.multiselect("Select categorical columns:", cat_cols, default=cat_cols[:min(3, len(cat_cols))])
                    
                    if selected_cat_cols:
                        for col in selected_cat_cols:
                            st.write(f"**{col} - Value Counts**:")
                            vc = df[col].value_counts().reset_index()
                            vc.columns = [col, 'Count']
                            
                            # Calculate percentage
                            vc['Percentage'] = (vc['Count'] / vc['Count'].sum()) * 100
                            
                            st.dataframe(vc)
                            
                            # Bar chart for distribution
                            fig, ax = plt.subplots(figsize=(10, 5))
                            sns.barplot(x=col, y='Count', data=vc[:10], ax=ax)
                            plt.xticks(rotation=45, ha='right')
                            plt.tight_layout()
                            st.pyplot(fig)
        
        # Column details
        with st.expander("Column Details"):
            selected_column = st.selectbox("Select a column to examine:", df.columns)
            
            if selected_column:
                col_data = df[selected_column]
                st.write(f"**Column**: {selected_column}")
                st.write(f"**Type**: {col_data.dtype}")
                st.write(f"**Unique Values**: {col_data.nunique()}")
                
                if col_data.dtype.kind in 'ifc':  # Numeric column
                    st.write(f"**Min**: {col_data.min()}")
                    st.write(f"**Max**: {col_data.max()}")
                    st.write(f"**Mean**: {col_data.mean()}")
                    st.write(f"**Median**: {col_data.median()}")
                    st.write(f"**Std Dev**: {col_data.std()}")
                    
                    # Histogram
                    fig, ax = plt.subplots(figsize=(10, 6))
                    sns.histplot(col_data.dropna(), kde=True, ax=ax)
                    plt.title(f'Distribution of {selected_column}')
                    st.pyplot(fig)
                    
                    # Boxplot
                    fig, ax = plt.subplots(figsize=(10, 2))
                    sns.boxplot(x=col_data.dropna(), ax=ax)
                    plt.title(f'Boxplot of {selected_column}')
                    st.pyplot(fig)
                else:
                    # For categorical columns
                    top_n = min(10, col_data.nunique())
                    value_counts = col_data.value_counts().head(top_n)
                    
                    fig, ax = plt.subplots(figsize=(10, 6))
                    sns.barplot(x=value_counts.index, y=value_counts.values, ax=ax)
                    plt.title(f'Top {top_n} categories in {selected_column}')
                    plt.xticks(rotation=45, ha='right')
                    plt.tight_layout()
                    st.pyplot(fig)

#######################
# VISUALIZATION PAGE
#######################
elif page == "Visualization":
    st.markdown('<p class="sub-header">Data Visualization</p>', unsafe_allow_html=True)
    
    if st.session_state.data is None:
        st.warning("Please upload or select a dataset first!")
    else:
        df = st.session_state.data
        
        viz_type = st.sidebar.selectbox(
            "Visualization Type",
            ["Scatter Plot", "Line Chart", "Bar Chart", "Histogram", "Box Plot", 
             "Violin Plot", "Pair Plot", "Heatmap", "3D Scatter Plot"]
        )
        
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        categoric_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        if viz_type == "Scatter Plot":
            st.markdown('<p class="highlight">Scatter Plot</p>', unsafe_allow_html=True)
            
            if len(numeric_cols) < 2:
                st.error("Need at least 2 numeric columns for a scatter plot")
            else:
                col1, col2 = st.columns(2)
                
                with col1:
                    x_col = st.selectbox("X-axis", numeric_cols, index=0)
                    y_col = st.selectbox("Y-axis", numeric_cols, index=min(1, len(numeric_cols)-1))
                
                with col2:
                    color_col = st.selectbox("Color by (optional)", ["None"] + df.columns.tolist())
                    size_col = st.selectbox("Size by (optional)", ["None"] + numeric_cols)
                
                # Create plotly scatter plot
                fig = px.scatter(
                    df, x=x_col, y=y_col, 
                    color=None if color_col == "None" else color_col,
                    size=None if size_col == "None" else size_col,
                    hover_data=df.columns,
                    title=f"{y_col} vs {x_col}"
                )
                
                st.plotly_chart(fig, use_container_width=True)
        
        elif viz_type == "Line Chart":
            st.markdown('<p class="highlight">Line Chart</p>', unsafe_allow_html=True)
            
            if len(numeric_cols) < 1:
                st.error("Need at least 1 numeric column for a line chart")
            else:
                x_col = st.selectbox("X-axis", df.columns, index=0)
                y_cols = st.multiselect("Y-axis (multiple)", numeric_cols, default=[numeric_cols[0]])
                
                if not y_cols:
                    st.error("Please select at least one column for Y-axis")
                else:
                    # Option to sort by x
                    sort_by_x = st.checkbox("Sort by X-axis")
                    
                    # Create plot
                    plot_data = df.copy()
                    if sort_by_x:
                        plot_data = plot_data.sort_values(x_col)
                    
                    fig = px.line(
                        plot_data, x=x_col, y=y_cols,
                        title=f"Line Chart of {', '.join(y_cols)} by {x_col}"
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
        
        elif viz_type == "Bar Chart":
            st.markdown('<p class="highlight">Bar Chart</p>', unsafe_allow_html=True)
            
            chart_type = st.radio("Bar Chart Type", ["Simple Bar Chart", "Grouped Bar Chart"])
            
            if chart_type == "Simple Bar Chart":
                x_col = st.selectbox("X-axis (Categories)", df.columns)
                y_col = st.selectbox("Y-axis (Values)", numeric_cols if numeric_cols else ["Count"])
                
                if y_col == "Count":
                    # Count the occurrences of each category
                    data = df[x_col].value_counts().reset_index()
                    data.columns = [x_col, "Count"]
                    fig = px.bar(data, x=x_col, y="Count", title=f"Count of {x_col}")
                else:
                    # Use the actual values
                    fig = px.bar(df, x=x_col, y=y_col, title=f"{y_col} by {x_col}")
                
                st.plotly_chart(fig, use_container_width=True)
            
            else:  # Grouped Bar Chart
                x_col = st.selectbox("X-axis (Primary Category)", df.columns)
                color_col = st.selectbox("Group by", categoric_cols if categoric_cols else ["None"])
                y_col = st.selectbox("Y-axis (Values)", numeric_cols if numeric_cols else ["Count"])
                
                if color_col == "None":
                    st.error("Need a categorical column for grouping")
                else:
                    if y_col == "Count":
                        # Count with grouping
                        fig = px.histogram(
                            df, x=x_col, color=color_col,
                            title=f"Count of {x_col} grouped by {color_col}",
                            barmode="group"
                        )
                    else:
                        # Use the actual values
                        fig = px.bar(
                            df, x=x_col, y=y_col, color=color_col,
                            title=f"{y_col} by {x_col} grouped by {color_col}",
                            barmode="group"
                        )
                    
                    st.plotly_chart(fig, use_container_width=True)
        
        elif viz_type == "Histogram":
            st.markdown('<p class="highlight">Histogram</p>', unsafe_allow_html=True)
            
            if not numeric_cols:
                st.error("No numeric columns available for histogram")
            else:
                col_for_hist = st.selectbox("Select column", numeric_cols)
                n_bins = st.slider("Number of bins", 5, 100, 20)
                
                # Create histogram with Plotly
                fig = px.histogram(
                    df, x=col_for_hist, nbins=n_bins,
                    marginal="box",  # can be 'rug', 'box', 'violin'
                    title=f"Histogram of {col_for_hist}"
                )
                
                st.plotly_chart(fig, use_container_width=True)
        
        elif viz_type == "Box Plot":
            st.markdown('<p class="highlight">Box Plot</p>', unsafe_allow_html=True)
            
            if not numeric_cols:
                st.error("No numeric columns available for box plot")
            else:
                y_col = st.selectbox("Y-axis (numeric)", numeric_cols)
                x_col = st.selectbox("X-axis (categorical, optional)", ["None"] + categoric_cols)
                
                if x_col == "None":
                    fig = px.box(df, y=y_col, title=f"Box Plot of {y_col}")
                else:
                    fig = px.box(df, x=x_col, y=y_col, title=f"Box Plot of {y_col} by {x_col}")
                
                st.plotly_chart(fig, use_container_width=True)
        
        elif viz_type == "Violin Plot":
            st.markdown('<p class="highlight">Violin Plot</p>', unsafe_allow_html=True)
            
            if not numeric_cols:
                st.error("No numeric columns available for violin plot")
            else:
                y_col = st.selectbox("Y-axis (numeric)", numeric_cols)
                x_col = st.selectbox("X-axis (categorical, optional)", ["None"] + categoric_cols)
                
                fig, ax = plt.subplots(figsize=(10, 6))
                
                if x_col == "None":
                    sns.violinplot(y=df[y_col], ax=ax)
                    plt.title(f"Violin Plot of {y_col}")
                else:
                    sns.violinplot(x=df[x_col], y=df[y_col], ax=ax)
                    plt.title(f"Violin Plot of {y_col} by {x_col}")
                    plt.xticks(rotation=45, ha='right')
                
                plt.tight_layout()
                st.pyplot(fig)
        
        elif viz_type == "Pair Plot":
            st.markdown('<p class="highlight">Pair Plot</p>', unsafe_allow_html=True)
            
            if len(numeric_cols) < 2:
                st.error("Need at least 2 numeric columns for a pair plot")
            else:
                selected_cols = st.multiselect(
                    "Select columns for pair plot (2-5 recommended)",
                    numeric_cols,
                    default=numeric_cols[:min(4, len(numeric_cols))]
                )
                
                hue_col = st.selectbox("Color by (optional)", ["None"] + categoric_cols)
                
                if len(selected_cols) < 2:
                    st.error("Please select at least 2 columns")
                else:
                    with st.spinner("Creating pair plot..."):
                        fig = plt.figure(figsize=(12, 8))
                        if hue_col == "None":
                            sns.pairplot(df[selected_cols])
                        else:
                            sns.pairplot(df[selected_cols + [hue_col]], hue=hue_col)
                        
                        st.pyplot(fig)
        
        elif viz_type == "Heatmap":
            st.markdown('<p class="highlight">Correlation Heatmap</p>', unsafe_allow_html=True)
            
            if len(numeric_cols) < 2:
                st.error("Need at least 2 numeric columns for a heatmap")
            else:
                selected_cols = st.multiselect(
                    "Select columns for correlation heatmap",
                    numeric_cols,
                    default=numeric_cols[:min(6, len(numeric_cols))]
                )
                
                if len(selected_cols) < 2:
                    st.error("Please select at least 2 columns")
                else:
                    corr = df[selected_cols].corr()
                    mask = np.triu(np.ones_like(corr, dtype=bool))
                    
                    fig, ax = plt.subplots(figsize=(10, 8))
                    sns.heatmap(corr, mask=mask, annot=True, cmap='coolwarm', 
                                center=0, square=True, linewidths=1, ax=ax)
                    plt.title('Correlation Heatmap')
                    st.pyplot(fig)
        
        elif viz_type == "3D Scatter Plot":
            st.markdown('<p class="highlight">3D Scatter Plot</p>', unsafe_allow_html=True)
            
            if len(numeric_cols) < 3:
                st.error("Need at least 3 numeric columns for a 3D scatter plot")
            else:
                col1, col2 = st.columns(2)
                
                with col1:
                    x_col = st.selectbox("X-axis", numeric_cols, index=0)
                    y_col = st.selectbox("Y-axis", numeric_cols, index=min(1, len(numeric_cols)-1))
                    z_col = st.selectbox("Z-axis", numeric_cols, index=min(2, len(numeric_cols)-1))
                
                with col2:
                    color_col = st.selectbox("Color by (optional)", ["None"] + df.columns.tolist())
                    size_col = st.selectbox("Size by (optional)", ["None"] + numeric_cols)
                
                fig = px.scatter_3d(
                    df, x=x_col, y=y_col, z=z_col,
                    color=None if color_col == "None" else color_col,
                    size=None if size_col == "None" else size_col,
                    opacity=0.7,
                    title=f"3D Scatter Plot: {x_col}, {y_col}, {z_col}"
                )
                
                # Improve layout
                fig.update_layout(
                    scene = dict(
                        xaxis_title=x_col,
                        yaxis_title=y_col,
                        zaxis_title=z_col
                    )
                )
                
                st.plotly_chart(fig, use_container_width=True)

#######################
# STATISTICAL ANALYSIS PAGE
#######################
elif page == "Statistical Analysis":
    st.markdown('<p class="sub-header">Statistical Analysis</p>', unsafe_allow_html=True)
    
    if st.session_state.data is None:
        st.warning("Please upload or select a dataset first!")
    else:
        df = st.session_state.data
        
        analysis_type = st.sidebar.selectbox(
            "Analysis Type",
            ["Descriptive Statistics", "Hypothesis Testing", "Regression Analysis", "Time Series Analysis"]
        )
        
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        categoric_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        if analysis_type == "Descriptive Statistics":
            st.markdown('<p class="highlight">Descriptive Statistics</p>', unsafe_allow_html=True)
            
            if not numeric_cols:
                st.error("No numeric columns available for analysis")
            else:
                selected_cols = st.multiselect(
                    "Select columns for analysis",
                    numeric_cols,
                    default=numeric_cols[:min(5, len(numeric_cols))]
                )
                
                if selected_cols:
                    st.write("**Basic Statistics**")
                    st.dataframe(df[selected_cols].describe().T)
                    
                    # Additional statistics
                    stats_df = pd.DataFrame(index=selected_cols)
                    stats_df['Skewness'] = [df[col].skew() for col in selected_cols]
                    stats_df['Kurtosis'] = [df[col].kurtosis() for col in selected_cols]
                    stats_df['IQR'] = [df[col].quantile(0.75) - df[col].quantile(0.25) for col in selected_cols]
                    
                    st.write("**Additional Statistics**")
                    st.dataframe(stats_df)
                    
                    # Visualization of distributions
                    for col in selected_cols:
                        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))
                        
                        # Histogram with KDE
                        sns.histplot(df[col].dropna(), kde=True, ax=ax1)
                        ax1.set_title(f"Distribution of {col}")
                        
                        # QQ Plot
                        from scipy import stats
                        stats.probplot(df[col].dropna(), plot=ax2)
                        ax2.set_title(f"Q-Q Plot of {col}")
                        
                        plt.tight_layout()
                        st.pyplot(fig)
        
        elif analysis_type == "Hypothesis Testing":
            st.markdown('<p class="highlight">Hypothesis Testing</p>', unsafe_allow_html=True)
            
            test_type = st.selectbox(
                "Select test type",
                ["One-sample t-test", "Two-sample t-test", "ANOVA", "Chi-square test", 
                 "Correlation test"]
            )
            
            if test_type == "One-sample t-test":
                if not numeric_cols:
                    st.error("No numeric columns available for t-test")
                else:
                    col = st.selectbox("Select column", numeric_cols)
                    mu = st.number_input("Hypothesized mean (μ₀)", value=float(df[col].mean()))
                    
                    from scipy import stats
                    
                    # Remove NaN values
                    data = df[col].dropna()
                    
                    # Perform t-test
                    t_stat, p_value = stats.ttest_1samp(data, mu)
                    
                    # Show results
                    st.write(f"**One-sample t-test for {col}**")
                    st.write(f"Null hypothesis (H₀): μ = {mu}")
                    st.write(f"Alternative hypothesis (H₁): μ ≠ {mu}")
                    
                    results_df = pd.DataFrame({
                        'Sample Mean': [data.mean()],
                        'Sample Std': [data.std()],
                        'Sample Size': [len(data)],
                        't-statistic': [t_stat],
                        'p-value': [p_value]
                    })
                    
                    st.dataframe(results_df)
                    
                    # Interpretation
                    alpha = 0.05
                    st.write(f"**Interpretation** (α = {alpha}):")
                    if p_value < alpha:
                        st.write(f"p-value ({p_value:.4f}) < α ({alpha})")
                        st.write("We reject the null hypothesis. There is significant evidence that the population mean is different from the hypothesized value.")
                    else:
                        st.write(f"p-value ({p_value:.4f}) ≥ α ({alpha})")
                        st.write("We fail to reject the null hypothesis. There is not enough evidence that the population mean is different from the hypothesized value.")
            
            elif test_type == "Two-sample t-test":
                if not numeric_cols:
                    st.error("No numeric columns available for t-test")
                else:
                    # Option 1: Use a numeric column and split by a categorical variable
                    # Option 2: Compare two numeric columns directly
                    comparison_type = st.radio(
                        "Comparison type",
                        ["Split one numeric column by category", "Compare two numeric columns"]
                    )
                    
                    if comparison_type == "Split one numeric column by category":
                        if not categoric_cols:
                            st.error("No categorical columns available for grouping")
                        else:
                            num_col = st.selectbox("Select numeric column", numeric_cols)
                            cat_col = st.selectbox("Select category for grouping", categoric_cols)
                            
                            # Get unique categories (limit to first 2 if more exist)
                            categories = df[cat_col].unique()
                            if len(categories) < 2:
                                st.error(f"Need at least 2 categories in {cat_col} for comparison")
                            else:
                                if len(categories) > 2:
                                    st.warning(f"More than 2 categories found in {cat_col}. Using the first 2 for t-test.")
                                
                                group1 = st.selectbox("Select first group", categories)
                                remaining = [c for c in categories if c != group1]
                                group2 = st.selectbox("Select second group", remaining)
                                
                                # Get the data for each group
                                data1 = df[df[cat_col] == group1][num_col].dropna()
                                data2 = df[df[cat_col] == group2][num_col].dropna()
                                
                                # Perform t-test
                                from scipy import stats
                                equal_var = st.checkbox("Assume equal variance", value=True)
                                t_stat, p_value = stats.ttest_ind(data1, data2, equal_var=equal_var)
                                
                                # Show results
                                test_name = "Independent samples t-test" if equal_var else "Welch's t-test"
                                st.write(f"**{test_name}**")
                                st.write(f"Comparing {num_col} between {group1} (n={len(data1)}) and {group2} (n={len(data2)})")
                                st.write(f"Null hypothesis (H₀): μ₁ = μ₂")
                                st.write(f"Alternative hypothesis (H₁): μ₁ ≠ μ₂")
                                
                                results_df = pd.DataFrame({
                                    'Group': [group1, group2],
                                    'Mean': [data1.mean(), data2.mean()],
                                    'Std': [data1.std(), data2.std()],
                                    'n': [len(data1), len(data2)]
                                })
                                
                                st.dataframe(results_df)
                                
                                st.write(f"**t-statistic**: {t_stat:.4f}")
                                st.write(f"**p-value**: {p_value:.4f}")
                                
                                # Interpretation
                                alpha = 0.05
                                st.write(f"**Interpretation** (α = {alpha}):")
                                if p_value < alpha:
                                    st.write(f"p-value ({p_value:.4f}) < α ({alpha})")
                                    st.write("We reject the null hypothesis. There is significant evidence of a difference in means between the two groups.")
                                else:
                                    st.write(f"p-value ({p_value:.4f}) ≥ α ({alpha})")
                                    st.write("We fail to reject the null hypothesis. There is not enough evidence of a difference in means between the two groups.")
                                
                                # Visualization of group comparison
                                fig, ax = plt.subplots(figsize=(10, 6))
                                sns.boxplot(x=df[cat_col], y=df[num_col], order=[group1, group2], ax=ax)
                                plt.title(f"Comparison of {num_col} by {cat_col}")
                                st.pyplot(fig)
                    
                    else:  # Compare two numeric columns
                        if len(numeric_cols) < 2:
                            st.error("Need at least 2 numeric columns for comparison")
                        else:
                            col1 = st.selectbox("Select first column", numeric_cols, index=0)
                            col2 = st.selectbox("Select second column", numeric_cols, index=min(1, len(numeric_cols)-1))
                            
                            # Get the data
                            data1 = df[col1].dropna()
                            data2 = df[col2].dropna()
                            
                            # Perform paired or independent t-test
                            test_variant = st.radio(
                                "Test type",
                                ["Paired t-test (same subjects)", "Independent t-test (different subjects)"]
                            )
                            
                            from scipy import stats
                            
                            if test_variant == "Paired t-test (same subjects)":
                                # Check if lengths match
                                if len(data1) != len(data2):
                                    st.error("For paired t-test, both columns must have same number of non-NaN values")
                                else:
                                    # Use only rows where both values are present
                                    valid_mask = df[[col1, col2]].notna().all(axis=1)
                                    paired_data1 = df.loc[valid_mask, col1]
                                    paired_data2 = df.loc[valid_mask, col2]
                                    
                                    t_stat, p_value = stats.ttest_rel(paired_data1, paired_data2)
                                    
                                    st.write(f"**Paired samples t-test**")
                                    st.write(f"Comparing {col1} and {col2} (n={len(paired_data1)})")
                            else:
                                equal_var = st.checkbox("Assume equal variance", value=True)
                                t_stat, p_value = stats.ttest_ind(data1, data2, equal_var=equal_var)
                                
                                test_name = "Independent samples t-test" if equal_var else "Welch's t-test"
                                st.write(f"**{test_name}**")
                                st.write(f"Comparing {col1} (n={len(data1)}) and {col2} (n={len(data2)})")
                            
                            st.write(f"Null hypothesis (H₀): μ₁ = μ₂")
                            st.write(f"Alternative hypothesis (H₁): μ₁ ≠ μ₂")
                            
                            results_df = pd.DataFrame({
                                'Variable': [col1, col2],
                                'Mean': [data1.mean(), data2.mean()],
                                'Std': [data1.std(), data2.std()],
                                'n': [len(data1), len(data2)]
                            })
                            
                            st.dataframe(results_df)
                            
                            st.write(f"**t-statistic**: {t_stat:.4f}")
                            st.write(f"**p-value**: {p_value:.4f}")
                            
                            # Interpretation
                            alpha = 0.05
                            st.write(f"**Interpretation** (α = {alpha}):")
                            if p_value < alpha:
                                st.write(f"p-value ({p_value:.4f}) < α ({alpha})")
                                st.write("We reject the null hypothesis. There is significant evidence of a difference between the variables.")
                            else:
                                st.write(f"p-value ({p_value:.4f}) ≥ α ({alpha})")
                                st.write("We fail to reject the null hypothesis. There is not enough evidence of a difference between the variables.")
                            
                            # Visualization
                            fig, ax = plt.subplots(figsize=(10, 6))
                            sns.boxplot(data=df[[col1, col2]].melt(), x='variable', y='value', ax=ax)
                            plt.title(f"Comparison of {col1} and {col2}")
                            st.pyplot(fig)
            
            elif test_type == "ANOVA":
                if not numeric_cols:
                    st.error("No numeric columns available for ANOVA")
                elif not categoric_cols:
                    st.error("No categorical columns available for grouping in ANOVA")
                else:
                    num_col = st.selectbox("Select numeric column (dependent variable)", numeric_cols)
                    cat_col = st.selectbox("Select categorical column (groups)", categoric_cols)
                    
                    # Get unique categories
                    categories = df[cat_col].unique()
                    
                    if len(categories) < 3:
                        st.warning("ANOVA is typically used for 3+ groups. Consider using t-test for 2 groups.")
                    
                    # Get the data for each group
                    groups = []
                    group_names = []
                    
                    for category in categories:
                        group_data = df[df[cat_col] == category][num_col].dropna().values
                        if len(group_data) > 0:
                            groups.append(group_data)
                            group_names.append(str(category))
                    
                    if len(groups) < 2:
                        st.error("Not enough valid groups for ANOVA")
                    else:
                        # Perform one-way ANOVA
                        from scipy import stats
                        f_stat, p_value = stats.f_oneway(*groups)
                        
                        # Show results
                        st.write(f"**One-way ANOVA**")
                        st.write(f"Testing if {num_col} means differ across groups in {cat_col}")
                        st.write(f"Null hypothesis (H₀): All group means are equal")
                        st.write(f"Alternative hypothesis (H₁): At least one group mean is different")
                        
                        # Summary statistics by group
                        group_summary = df.groupby(cat_col)[num_col].agg(['count', 'mean', 'std']).reset_index()
                        st.dataframe(group_summary)
                        
                        st.write(f"**F-statistic**: {f_stat:.4f}")
                        st.write(f"**p-value**: {p_value:.4f}")
                        
                        # Interpretation
                        alpha = 0.05
                        st.write(f"**Interpretation** (α = {alpha}):")
                        if p_value < alpha:
                            st.write(f"p-value ({p_value:.4f}) < α ({alpha})")
                            st.write("We reject the null hypothesis. There is significant evidence that at least one group mean is different.")
                            
                            # Post-hoc test (Tukey's HSD)
                            st.write("**Post-hoc analysis: Tukey's HSD test**")
                            try:
                                from statsmodels.stats.multicomp import pairwise_tukeyhsd
                                
                                # Prepare data for Tukey's test
                                data = df[[cat_col, num_col]].dropna()
                                tukey = pairwise_tukeyhsd(
                                    data[num_col], data[cat_col],
                                    alpha=alpha
                                )
                                
                                # Display results
                                result_df = pd.DataFrame(
                                    data=tukey._results_table.data[1:],
                                    columns=tukey._results_table.data[0]
                                )
                                st.dataframe(result_df)
                                
                            except ImportError:
                                st.warning("statsmodels library not available for Tukey's test")
                        else:
                            st.write(f"p-value ({p_value:.4f}) ≥ α ({alpha})")
                            st.write("We fail to reject the null hypothesis. There is not enough evidence that the group means differ.")
                        
                        # Visualization
                        fig, ax = plt.subplots(figsize=(10, 6))
                        sns.boxplot(x=df[cat_col], y=df[num_col], ax=ax)
                        plt.title(f"Distribution of {num_col} across {cat_col} groups")
                        plt.xticks(rotation=45, ha='right')
                        plt.tight_layout()
                        st.pyplot(fig)
            
            elif test_type == "Chi-square test":
                if not categoric_cols or len(categoric_cols) < 2:
                    st.error("Need at least 2 categorical columns for chi-square test")
                else:
                    test_subtype = st.radio(
                        "Chi-square test type",
                        ["Independence test (two categorical variables)", 
                         "Goodness of fit test (one categorical variable)"]
                    )
                    
                    if test_subtype == "Independence test (two categorical variables)":
                        col1 = st.selectbox("Select first categorical variable", categoric_cols, index=0)
                        col2 = st.selectbox("Select second categorical variable", categoric_cols, index=min(1, len(categoric_cols)-1))
                        
                        # Create contingency table
                        contingency = pd.crosstab(df[col1], df[col2])
                        
                        # Display crosstab
                        st.write("**Contingency Table** (observed frequencies)")
                        st.dataframe(contingency)
                        
                        # Perform chi-square test
                        from scipy import stats
                        chi2, p, dof, expected = stats.chi2_contingency(contingency)
                        
                        # Show expected frequencies
                        st.write("**Expected Frequencies** (if variables were independent)")
                        expected_df = pd.DataFrame(
                            expected, 
                            index=contingency.index,
                            columns=contingency.columns
                        )
                        st.dataframe(expected_df)
                        
                        # Show test results
                        st.write(f"**Chi-square Test of Independence**")
                        st.write(f"Testing association between {col1} and {col2}")
                        st.write(f"Null hypothesis (H₀): {col1} and {col2} are independent")
                        st.write(f"Alternative hypothesis (H₁): {col1} and {col2} are not independent (are associated)")
                        
                        results_df = pd.DataFrame({
                            'Chi-square statistic': [chi2],
                            'Degrees of freedom': [dof],
                            'p-value': [p]
                        })
                        
                        st.dataframe(results_df)
                        
                        # Interpretation
                        alpha = 0.05
                        st.write(f"**Interpretation** (α = {alpha}):")
                        if p < alpha:
                            st.write(f"p-value ({p:.4f}) < α ({alpha})")
                            st.write(f"We reject the null hypothesis. There is significant evidence of an association between {col1} and {col2}.")
                        else:
                            st.write(f"p-value ({p:.4f}) ≥ α ({alpha})")
                            st.write(f"We fail to reject the null hypothesis. There is not enough evidence of an association between {col1} and {col2}.")
                        
                        # Visualization
                        fig, ax = plt.subplots(figsize=(10, 8))
                        sns.heatmap(contingency, annot=True, fmt='d', cmap='YlGnBu', ax=ax)
                        plt.title(f"Contingency table: {col1} vs {col2}")
                        plt.tight_layout()
                        st.pyplot(fig)
                    
                    else:  # Goodness of fit test
                        col = st.selectbox("Select categorical variable", categoric_cols)
                        
                        # Get observed frequencies
                        observed = df[col].value_counts().sort_index()
                        categories = observed.index.tolist()
                        
                        st.write("**Observed Frequencies**")
                        st.dataframe(observed.reset_index().rename(columns={'index': col, col: 'Frequency'}))
                        
                        # Specify expected probabilities
                        st.write("**Expected Probabilities**")
                        st.write("Enter the expected probability for each category (must sum to 1)")
                        
                        total_prob = 0
                        expected_probs = []
                        
                        for i, category in enumerate(categories):
                            if i == len(categories) - 1:
                                # Auto-calculate the last probability to ensure sum to 1
                                prob = 1 - total_prob
                                st.write(f"{category}: {prob:.4f} (auto-calculated)")
                            else:
                                max_prob = 1 - total_prob - (len(categories) - i - 1) * 0.01
                                prob = st.slider(
                                    f"Probability for {category}",
                                    0.0, max_prob, 1/len(categories),
                                    format="%.4f"
                                )
                                total_prob += prob
                            
                            expected_probs.append(prob)
                        
                        # Calculate expected frequencies
                        n = len(df[col].dropna())
                        expected = [p * n for p in expected_probs]
                        
                        # Create a dataframe for comparison
                        comparison_df = pd.DataFrame({
                            'Category': categories,
                            'Observed': observed.values,
                            'Expected': expected,
                            'Expected %': [f"{p:.2%}" for p in expected_probs]
                        })
                        
                        st.write("**Observed vs Expected**")
                        st.dataframe(comparison_df)
                        
                        # Perform chi-square test
                        from scipy import stats
                        chi2, p = stats.chisquare(observed, expected)
                        
                        # Show test results
                        st.write(f"**Chi-square Goodness of Fit Test**")
                        st.write(f"Testing if the distribution of {col} matches the expected distribution")
                        st.write(f"Null hypothesis (H₀): The observed distribution matches the expected distribution")
                        st.write(f"Alternative hypothesis (H₁): The observed distribution differs from the expected distribution")
                        
                        results_df = pd.DataFrame({
                            'Chi-square statistic': [chi2],
                            'Degrees of freedom': [len(categories) - 1],
                            'p-value': [p]
                        })
                        
                        st.dataframe(results_df)
                        
                        # Interpretation
                        alpha = 0.05
                        st.write(f"**Interpretation** (α = {alpha}):")
                        if p < alpha:
                            st.write(f"p-value ({p:.4f}) < α ({alpha})")
                            st.write("We reject the null hypothesis. There is significant evidence that the observed distribution differs from the expected distribution.")
                        else:
                            st.write(f"p-value ({p:.4f}) ≥ α ({alpha})")
                            st.write("We fail to reject the null hypothesis. There is not enough evidence that the observed distribution differs from the expected distribution.")
                        
                        # Visualization
                        fig, ax = plt.subplots(figsize=(10, 6))
                        
                        x = range(len(categories))
                        width = 0.35
                        
                        ax.bar([i - width/2 for i in x], observed, width, label='Observed')
                        ax.bar([i + width/2 for i in x], expected, width, label='Expected')
                        
                        ax.set_xticks(x)
                        ax.set_xticklabels(categories)
                        ax.set_title(f"Goodness of Fit: {col}")
                        ax.legend()
                        
                        plt.tight_layout()
                        st.pyplot(fig)
            
            elif test_type == "Correlation test":
                if len(numeric_cols) < 2:
                    st.error("Need at least 2 numeric columns for correlation test")
                else:
                    col1 = st.selectbox("Select first numeric variable", numeric_cols, index=0)
                    col2 = st.selectbox("Select second numeric variable", numeric_cols, index=min(1, len(numeric_cols)-1))
                    
                    # Select correlation method
                    corr_method = st.selectbox(
                        "Correlation method",
                        ["Pearson (linear relationship)",
                         "Spearman (monotonic relationship)",
                         "Kendall (rank correlation)"]
                    )
                    
                    # Get method for scipy
                    method_dict = {
                        "Pearson (linear relationship)": "pearson",
                        "Spearman (monotonic relationship)": "spearman",
                        "Kendall (rank correlation)": "kendall"
                    }
                    
                    method = method_dict[corr_method]
                    
                    # Compute correlation
                    from scipy import stats
                    
                    # Get data, removing rows with NaN in either column
                    mask = df[[col1, col2]].notna().all(axis=1)
                    data1 = df.loc[mask, col1]
                    data2 = df.loc[mask, col2]
                    
                    if method == "pearson":
                        corr, p_value = stats.pearsonr(data1, data2)
                    elif method == "spearman":
                        corr, p_value = stats.spearmanr(data1, data2)
                    else:  # kendall
                        corr, p_value = stats.kendalltau(data1, data2)
                    
                    # Show results
                    st.write(f"**{corr_method.split(' ')[0]} Correlation Test**")
                    st.write(f"Testing correlation between {col1} and {col2}")
                    st.write(f"Null hypothesis (H₀): No correlation (ρ = 0)")
                    st.write(f"Alternative hypothesis (H₁): Correlation exists (ρ ≠ 0)")
                    
                    results_df = pd.DataFrame({
                        'Correlation coefficient': [corr],
                        'p-value': [p_value],
                        'Sample size': [len(data1)]
                    })
                    
                    st.dataframe(results_df)
                    
                    # Interpretation
                    alpha = 0.05
                    st.write(f"**Interpretation** (α = {alpha}):")
                    
                    if p_value < alpha:
                        st.write(f"p-value ({p_value:.4f}) < α ({alpha})")
                        st.write(f"We reject the null hypothesis. There is significant evidence of a {corr_method.lower().split(' ')[0]} correlation between {col1} and {col2}.")
                        
                        # Interpret strength
                        abs_corr = abs(corr)
                        if abs_corr < 0.3:
                            strength = "weak"
                        elif abs_corr < 0.7:
                            strength = "moderate"
                        else:
                            strength = "strong"
                        
                        direction = "positive" if corr > 0 else "negative"
                        st.write(f"The correlation is {direction} and {strength} (r = {corr:.4f}).")
                    else:
                        st.write(f"p-value ({p_value:.4f}) ≥ α ({alpha})")
                        st.write(f"We fail to reject the null hypothesis. There is not enough evidence of a correlation between {col1} and {col2}.")
                    
                    # Visualization
                    fig, ax = plt.subplots(figsize=(10, 6))
                    
                    # Scatter plot with regression line
                    sns.regplot(x=data1, y=data2, ax=ax)
                    
                    plt.title(f"{corr_method.split(' ')[0]} correlation: r = {corr:.4f}, p = {p_value:.4f}")
                    plt.xlabel(col1)
                    plt.ylabel(col2)
                    
                    st.pyplot(fig)
        
        elif analysis_type == "Regression Analysis":
            st.markdown('<p class="highlight">Regression Analysis</p>', unsafe_allow_html=True)
            
            if len(numeric_cols) < 2:
                st.error("Need at least 2 numeric columns for regression analysis")
            else:
                regression_type = st.selectbox(
                    "Regression type",
                    ["Linear Regression", "Multiple Linear Regression", 
                     "Polynomial Regression", "Logistic Regression"]
                )
                
                if regression_type == "Linear Regression":
                    x_col = st.selectbox("Select independent variable (X)", numeric_cols, index=0)
                    y_col = st.selectbox("Select dependent variable (Y)", numeric_cols, index=min(1, len(numeric_cols)-1))
                    
                    # Check for missing values
                    mask = df[[x_col, y_col]].notna().all(axis=1)
                    X = df.loc[mask, x_col].values.reshape(-1, 1)
                    y = df.loc[mask, y_col].values
                    
                    # Train linear regression model
                    from sklearn.linear_model import LinearRegression
                    from sklearn.metrics import mean_squared_error, r2_score
                    
                    model = LinearRegression()
                    model.fit(X, y)
                    
                    # Make predictions
                    y_pred = model.predict(X)
                    
                    # Calculate metrics
                    mse = mean_squared_error(y, y_pred)
                    rmse = np.sqrt(mse)
                    r2 = r2_score(y, y_pred)
                    
                    # Show model results
                    st.write("**Simple Linear Regression**")
                    st.write(f"Model: {y_col} = {model.intercept_:.4f} + {model.coef_[0]:.4f} × {x_col}")
                    
                    results_df = pd.DataFrame({
                        'Coefficient': [model.coef_[0]],
                        'Intercept': [model.intercept_],
                        'R²': [r2],
                        'RMSE': [rmse],
                        'MSE': [mse],
                        'Sample Size': [len(X)]
                    })
                    
                    st.dataframe(results_df)
                    
                    # Interpretation
                    st.write("**Interpretation**:")
                    st.write(f"- For each unit increase in {x_col}, {y_col} changes by {model.coef_[0]:.4f} units")
                    st.write(f"- The model explains {r2:.2%} of the variance in {y_col}")
                    
                    # Create scatter plot with regression line
                    fig, ax = plt.subplots(figsize=(10, 6))
                    
                    ax.scatter(X, y, alpha=0.7)
                    ax.plot(X, y_pred, color='red', linewidth=2)
                    
                    plt.title(f'Linear Regression: {y_col} vs {x_col}')
                    plt.xlabel(x_col)
                    plt.ylabel(y_col)
                    
                    st.pyplot(fig)
                    
                    # Residual plot
                    residuals = y - y_pred
                    
                    fig, ax = plt.subplots(figsize=(10, 6))
                    
                    ax.scatter(y_pred, residuals, alpha=0.7)
                    ax.axhline(y=0, color='red', linestyle='--')
                    
                    plt.title('Residual Plot')
                    plt.xlabel('Predicted Values')
                    plt.ylabel('Residuals')
                    
                    st.pyplot(fig)
                
                elif regression_type == "Multiple Linear Regression":
                    # Select feature columns and target column
                    target_col = st.selectbox("Select dependent variable (Y)", numeric_cols)
                    potential_features = [col for col in numeric_cols if col != target_col]
                    
                    if len(potential_features) < 1:
                        st.error("Need at least one feature for multiple regression")
                    else:
                        selected_features = st.multiselect(
                            "Select independent variables (X)",
                            potential_features,
                            default=potential_features[:min(3, len(potential_features))]
                        )
                        
                        if not selected_features:
                            st.error("Please select at least one feature")
                        else:
                            # Prepare data
                            all_cols = selected_features + [target_col]
                            mask = df[all_cols].notna().all(axis=1)
                            X = df.loc[mask, selected_features]
                            y = df.loc[mask, target_col]
                            
                            # Train model
                            from sklearn.linear_model import LinearRegression
                            from sklearn.metrics import mean_squared_error, r2_score
                            
                            model = LinearRegression()
                            model.fit(X, y)
                            
                            # Make predictions
                            y_pred = model.predict(X)
                            
                            # Calculate metrics
                            mse = mean_squared_error(y, y_pred)
                            rmse = np.sqrt(mse)
                            r2 = r2_score(y, y_pred)
                            adj_r2 = 1 - (1 - r2) * (len(y) - 1) / (len(y) - len(selected_features) - 1)
                            
                            # Show model formula
                            st.write("**Multiple Linear Regression**")
                            
                            formula = f"{target_col} = {model.intercept_:.4f}"
                            for i, feature in enumerate(selected_features):
                                coef = model.coef_[i]
                                sign = "+" if coef >= 0 else ""
                                formula += f" {sign} {coef:.4f} × {feature}"
                            
                            st.write(f"Model: {formula}")
                            
                            # Show model summary
                            coef_df = pd.DataFrame({
                                'Feature': selected_features,
                                'Coefficient': model.coef_
                            })
                            
                            st.write("**Coefficients**")
                            st.dataframe(coef_df)
                            
                            metrics_df = pd.DataFrame({
                                'Metric': ['Intercept', 'R²', 'Adjusted R²', 'RMSE', 'MSE', 'Sample Size'],
                                'Value': [model.intercept_, r2, adj_r2, rmse, mse, len(X)]
                            })
                            
                            st.write("**Model Metrics**")
                            st.dataframe(metrics_df)
                            
                            # Interpretation
                            st.write("**Interpretation**:")
                            for feature, coef in zip(selected_features, model.coef_):
                                st.write(f"- Holding other variables constant, for each unit increase in {feature}, {target_col} changes by {coef:.4f} units")
                            
                            st.write(f"- The model explains {r2:.2%} of the variance in {target_col}")
                            
                            # Residual plots
                            residuals = y - y_pred
                            
                            fig, ax = plt.subplots(figsize=(10, 6))
                            
                            ax.scatter(y_pred, residuals, alpha=0.7)
                            ax.axhline(y=0, color='red', linestyle='--')
                            
                            plt.title('Residual Plot')
                            plt.xlabel('Predicted Values')
                            plt.ylabel('Residuals')
                            
                            st.pyplot(fig)
                            
                            # Predicted vs Actual
                            fig, ax = plt.subplots(figsize=(10, 6))
                            
                            ax.scatter(y, y_pred, alpha=0.7)
                            
                            # Add perfect prediction line
                            min_val = min(y.min(), y_pred.min())
                            max_val = max(y.max(), y_pred.max())
                            ax.plot([min_val, max_val], [min_val, max_val], 'r--')
                            
                            plt.title('Predicted vs Actual')
                            plt.xlabel('Actual Values')
                            plt.ylabel('Predicted Values')
                            
                            st.pyplot(fig)
                
                elif regression_type == "Polynomial Regression":
                    x_col = st.selectbox("Select independent variable (X)", numeric_cols, index=0)
                    y_col = st.selectbox("Select dependent variable (Y)", numeric_cols, index=min(1, len(numeric_cols)-1))
                    
                    # Degree of polynomial
                    degree = st.slider("Polynomial degree", 1, 10, 2)
                    
                    # Check for missing values
                    mask = df[[x_col, y_col]].notna().all(axis=1)
                    X = df.loc[mask, x_col].values.reshape(-1, 1)
                    y = df.loc[mask, y_col].values
                    
                    # Train polynomial regression model
                    from sklearn.preprocessing import PolynomialFeatures
                    from sklearn.linear_model import LinearRegression
                    from sklearn.pipeline import make_pipeline
                    from sklearn.metrics import mean_squared_error, r2_score
                    
                    model = make_pipeline(
                        PolynomialFeatures(degree=degree),
                        LinearRegression()
                    )
                    
                    model.fit(X, y)
                    
                    # Make predictions
                    y_pred = model.predict(X)
                    
                    # Calculate metrics
                    mse = mean_squared_error(y, y_pred)
                    rmse = np.sqrt(mse)
                    r2 = r2_score(y, y_pred)
                    
                    # Show results
                    st.write(f"**Polynomial Regression (degree {degree})**")
                    
                    results_df = pd.DataFrame({
                        'R²': [r2],
                        'RMSE': [rmse],
                        'MSE': [mse],
                        'Sample Size': [len(X)]
                    })
                    
                    st.dataframe(results_df)
                    
                    # Get the coefficients
                    linear_regressor = model.steps[1][1]
                    poly_features = model.steps[0][1]
                    
                    # Generate feature names
                    feature_names = poly_features.get_feature_names_out([x_col])
                    
                    coef_df = pd.DataFrame({
                        'Term': feature_names,
                        'Coefficient': linear_regressor.coef_
                    })
                    
                    st.write("**Polynomial Coefficients**")
                    st.dataframe(coef_df)
                    
                    # Interpretation
                    st.write("**Interpretation**:")
                    st.write(f"- The model explains {r2:.2%} of the variance in {y_col}")
                    
                    # Visualization
                    # Sorting for better visualization
                    X_sorted = np.sort(X, axis=0)
                    y_pred_sorted = model.predict(X_sorted)
                    
                    fig, ax = plt.subplots(figsize=(10, 6))
                    
                    ax.scatter(X, y, alpha=0.7)
                    ax.plot(X_sorted, y_pred_sorted, color='red', linewidth=2)
                    
                    plt.title(f'Polynomial Regression (degree {degree}): {y_col} vs {x_col}')
                    plt.xlabel(x_col)
                    plt.ylabel(y_col)
                    
                    st.pyplot(fig)
                    
                    # Residual plot
                    residuals = y - y_pred
                    
                    fig, ax = plt.subplots(figsize=(10, 6))
                    
                    ax.scatter(y_pred, residuals, alpha=0.7)
                    ax.axhline(y=0, color='red', linestyle='--')
                    
                    plt.title('Residual Plot')
                    plt.xlabel('Predicted Values')
                    plt.ylabel('Residuals')
                    
                    st.pyplot(fig)
                
                elif regression_type == "Logistic Regression":
                    # Select target (must be binary)
                    all_cols = numeric_cols + categoric_cols
                    target_col = st.selectbox("Select binary target variable", all_cols)
                    
                    # Check if the target is binary
                    unique_values = df[target_col].dropna().nunique()
                    
                    if unique_values != 2:
                        st.error(f"The target variable must be binary (has {unique_values} unique values)")
                    else:
                        # For categorical targets, map to 0/1
                        if df[target_col].dtype == 'object' or df[target_col].dtype == 'category':
                            unique_cats = df[target_col].dropna().unique()
                            st.write(f"Target categories: {', '.join(map(str, unique_cats))}")
                            
                            # Map first category to 0, second to 1
                            class_map = {unique_cats[0]: 0, unique_cats[1]: 1}
                            y = df[target_col].map(class_map).dropna()
                            
                            # Show mapping
                            st.write(f"Mapping: {unique_cats[0]} → 0, {unique_cats[1]} → 1")
                        else:
                            unique_vals = sorted(df[target_col].dropna().unique())
                            st.write(f"Target values: {', '.join(map(str, unique_vals))}")
                            
                            # Map lower value to 0, higher to 1
                            class_map = {unique_vals[0]: 0, unique_vals[1]: 1}
                            y = df[target_col].map(class_map).dropna()
                            
                            # Show mapping
                            st.write(f"Mapping: {unique_vals[0]} → 0, {unique_vals[1]} → 1")
                        
                        # Select features
                        potential_features = [col for col in numeric_cols if col != target_col]
                        
                        if len(potential_features) < 1:
                            st.error("Need at least one numeric feature for logistic regression")
                        else:
                            selected_features = st.multiselect(
                                "Select features",
                                potential_features,
                                default=potential_features[:min(3, len(potential_features))]
                            )
                            
                            if not selected_features:
                                st.error("Please select at least one feature")
                            else:
                                # Prepare data
                                all_cols = selected_features + [target_col]
                                mask = df[all_cols].notna().all(axis=1)
                                X = df.loc[mask, selected_features]
                                y = df.loc[mask, target_col].map(class_map)
                                
                                # Train model
                                from sklearn.linear_model import LogisticRegression
                                from sklearn.metrics import (
                                    accuracy_score, precision_score, recall_score, 
                                    f1_score, roc_auc_score, confusion_matrix
                                )
                                
                                model = LogisticRegression(max_iter=1000)
                                model.fit(X, y)
                                
                                # Make predictions
                                y_pred_proba = model.predict_proba(X)[:, 1]
                                y_pred = model.predict(X)
                                
                                # Calculate metrics
                                accuracy = accuracy_score(y, y_pred)
                                precision = precision_score(y, y_pred)
                                recall = recall_score(y, y_pred)
                                f1 = f1_score(y, y_pred)
                                roc_auc = roc_auc_score(y, y_pred_proba)
                                conf_matrix = confusion_matrix(y, y_pred)
                                
                                # Show logistic regression equation
                                st.write("**Logistic Regression**")
                                
                                formula = f"logit(P) = {model.intercept_[0]:.4f}"
                                for i, feature in enumerate(selected_features):
                                    coef = model.coef_[0, i]
                                    sign = "+" if coef >= 0 else ""
                                    formula += f" {sign} {coef:.4f} × {feature}"
                                
                                st.write(f"Model: {formula}")
                                st.write("Where P is the probability of the target being 1")
                                
                                # Show coefficients and odds ratios
                                coef_df = pd.DataFrame({
                                    'Feature': selected_features,
                                    'Coefficient': model.coef_[0],
                                    'Odds Ratio': np.exp(model.coef_[0])
                                })
                                
                                st.write("**Coefficients and Odds Ratios**")
                                st.dataframe(coef_df)
                                
                                # Show metrics
                                metrics_df = pd.DataFrame({
                                    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC'],
                                    'Value': [accuracy, precision, recall, f1, roc_auc]
                                })
                                
                                st.write("**Model Performance**")
                                st.dataframe(metrics_df)
                                
                                # Confusion matrix
                                st.write("**Confusion Matrix**")
                                
                                cm_df = pd.DataFrame(
                                    conf_matrix,
                                    index=['Actual 0', 'Actual 1'],
                                    columns=['Predicted 0', 'Predicted 1']
                                )
                                
                                st.dataframe(cm_df)
                                
                                # Visualize confusion matrix
                                fig, ax = plt.subplots(figsize=(8, 6))
                                
                                sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                                            xticklabels=['Predicted 0', 'Predicted 1'],
                                            yticklabels=['Actual 0', 'Actual 1'], ax=ax)
                                
                                plt.title('Confusion Matrix')
                                st.pyplot(fig)
                                
                                # ROC curve
                                from sklearn.metrics import roc_curve
                                
                                fpr, tpr, _ = roc_curve(y, y_pred_proba)
                                
                                fig, ax = plt.subplots(figsize=(8, 6))
                                
                                ax.plot(fpr, tpr, label=f'AUC = {roc_auc:.4f}')
                                ax.plot([0, 1], [0, 1], 'k--')
                                
                                plt.xlabel('False Positive Rate')
                                plt.ylabel('True Positive Rate')
                                plt.title('ROC Curve')
                                plt.legend()
                                
                                st.pyplot(fig)
                                
                                # Interpretation
                                st.write("**Interpretation**:")
                                st.write("**Odds Ratios:**")
                                for feature, odds in zip(selected_features, np.exp(model.coef_[0])):
                                    if odds > 1:
                                        st.write(f"- For each unit increase in {feature}, the odds of the target being 1 increase by a factor of {odds:.4f}")
                                    else:
                                        st.write(f"- For each unit increase in {feature}, the odds of the target being 1 decrease by a factor of {1/odds:.4f}")
        
        elif analysis_type == "Time Series Analysis":
            st.markdown('<p class="highlight">Time Series Analysis</p>', unsafe_allow_html=True)
            
            # Check if there's a datetime column
            datetime_cols = []
            for col in df.columns:
                if pd.api.types.is_datetime64_any_dtype(df[col]):
                    datetime_cols.append(col)
                elif df[col].dtype == 'object':
                    # Try to convert to datetime
                    try:
                        pd.to_datetime(df[col])
                        datetime_cols.append(col)
                    except:
                        pass
            
            if not datetime_cols:
                st.warning("No datetime columns detected. Please convert a column to datetime first.")
                
                # Let user select a column to convert
                date_col = st.selectbox("Select a column to convert to datetime", df.columns)
                
                if st.button("Convert to datetime"):
                    try:
                        df[date_col] = pd.to_datetime(df[date_col])
                        st.session_state.data = df
                        st.success(f"Converted {date_col} to datetime format!")
                        datetime_cols = [date_col]
                    except Exception as e:
                        st.error(f"Error converting to datetime: {e}")
            
            if datetime_cols:
                # Choose time column
                time_col = st.selectbox("Select time column", datetime_cols)
                
                # Choose value column
                value_col = st.selectbox("Select value column to analyze", numeric_cols)
                
                # Sort by time
                df_ts = df[[time_col, value_col]].dropna().sort_values(time_col)
                
                # Set time as index
                df_ts = df_ts.set_index(time_col)
                
                st.write(f"**Time Series: {value_col} over {time_col}**")
                
                # Determine frequency
                try:
                    # Try to infer frequency
                    freq = pd.infer_freq(df_ts.index)
                    if freq:
                        st.write(f"Detected frequency: {freq}")
                    else:
                        st.write("Frequency could not be automatically detected")
                        
                    # Calculate timedelta
                    time_diff = df_ts.index[1:] - df_ts.index[:-1]
                    avg_diff = time_diff.mean()
                    st.write(f"Average time difference: {avg_diff}")
                    
                except Exception as e:
                    st.error(f"Error detecting frequency: {e}")
                
                # Plot time series
                fig, ax = plt.subplots(figsize=(12, 6))
                
                ax.plot(df_ts.index, df_ts[value_col])
                
                plt.title(f'Time Series: {value_col} over time')
                plt.xlabel(time_col)
                plt.ylabel(value_col)
                plt.grid(True)
                plt.tight_layout()
                
                st.pyplot(fig)
                
                # Time series analysis options
                ts_options = st.radio(
                    "Time series analysis options",
                    ["Trend and Seasonality", "Autocorrelation", "Moving Averages", 
                     "Resampling"]
                )
                
                if ts_options == "Trend and Seasonality":
                    try:
                        from statsmodels.tsa.seasonal import seasonal_decompose
                        
                        # Check if enough data points
                        if len(df_ts) < 4:
                            st.error("Need at least 4 data points for decomposition")
                        else:
                            # Try to decompose the time series
                            decomp_model = st.selectbox(
                                "Decomposition model",
                                ["additive", "multiplicative"]
                            )
                            
                            # Get a reasonable period for seasonal decomposition
                            suggested_period = min(len(df_ts) // 2, 12)  # Default to 12 or half data length
                            period = st.slider(
                                "Period for seasonal decomposition",
                                2, min(len(df_ts)-1, 52), suggested_period
                            )
                            
                            # Perform decomposition
                            try:
                                result = seasonal_decompose(
                                    df_ts[value_col], model=decomp_model, period=period
                                )
                                
                                # Create figure
                                fig, axes = plt.subplots(4, 1, figsize=(12, 10))
                                
                                # Original
                                axes[0].plot(result.observed)
                                axes[0].set_title('Original')
                                axes[0].grid(True)
                                
                                # Trend
                                axes[1].plot(result.trend)
                                axes[1].set_title('Trend')
                                axes[1].grid(True)
                                
                                # Seasonal
                                axes[2].plot(result.seasonal)
                                axes[2].set_title('Seasonality')
                                axes[2].grid(True)
                                
                                # Residual
                                axes[3].plot(result.resid)
                                axes[3].set_title('Residuals')
                                axes[3].grid(True)
                                
                                plt.tight_layout()
                                st.pyplot(fig)
                                
                                # Show trend summary
                                trend_change = result.trend.iloc[-1] - result.trend.iloc[0]
                                trend_pct = (trend_change / abs(result.trend.iloc[0])) * 100 if result.trend.iloc[0] != 0 else float('inf')
                                
                                st.write("**Trend Summary**")
                                st.write(f"- Overall trend change: {trend_change:.4f} ({trend_pct:.2f}%)")
                                
                                # Seasonal pattern
                                st.write("**Seasonal Pattern**")
                                if decomp_model == "additive":
                                    st.write(f"- Seasonal amplitude: {result.seasonal.max() - result.seasonal.min():.4f}")
                                else:
                                    st.write(f"- Seasonal amplitude factor: {result.seasonal.max() / result.seasonal.min():.4f}x")
                                
                            except Exception as e:
                                st.error(f"Error in decomposition: {e}")
                                st.write("Try a different period or model.")
                    
                    except ImportError:
                        st.error("statsmodels library not available")
                
                elif ts_options == "Autocorrelation":
                    try:
                        from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
                        
                        # Calculate and plot ACF and PACF
                        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
                        
                        lags = min(40, len(df_ts) // 2)
                        
                        plot_acf(df_ts[value_col].dropna(), lags=lags, ax=ax1)
                        ax1.set_title('Autocorrelation Function (ACF)')
                        
                        plot_pacf(df_ts[value_col].dropna(), lags=lags, ax=ax2)
                        ax2.set_title('Partial Autocorrelation Function (PACF)')
                        
                        plt.tight_layout()
                        st.pyplot(fig)
                        
                        # Interpretation
                        st.write("**Interpretation**:")
                        st.write("- ACF shows correlation between a time series and its lagged values")
                        st.write("- PACF shows correlation between a time series and a specific lag, removing effects of intermediate lags")
                        st.write("- Significant spikes suggest potential AR/MA orders for ARIMA modeling")
                        st.write("- Seasonality often appears as regular patterns in ACF")
                    
                    except ImportError:
                        st.error("statsmodels library not available")
                
                elif ts_options == "Moving Averages":
                    # Moving averages of different windows
                    window_sizes = st.multiselect(
                        "Select moving average window sizes",
                        [3, 5, 7, 14, 21, 30, 60, 90],
                        default=[7, 30]
                    )
                    
                    if window_sizes:
                        # Create plot
                        fig, ax = plt.subplots(figsize=(12, 6))
                        
                        # Plot original data
                        ax.plot(df_ts.index, df_ts[value_col], label='Original', alpha=0.5)
                        
                        # Plot moving averages
                        for window in window_sizes:
                            if window < len(df_ts):
                                ma = df_ts[value_col].rolling(window=window).mean()
                                ax.plot(df_ts.index, ma, label=f'MA({window})')
                            else:
                                st.warning(f"Window size {window} is larger than the data size")
                        
                        plt.title(f'Moving Averages of {value_col}')
                        plt.xlabel(time_col)
                        plt.ylabel(value_col)
                        plt.legend()
                        plt.grid(True)
                        plt.tight_layout()
                        
                        st.pyplot(fig)
                        
                        # Add exponential moving average
                        if st.checkbox("Show Exponential Moving Average"):
                            fig, ax = plt.subplots(figsize=(12, 6))
                            
                            # Plot original data
                            ax.plot(df_ts.index, df_ts[value_col], label='Original', alpha=0.5)
                            
                            # Calculate and plot EMAs with different spans
                            for span in [0.2, 0.5, 0.8]:
                                ema = df_ts[value_col].ewm(span=span).mean()
                                ax.plot(df_ts.index, ema, label=f'EMA(span={span})')
                            
                            plt.title(f'Exponential Moving Averages of {value_col}')
                            plt.xlabel(time_col)
                            plt.ylabel(value_col)
                            plt.legend()
                            plt.grid(True)
                            plt.tight_layout()
                            
                            st.pyplot(fig)
                
                elif ts_options == "Resampling":
                    # Resampling options
                    freq_options = {
                        "Daily": "D",
                        "Weekly": "W",
                        "Monthly": "M",
                        "Quarterly": "Q",
                        "Yearly": "Y",
                        "Hourly": "H",
                        "30 Minutes": "30min"
                    }
                    
                    selected_freq = st.selectbox(
                        "Select resampling frequency",
                        list(freq_options.keys())
                    )
                    
                    agg_method = st.selectbox(
                        "Aggregation method",
                        ["Mean", "Sum", "Max", "Min", "Count", "First", "Last"]
                    )
                    
                    # Get pandas resample agg method
                    agg_dict = {
                        "Mean": "mean",
                        "Sum": "sum",
                        "Max": "max",
                        "Min": "min",
                        "Count": "count",
                        "First": "first",
                        "Last": "last"
                    }
                    
                    # Resample the data
                    freq = freq_options[selected_freq]
                    agg = agg_dict[agg_method]
                    
                    try:
                        resampled = df_ts[value_col].resample(freq).agg(agg)
                        
                        # Display resampled data
                        st.write(f"**Resampled Data ({selected_freq}, {agg_method})**")
                        st.dataframe(resampled.head())
                        
                        # Plot
                        fig, ax = plt.subplots(figsize=(12, 6))
                        
                        ax.plot(df_ts.index, df_ts[value_col], 'b-', alpha=0.5, label='Original')
                        ax.plot(resampled.index, resampled, 'r.-', label=f'Resampled ({selected_freq})')
                        
                        plt.title(f'Resampled Time Series: {value_col} ({selected_freq}, {agg_method})')
                        plt.xlabel(time_col)
                        plt.ylabel(value_col)
                        plt.legend()
                        plt.grid(True)
                        plt.tight_layout()
                        
                        st.pyplot(fig)
                    
                    except Exception as e:
                        st.error(f"Error in resampling: {e}")
                        st.write("Try a different frequency or make sure the time index is properly formatted")

#######################
# MACHINE LEARNING PAGE
#######################
elif page == "Machine Learning":
    st.markdown('<p class="sub-header">Machine Learning</p>', unsafe_allow_html=True)
    
    if st.session_state.data is None:
        st.warning("Please upload or select a dataset first!")
    else:
        df = st.session_state.data
        
        ml_task = st.sidebar.selectbox(
            "Machine Learning Task",
            ["Supervised Learning", "Clustering", "Dimensionality Reduction", 
             "Model Evaluation"]
        )
        
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        categoric_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        if ml_task == "Supervised Learning":
            st.markdown('<p class="highlight">Supervised Learning</p>', unsafe_allow_html=True)
            
            # Select problem type
            problem_type = st.selectbox(
                "Problem type",
                ["Classification", "Regression"]
            )
            
            # Target selection
            if problem_type == "Classification":
                target_col = st.selectbox("Select target variable", df.columns)
                
                # Check if target is suitable for classification
                unique_vals = df[target_col].nunique()
                if unique_vals > 10:
                    st.warning(f"Target has {unique_vals} unique values. Consider using regression instead.")
                
                # Display class distribution
                st.write("**Class Distribution**")
                class_dist = df[target_col].value_counts().reset_index()
                class_dist.columns = [target_col, 'Count']
                
                fig, ax = plt.subplots(figsize=(10, 5))
                sns.barplot(x=target_col, y='Count', data=class_dist, ax=ax)
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                st.pyplot(fig)
                
            else:  # Regression
                target_col = st.selectbox("Select target variable", numeric_cols)
                
                # Display target distribution
                st.write("**Target Distribution**")
                fig, ax = plt.subplots(figsize=(10, 5))
                sns.histplot(df[target_col].dropna(), kde=True, ax=ax)
                plt.tight_layout()
                st.pyplot(fig)
            
            # Feature selection
            potential_features = [col for col in df.columns if col != target_col]
            selected_features = st.multiselect(
                "Select features",
                potential_features,
                default=potential_features[:min(5, len(potential_features))]
            )
            
            if len(selected_features) == 0:
                st.error("Please select at least one feature")
            else:
                # Preprocessing options
                with st.expander("Preprocessing Options"):
                    # Handle missing values
                    missing_strategy = st.selectbox(
                        "Missing values strategy",
                        ["Drop rows", "Mean/Mode imputation", "Median imputation"]
                    )
                    
                    # Handle categorical features
                    cat_encoding = st.selectbox(
                        "Categorical encoding",
                        ["One-hot encoding", "Label encoding"]
                    )
                    
                    # Feature scaling
                    scaling = st.selectbox(
                        "Feature scaling",
                        ["None", "StandardScaler", "MinMaxScaler"]
                    )
                    
                    # Feature engineering
                    poly_features = st.checkbox("Add polynomial features")
                    if poly_features:
                        poly_degree = st.slider("Polynomial degree", 2, 5, 2)
                    
                    interaction_terms = st.checkbox("Add interaction terms")
                
                # Train-test split
                with st.expander("Train-Test Split"):
                    test_size = st.slider("Test size (%)", 10, 50, 20) / 100
                    use_stratify = st.checkbox("Use stratified sampling", True)
                    random_state = st.number_input("Random state", 0, 1000, 42)
                
                # Model selection
                with st.expander("Model Selection"):
                    if problem_type == "Classification":
                        model_options = [
                            "Logistic Regression", 
                            "Decision Tree", 
                            "Random Forest",
                            "Gradient Boosting",
                            "Support Vector Machine",
                            "K-Nearest Neighbors"
                        ]
                    else:  # Regression
                        model_options = [
                            "Linear Regression",
                            "Decision Tree",
                            "Random Forest",
                            "Gradient Boosting",
                            "Support Vector Machine",
                            "K-Nearest Neighbors"
                        ]
                    
                    selected_model = st.selectbox("Select model", model_options)
                    
                    # Hyperparameters
                    st.write("**Hyperparameters**")
                    
                    if selected_model == "Logistic Regression" or selected_model == "Linear Regression":
                        C = st.slider("Regularization strength (C)", 0.01, 10.0, 1.0)
                        max_iter = st.slider("Maximum iterations", 100, 1000, 100)
                        
                        model_params = {
                            "C": C,
                            "max_iter": max_iter
                        }
                        
                    elif selected_model == "Decision Tree":
                        max_depth = st.slider("Maximum depth", 1, 20, 5)
                        min_samples_split = st.slider("Minimum samples to split", 2, 20, 2)
                        
                        model_params = {
                            "max_depth": max_depth,
                            "min_samples_split": min_samples_split,
                            "random_state": random_state
                        }
                        
                    elif selected_model == "Random Forest":
                        n_estimators = st.slider("Number of trees", 10, 200, 100)
                        max_depth = st.slider("Maximum depth", 1, 20, 5)
                        
                        model_params = {
                            "n_estimators": n_estimators,
                            "max_depth": max_depth,
                            "random_state": random_state
                        }
                        
                    elif selected_model == "Gradient Boosting":
                        n_estimators = st.slider("Number of boosting stages", 10, 200, 100)
                        learning_rate = st.slider("Learning rate", 0.01, 0.5, 0.1)
                        
                        model_params = {
                            "n_estimators": n_estimators,
                            "learning_rate": learning_rate,
                            "random_state": random_state
                        }
                        
                    elif selected_model == "Support Vector Machine":
                        C = st.slider("Regularization strength (C)", 0.01, 10.0, 1.0)
                        kernel = st.selectbox("Kernel", ["linear", "rbf", "poly"])
                        
                        model_params = {
                            "C": C,
                            "kernel": kernel
                        }
                        
                    elif selected_model == "K-Nearest Neighbors":
                        n_neighbors = st.slider("Number of neighbors", 1, 20, 5)
                        weights = st.selectbox("Weight function", ["uniform", "distance"])
                        
                        model_params = {
                            "n_neighbors": n_neighbors,
                            "weights": weights
                        }
                
                # Train the model
                if st.button("Train Model"):
                    with st.spinner("Training in progress..."):
                        # Prepare data
                        X = df[selected_features].copy()
                        y = df[target_col].copy()
                        
                        # Handle missing values
                        if missing_strategy == "Drop rows":
                            mask = X[X.columns].notna().all(axis=1) & y.notna()
                            X = X[mask]
                            y = y[mask]
                        else:
                            # Imputation will be part of the pipeline
                            pass
                        
                        # Check if we have enough data
                        if len(X) < 10:
                            st.error("Not enough data after preprocessing")
                        else:
                            # Split data
                            from sklearn.model_selection import train_test_split
                            
                            stratify_param = None
                            if use_stratify and problem_type == "Classification":
                                stratify_param = y
                            try:
                                X_train, X_test, y_train, y_test = train_test_split(
                                    X, y, test_size=test_size, random_state=random_state,
                                    stratify=stratify_param
                                )
                                
                                # Preprocessing steps
                                from sklearn.pipeline import Pipeline
                                from sklearn.compose import ColumnTransformer
                                from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, LabelEncoder, PolynomialFeatures
                                from sklearn.impute import SimpleImputer
                                
                                # Identify numeric and categorical columns
                                numeric_features = X.select_dtypes(include=['number']).columns.tolist()
                                categorical_features = X.select_dtypes(exclude=['number']).columns.tolist()
                                
                                # Create preprocessing pipeline
                                preprocessor_steps = []
                                
                                # Numeric preprocessing
                                if numeric_features:
                                    # Numeric imputation
                                    if missing_strategy == "Mean/Mode imputation":
                                        num_imputer = SimpleImputer(strategy='mean')
                                    elif missing_strategy == "Median imputation":
                                        num_imputer = SimpleImputer(strategy='median')
                                    
                                    # Scaling
                                    if scaling == "StandardScaler":
                                        num_scaler = StandardScaler()
                                    elif scaling == "MinMaxScaler":
                                        num_scaler = MinMaxScaler()
                                    else:
                                        num_scaler = None
                                    
                                    if scaling != "None":
                                        num_pipeline = Pipeline([
                                            ('imputer', num_imputer),
                                            ('scaler', num_scaler)
                                        ])
                                    else:
                                        num_pipeline = Pipeline([
                                            ('imputer', num_imputer)
                                        ])
                                    
                                    preprocessor_steps.append(
                                        ('num', num_pipeline, numeric_features)
                                    )
                                
                                # Categorical preprocessing
                                if categorical_features:
                                    # Categorical imputation
                                    cat_imputer = SimpleImputer(strategy='most_frequent')
                                    
                                    # Encoding
                                    if cat_encoding == "One-hot encoding":
                                        cat_encoder = OneHotEncoder(handle_unknown='ignore')
                                        cat_pipeline = Pipeline([
                                            ('imputer', cat_imputer),
                                            ('encoder', cat_encoder)
                                        ])
                                    else:  # Label encoding
                                        cat_pipeline = Pipeline([
                                            ('imputer', cat_imputer)
                                        ])
                                    
                                    preprocessor_steps.append(
                                        ('cat', cat_pipeline, categorical_features)
                                    )
                                
                                # Create the main preprocessor
                                preprocessor = ColumnTransformer(
                                    transformers=preprocessor_steps,
                                    remainder='drop'
                                )
                                
                                # For label encoding, we need to transform y if it's categorical
                                if problem_type == "Classification" and y.dtype == 'object':
                                    label_encoder = LabelEncoder()
                                    y_train = label_encoder.fit_transform(y_train)
                                    y_test = label_encoder.transform(y_test)
                                
                                # Create model
                                if selected_model == "Logistic Regression":
                                    from sklearn.linear_model import LogisticRegression
                                    model = LogisticRegression(**model_params)
                                    
                                elif selected_model == "Linear Regression":
                                    from sklearn.linear_model import LinearRegression
                                    model = LinearRegression(**model_params)
                                    
                                elif selected_model == "Decision Tree":
                                    if problem_type == "Classification":
                                        from sklearn.tree import DecisionTreeClassifier
                                        model = DecisionTreeClassifier(**model_params)
                                    else:
                                        from sklearn.tree import DecisionTreeRegressor
                                        model = DecisionTreeRegressor(**model_params)
                                    
                                elif selected_model == "Random Forest":
                                    if problem_type == "Classification":
                                        from sklearn.ensemble import RandomForestClassifier
                                        model = RandomForestClassifier(**model_params)
                                    else:
                                        from sklearn.ensemble import RandomForestRegressor
                                        model = RandomForestRegressor(**model_params)
                                    
                                elif selected_model == "Gradient Boosting":
                                    if problem_type == "Classification":
                                        from sklearn.ensemble import GradientBoostingClassifier
                                        model = GradientBoostingClassifier(**model_params)
                                    else:
                                        from sklearn.ensemble import GradientBoostingRegressor
                                        model = GradientBoostingRegressor(**model_params)
                                    
                                elif selected_model == "Support Vector Machine":
                                    if problem_type == "Classification":
                                        from sklearn.svm import SVC
                                        model = SVC(**model_params, probability=True)
                                    else:
                                        from sklearn.svm import SVR
                                        model = SVR(**model_params)
                                    
                                elif selected_model == "K-Nearest Neighbors":
                                    if problem_type == "Classification":
                                        from sklearn.neighbors import KNeighborsClassifier
                                        model = KNeighborsClassifier(**model_params)
                                    else:
                                        from sklearn.neighbors import KNeighborsRegressor
                                        model = KNeighborsRegressor(**model_params)
                                
                                # Feature engineering steps
                                steps = [('preprocessor', preprocessor)]
                                
                                if poly_features and numeric_features:
                                    steps.append(('poly', PolynomialFeatures(degree=poly_degree)))
                                
                                # Add final model
                                steps.append(('model', model))
                                
                                # Create final pipeline
                                pipeline = Pipeline(steps)
                                
                                # Train the model
                                pipeline.fit(X_train, y_train)
                                
                                # Make predictions
                                y_pred = pipeline.predict(X_test)
                                
                                # Store the model in session state
                                st.session_state.ml_model = {
                                    'pipeline': pipeline,
                                    'X_train': X_train,
                                    'X_test': X_test,
                                    'y_train': y_train,
                                    'y_test': y_test,
                                    'y_pred': y_pred,
                                    'problem_type': problem_type,
                                    'selected_model': selected_model,
                                    'features': selected_features,
                                    'target': target_col
                                }
                                
                                # Show success
                                st.success("Model trained successfully!")
                                
                                # Evaluate the model
                                st.write("**Model Evaluation**")
                                
                                if problem_type == "Classification":
                                    from sklearn.metrics import (
                                        accuracy_score, precision_score, recall_score,
                                        f1_score, classification_report, confusion_matrix,
                                        roc_auc_score, roc_curve
                                    )
                                    
                                    # For multi-class problems, we need to adjust metrics
                                    multi_class = len(np.unique(y_test)) > 2
                                    
                                    # Classification metrics
                                    accuracy = accuracy_score(y_test, y_pred)
                                    
                                    if multi_class:
                                        precision = precision_score(y_test, y_pred, average='weighted')
                                        recall = recall_score(y_test, y_pred, average='weighted')
                                        f1 = f1_score(y_test, y_pred, average='weighted')
                                    else:
                                        precision = precision_score(y_test, y_pred)
                                        recall = recall_score(y_test, y_pred)
                                        f1 = f1_score(y_test, y_pred)
                                    
                                    # Display metrics
                                    metrics_df = pd.DataFrame({
                                        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score'],
                                        'Value': [accuracy, precision, recall, f1]
                                    })
                                    
                                    st.dataframe(metrics_df)
                                    
                                    # Classification report
                                    st.write("**Classification Report**")
                                    report = classification_report(y_test, y_pred, output_dict=True)
                                    report_df = pd.DataFrame(report).transpose()
                                    st.dataframe(report_df)
                                    
                                    # Confusion matrix
                                    st.write("**Confusion Matrix**")
                                    cm = confusion_matrix(y_test, y_pred)
                                    
                                    fig, ax = plt.subplots(figsize=(8, 6))
                                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
                                    plt.xlabel('Predicted')
                                    plt.ylabel('True')
                                    st.pyplot(fig)
                                    
                                    # ROC Curve for binary classification
                                    if not multi_class:
                                        try:
                                            y_prob = pipeline.predict_proba(X_test)[:, 1]
                                            fpr, tpr, _ = roc_curve(y_test, y_prob)
                                            auc = roc_auc_score(y_test, y_prob)
                                            
                                            fig, ax = plt.subplots(figsize=(8, 6))
                                            ax.plot(fpr, tpr, label=f'AUC = {auc:.4f}')
                                            ax.plot([0, 1], [0, 1], 'k--')
                                            plt.xlabel('False Positive Rate')
                                            plt.ylabel('True Positive Rate')
                                            plt.title('ROC Curve')
                                            plt.legend()
                                            st.pyplot(fig)
                                        except:
                                            st.write("ROC curve cannot be generated for this model")
                                
                                else:  # Regression metrics
                                    from sklearn.metrics import (
                                        mean_squared_error, mean_absolute_error,
                                        r2_score, explained_variance_score
                                    )
                                    
                                    # Calculate metrics
                                    mse = mean_squared_error(y_test, y_pred)
                                    rmse = np.sqrt(mse)
                                    mae = mean_absolute_error(y_test, y_pred)
                                    r2 = r2_score(y_test, y_pred)
                                    evs = explained_variance_score(y_test, y_pred)
                                    
                                    # Display metrics
                                    metrics_df = pd.DataFrame({
                                        'Metric': ['MSE', 'RMSE', 'MAE', 'R²', 'Explained Variance'],
                                        'Value': [mse, rmse, mae, r2, evs]
                                    })
                                    
                                    st.dataframe(metrics_df)
                                    
                                    # Predicted vs Actual plot
                                    fig, ax = plt.subplots(figsize=(8, 6))
                                    ax.scatter(y_test, y_pred, alpha=0.7)
                                    
                                    # Add perfect prediction line
                                    max_val = max(y_test.max(), y_pred.max())
                                    min_val = min(y_test.min(), y_pred.min())
                                    ax.plot([min_val, max_val], [min_val, max_val], 'r--')
                                    
                                    plt.xlabel('Actual')
                                    plt.ylabel('Predicted')
                                    plt.title('Predicted vs Actual')
                                    st.pyplot(fig)
                                    
                                    # Residual plot
                                    residuals = y_test - y_pred
                                    
                                    fig, ax = plt.subplots(figsize=(8, 6))
                                    ax.scatter(y_pred, residuals, alpha=0.7)
                                    ax.axhline(y=0, color='r', linestyle='--')
                                    
                                    plt.xlabel('Predicted')
                                    plt.ylabel('Residuals')
                                    plt.title('Residual Plot')
                                    st.pyplot(fig)
                                
                                # Feature importance (if available)
                                if hasattr(pipeline.named_steps['model'], 'feature_importances_'):
                                    st.write("**Feature Importance**")
                                    
                                    # Get feature names after preprocessing
                                    feature_names = selected_features
                                    if categorical_features and cat_encoding == "One-hot encoding":
                                        # This is a simplification; actual feature names after one-hot encoding are more complex
                                        feature_names = [f for f in feature_names if f not in categorical_features]
                                        for cat in categorical_features:
                                            unique_vals = X[cat].nunique()
                                            for i in range(unique_vals):
                                                feature_names.append(f"{cat}_{i}")
                                    
                                    # For simple pipelines, try to get feature importance
                                    importance = pipeline.named_steps['model'].feature_importances_
                                    
                                    # Create DataFrame (using the first N feature names that match the importance length)
                                    feature_names = feature_names[:len(importance)]
                                    importance_df = pd.DataFrame({
                                        'Feature': feature_names[:len(importance)],
                                        'Importance': importance
                                    }).sort_values('Importance', ascending=False)
                                    
                                    st.dataframe(importance_df)
                                    
                                    # Plot
                                    fig, ax = plt.subplots(figsize=(10, len(importance_df) * 0.3 + 2))
                                    sns.barplot(x='Importance', y='Feature', data=importance_df.head(10), ax=ax)
                                    plt.title('Feature Importance')
                                    st.pyplot(fig)
                                
                                # Prediction interface
                                st.write("**Make Predictions**")
                                st.write("Enter values for each feature to get a prediction:")
                                
                                # Create input fields for each feature
                                input_data = {}
                                
                                for feature in selected_features:
                                    if feature in numeric_features:
                                        min_val = float(X[feature].min())
                                        max_val = float(X[feature].max())
                                        mean_val = float(X[feature].mean())
                                        
                                        input_data[feature] = st.slider(
                                            f"{feature}", min_val, max_val, mean_val
                                        )
                                    else:
                                        # For categorical features
                                        options = X[feature].unique().tolist()
                                        input_data[feature] = st.selectbox(
                                            f"{feature}", options
                                        )
                                
                                # Make prediction button
                                if st.button("Predict"):
                                    # Convert input_data to DataFrame
                                    input_df = pd.DataFrame([input_data])
                                    
                                    # Make prediction
                                    prediction = pipeline.predict(input_df)
                                    
                                    st.write("**Prediction Result**")
                                    
                                    if problem_type == "Classification":
                                        if hasattr(pipeline, 'predict_proba'):
                                            proba = pipeline.predict_proba(input_df)
                                            
                                            st.write(f"Predicted class: **{prediction[0]}**")
                                            
                                            # For binary classification
                                            if proba.shape[1] == 2:
                                                st.write(f"Probability: **{proba[0][1]:.4f}**")
                                            else:
                                                # For multi-class, show all probabilities
                                                proba_df = pd.DataFrame(
                                                    proba[0],
                                                    index=pipeline.classes_,
                                                    columns=['Probability']
                                                ).sort_values('Probability', ascending=False)
                                                
                                                st.dataframe(proba_df)
                                        else:
                                            st.write(f"Predicted class: **{prediction[0]}**")
                                    else:
                                        st.write(f"Predicted value: **{prediction[0]:.4f}**")
                                
                            except Exception as e:
                                st.error(f"Error during model training: {e}")
                                import traceback
                                st.text(traceback.format_exc())
        
        elif ml_task == "Clustering":
            st.markdown('<p class="highlight">Clustering</p>', unsafe_allow_html=True)
            
            if len(numeric_cols) < 2:
                st.error("Clustering requires at least 2 numeric columns")
            else:
                # Feature selection
                features = st.multiselect(
                    "Select features for clustering", 
                    numeric_cols,
                    default=numeric_cols[:min(3, len(numeric_cols))]
                )
                
                if len(features) < 2:
                    st.error("Please select at least 2 features")
                else:
                    # Clustering algorithm
                    algorithm = st.selectbox(
                        "Clustering algorithm",
                        ["K-Means", "DBSCAN", "Hierarchical", "Gaussian Mixture"]
                    )
                    
                    # Parameters based on algorithm
                    if algorithm == "K-Means":
                        n_clusters = st.slider("Number of clusters", 2, 10, 3)
                        
                        if st.button("Perform Clustering"):
                            with st.spinner("Clustering in progress..."):
                                # Get data
                                X = df[features].dropna()
                                
                                # Normalize data
                                from sklearn.preprocessing import StandardScaler
                                scaler = StandardScaler()
                                X_scaled = scaler.fit_transform(X)
                                
                                # Apply K-Means
                                from sklearn.cluster import KMeans
                                kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                                clusters = kmeans.fit_predict(X_scaled)
                                
                                # Add clusters to dataframe
                                clustered_df = X.copy()
                                clustered_df['Cluster'] = clusters
                                
                                # Show results
                                st.write("**Clustering Results**")
                                
                                # Cluster summary
                                cluster_counts = clustered_df['Cluster'].value_counts().reset_index()
                                cluster_counts.columns = ['Cluster', 'Count']
                                
                                fig, ax = plt.subplots(figsize=(10, 5))
                                sns.barplot(x='Cluster', y='Count', data=cluster_counts, ax=ax)
                                plt.title('Cluster Sizes')
                                st.pyplot(fig)
                                
                                # Cluster centers
                                st.write("**Cluster Centers**")
                                centers = pd.DataFrame(
                                    scaler.inverse_transform(kmeans.cluster_centers_),
                                    columns=features
                                )
                                centers['Cluster'] = centers.index
                                st.dataframe(centers)
                                
                                # Visualizations
                                if len(features) == 2:
                                    # 2D plot
                                    fig, ax = plt.subplots(figsize=(10, 8))
                                    scatter = ax.scatter(
                                        clustered_df[features[0]], 
                                        clustered_df[features[1]],
                                        c=clustered_df['Cluster'], 
                                        cmap='viridis', 
                                        alpha=0.7
                                    )
                                    
                                    # Add cluster centers
                                    ax.scatter(
                                        centers[features[0]],
                                        centers[features[1]],
                                        c='red',
                                        marker='X',
                                        s=200,
                                        label='Cluster Centers'
                                    )
                                    
                                    plt.xlabel(features[0])
                                    plt.ylabel(features[1])
                                    plt.title(f'K-Means Clustering ({features[0]} vs {features[1]})')
                                    plt.colorbar(scatter, label='Cluster')
                                    plt.legend()
                                    st.pyplot(fig)
                                elif len(features) == 3:
                                    # 3D plot
                                    try:
                                        import plotly.express as px
                                        
                                        fig = px.scatter_3d(
                                            clustered_df, 
                                            x=features[0], 
                                            y=features[1], 
                                            z=features[2],
                                            color='Cluster', 
                                            title=f'K-Means Clustering ({features[0]}, {features[1]}, {features[2]})'
                                        )
                                        
                                        st.plotly_chart(fig)
                                    except ImportError:
                                        st.error("Plotly is required for 3D visualization")
                                else:
                                    # For more than 3 features, show pairplot
                                    if len(clustered_df) <= 1000:  # Limit for performance
                                        fig = plt.figure(figsize=(10, 8))
                                        sns.pairplot(
                                            clustered_df,
                                            vars=features[:4],  # Limit to first 4
                                            hue='Cluster',
                                            palette='viridis'
                                        )
                                        st.pyplot(fig)
                                    else:
                                        st.warning("Too many data points for pairplot. Showing scatter matrix instead.")
                                        # Sample for better visualization
                                        sample_df = clustered_df.sample(1000, random_state=42)
                                        fig = plt.figure(figsize=(10, 8))
                                        pd.plotting.scatter_matrix(
                                            sample_df[features[:4]],  # Limit to first 4
                                            c=sample_df['Cluster'],
                                            figsize=(12, 12),
                                            alpha=0.5,
                                            diagonal='kde'
                                        )
                                        st.pyplot(fig)
                                
                                # Evaluate clustering
                                from sklearn.metrics import silhouette_score
                                
                                silhouette_avg = silhouette_score(X_scaled, clusters)
                                st.write(f"**Silhouette Score**: {silhouette_avg:.4f}")
                                
                                # Interpretation
                                st.write("**Interpretation**:")
                                if silhouette_avg > 0.7:
                                    st.write("Excellent separation between clusters")
                                elif silhouette_avg > 0.5:
                                    st.write("Good separation between clusters")
                                elif silhouette_avg > 0.25:
                                    st.write("Clusters have some overlap")
                                else:
                                    st.write("Significant overlap between clusters")
                    
                    elif algorithm == "DBSCAN":
                        eps = st.slider("Epsilon (neighborhood size)", 0.1, 2.0, 0.5)
                        min_samples = st.slider("Minimum samples in neighborhood", 2, 10, 5)
                        
                        if st.button("Perform Clustering"):
                            with st.spinner("Clustering in progress..."):
                                # Get data
                                X = df[features].dropna()
                                
                                # Normalize data
                                from sklearn.preprocessing import StandardScaler
                                scaler = StandardScaler()
                                X_scaled = scaler.fit_transform(X)
                                
                                # Apply DBSCAN
                                from sklearn.cluster import DBSCAN
                                dbscan = DBSCAN(eps=eps, min_samples=min_samples)
                                clusters = dbscan.fit_predict(X_scaled)
                                
                                # Add clusters to dataframe
                                clustered_df = X.copy()
                                clustered_df['Cluster'] = clusters
                                
                                # Show results
                                st.write("**Clustering Results**")
                                
                                # Count clusters
                                n_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)
                                noise_points = list(clusters).count(-1)
                                
                                st.write(f"Number of clusters: {n_clusters}")
                                st.write(f"Number of noise points: {noise_points} ({noise_points/len(clusters):.2%})")
                                
                                # Cluster summary
                                cluster_counts = clustered_df['Cluster'].value_counts().reset_index()
                                cluster_counts.columns = ['Cluster', 'Count']
                                
                                fig, ax = plt.subplots(figsize=(10, 5))
                                sns.barplot(x='Cluster', y='Count', data=cluster_counts, ax=ax)
                                plt.title('Cluster Sizes')
                                st.pyplot(fig)
                                
                                # Visualizations
                                if len(features) == 2:
                                    # 2D plot
                                    fig, ax = plt.subplots(figsize=(10, 8))
                                    scatter = ax.scatter(
                                        clustered_df[features[0]], 
                                        clustered_df[features[1]],
                                        c=clustered_df['Cluster'], 
                                        cmap='viridis', 
                                        alpha=0.7
                                    )
                                    
                                    plt.xlabel(features[0])
                                    plt.ylabel(features[1])
                                    plt.title(f'DBSCAN Clustering ({features[0]} vs {features[1]})')
                                    plt.colorbar(scatter, label='Cluster')
                                    st.pyplot(fig)
                                elif len(features) == 3:
                                    # 3D plot
                                    try:
                                        import plotly.express as px
                                        
                                        fig = px.scatter_3d(
                                            clustered_df, 
                                            x=features[0], 
                                            y=features[1], 
                                            z=features[2],
                                            color='Cluster', 
                                            title=f'DBSCAN Clustering ({features[0]}, {features[1]}, {features[2]})'
                                        )
                                        
                                        st.plotly_chart(fig)
                                    except ImportError:
                                        st.error("Plotly is required for 3D visualization")
                                else:
                                    # For more than 3 features, show pairplot
                                    if len(clustered_df) <= 1000:  # Limit for performance
                                        fig = plt.figure(figsize=(10, 8))
                                        sns.pairplot(
                                            clustered_df,
                                            vars=features[:4],  # Limit to first 4
                                            hue='Cluster',
                                            palette='viridis'
                                        )
                                        st.pyplot(fig)
                                    else:
                                        st.warning("Too many data points for pairplot. Showing sampled scatter matrix")
                                        # Sample for better visualization
                                        sample_df = clustered_df.sample(1000, random_state=42)
                                        fig = plt.figure(figsize=(10, 8))
                                        pd.plotting.scatter_matrix(
                                            sample_df[features[:4]],  # Limit to first 4
                                            c=sample_df['Cluster'],
                                            figsize=(12, 12),
                                            alpha=0.5,
                                            diagonal='kde'
                                        )
                                        st.pyplot(fig)
                                
                                # Evaluate clustering if there's more than 1 cluster and no all-noise
                                if n_clusters > 1 and noise_points < len(clusters):
                                    try:
                                        from sklearn.metrics import silhouette_score
                                        
                                        # Filter out noise points for silhouette calculation
                                        mask = clusters != -1
                                        silhouette_avg = silhouette_score(X_scaled[mask], clusters[mask])
                                        st.write(f"**Silhouette Score**: {silhouette_avg:.4f}")
                                        
                                        # Interpretation
                                        st.write("**Interpretation**:")
                                        if silhouette_avg > 0.7:
                                            st.write("Excellent separation between clusters")
                                        elif silhouette_avg > 0.5:
                                            st.write("Good separation between clusters")
                                        elif silhouette_avg > 0.25:
                                            st.write("Clusters have some overlap")
                                        else:
                                            st.write("Significant overlap between clusters")
                                    except Exception as e:
                                        st.error(f"Could not calculate silhouette score: {e}")
                    
                    elif algorithm == "Hierarchical":
                        n_clusters = st.slider("Number of clusters", 2, 10, 3)
                        linkage = st.selectbox(
                            "Linkage method",
                            ["ward", "complete", "average", "single"]
                        )
                        
                        if st.button("Perform Clustering"):
                            with st.spinner("Clustering in progress..."):
                                # Get data
                                X = df[features].dropna()
                                
                                # Normalize data
                                from sklearn.preprocessing import StandardScaler
                                scaler = StandardScaler()
                                X_scaled = scaler.fit_transform(X)
                                
                                # Apply Hierarchical Clustering
                                from sklearn.cluster import AgglomerativeClustering
                                hc = AgglomerativeClustering(
                                    n_clusters=n_clusters, 
                                    linkage=linkage
                                )
                                clusters = hc.fit_predict(X_scaled)
                                
                                # Add clusters to dataframe
                                clustered_df = X.copy()
                                clustered_df['Cluster'] = clusters
                                
                                # Show results
                                st.write("**Clustering Results**")
                                
                                # Cluster summary
                                cluster_counts = clustered_df['Cluster'].value_counts().reset_index()
                                cluster_counts.columns = ['Cluster', 'Count']
                                
                                fig, ax = plt.subplots(figsize=(10, 5))
                                sns.barplot(x='Cluster', y='Count', data=cluster_counts, ax=ax)
                                plt.title('Cluster Sizes')
                                st.pyplot(fig)
                                
                                # Visualizations
                                if len(features) == 2:
                                    # 2D plot
                                    fig, ax = plt.subplots(figsize=(10, 8))
                                    scatter = ax.scatter(
                                        clustered_df[features[0]], 
                                        clustered_df[features[1]],
                                        c=clustered_df['Cluster'], 
                                        cmap='viridis', 
                                        alpha=0.7
                                    )
                                    
                                    plt.xlabel(features[0])
                                    plt.ylabel(features[1])
                                    plt.title(f'Hierarchical Clustering ({features[0]} vs {features[1]})')
                                    plt.colorbar(scatter, label='Cluster')
                                    st.pyplot(fig)
                                elif len(features) == 3:
                                    # 3D plot
                                    try:
                                        import plotly.express as px
                                        
                                        fig = px.scatter_3d(
                                            clustered_df, 
                                            x=features[0], 
                                            y=features[1], 
                                            z=features[2],
                                            color='Cluster', 
                                            title=f'Hierarchical Clustering ({features[0]}, {features[1]}, {features[2]})'
                                        )
                                        
                                        st.plotly_chart(fig)
                                    except ImportError:
                                        st.error("Plotly is required for 3D visualization")
                                else:
                                    # For more than 3 features, show pairplot
                                    if len(clustered_df) <= 1000:  # Limit for performance
                                        fig = plt.figure(figsize=(10, 8))
                                        sns.pairplot(
                                            clustered_df,
                                            vars=features[:4],  # Limit to first 4
                                            hue='Cluster',
                                            palette='viridis'
                                        )
                                        st.pyplot(fig)
                                    else:
                                        st.warning("Too many data points for pairplot. Showing scatter matrix instead.")
                                        # Sample for better visualization
                                        sample_df = clustered_df.sample(1000, random_state=42)
                                        fig = plt.figure(figsize=(10, 8))
                                        pd.plotting.scatter_matrix(
                                            sample_df[features[:4]],  # Limit to first 4
                                            c=sample_df['Cluster'],
                                            figsize=(12, 12),
                                            alpha=0.5,
                                            diagonal='kde'
                                        )
                                        st.pyplot(fig)
                                
                                # Evaluate clustering
                                from sklearn.metrics import silhouette_score
                                
                                silhouette_avg = silhouette_score(X_scaled, clusters)
                                st.write(f"**Silhouette Score**: {silhouette_avg:.4f}")
                                
                                # Interpretation
                                st.write("**Interpretation**:")
                                if silhouette_avg > 0.7:
                                    st.write("Excellent separation between clusters")
                                elif silhouette_avg > 0.5:
                                    st.write("Good separation between clusters")
                                elif silhouette_avg > 0.25:
                                    st.write("Clusters have some overlap")
                                else:
                                    st.write("Significant overlap between clusters")
                                
                                # Dendrogram (for small datasets)
                                if len(X) <= 100:
                                    try:
                                        from scipy.cluster import hierarchy
                                        
                                        st.write("**Dendrogram**")
                                        
                                        fig, ax = plt.subplots(figsize=(12, 8))
                                        
                                        # Calculate linkage matrix
                                        Z = hierarchy.linkage(X_scaled, method=linkage)
                                        
                                        # Plot dendrogram
                                        hierarchy.dendrogram(Z, ax=ax)
                                        
                                        plt.title('Hierarchical Clustering Dendrogram')
                                        plt.xlabel('Data Points')
                                        plt.ylabel('Distance')
                                        st.pyplot(fig)
                                    except Exception as e:
                                        st.error(f"Error generating dendrogram: {e}")
                                else:
                                    st.info("Dendrogram not shown for large datasets (>100 points)")
                    
                    elif algorithm == "Gaussian Mixture":
                        n_components = st.slider("Number of components", 2, 10, 3)
                        
                        if st.button("Perform Clustering"):
                            with st.spinner("Clustering in progress..."):
                                # Get data
                                X = df[features].dropna()
                                
                                # Normalize data
                                from sklearn.preprocessing import StandardScaler
                                scaler = StandardScaler()
                                X_scaled = scaler.fit_transform(X)
                                
                                # Apply Gaussian Mixture
                                from sklearn.mixture import GaussianMixture
                                gmm = GaussianMixture(n_components=n_components, random_state=42)
                                gmm.fit(X_scaled)
                                clusters = gmm.predict(X_scaled)
                                
                                # Get probabilities
                                probs = gmm.predict_proba(X_scaled)
                                
                                # Add clusters to dataframe
                                clustered_df = X.copy()
                                clustered_df['Cluster'] = clusters
                                
                                # Add highest probability
                                clustered_df['Probability'] = np.max(probs, axis=1)
                                
                                # Show results
                                st.write("**Clustering Results**")
                                
                                # Cluster summary
                                cluster_counts = clustered_df['Cluster'].value_counts().reset_index()
                                cluster_counts.columns = ['Cluster', 'Count']
                                
                                fig, ax = plt.subplots(figsize=(10, 5))
                                sns.barplot(x='Cluster', y='Count', data=cluster_counts, ax=ax)
                                plt.title('Cluster Sizes')
                                st.pyplot(fig)
                                
                                # Visualizations
                                if len(features) == 2:
                                    # 2D plot
                                    fig, ax = plt.subplots(figsize=(10, 8))
                                    scatter = ax.scatter(
                                        clustered_df[features[0]], 
                                        clustered_df[features[1]],
                                        c=clustered_df['Cluster'], 
                                        cmap='viridis', 
                                        alpha=0.7
                                    )
                                    
                                    # Add means of components
                                    means = scaler.inverse_transform(gmm.means_)
                                    ax.scatter(
                                        means[:, 0],
                                        means[:, 1],
                                        c='red',
                                        marker='X',
                                        s=200,
                                        label='Component Means'
                                    )
                                    
                                    plt.xlabel(features[0])
                                    plt.ylabel(features[1])
                                    plt.title(f'Gaussian Mixture ({features[0]} vs {features[1]})')
                                    plt.colorbar(scatter, label='Cluster')
                                    plt.legend()
                                    st.pyplot(fig)
                                    
                                    # Plot confidence ellipses
                                    if st.checkbox("Show confidence ellipses"):
                                        from matplotlib.patches import Ellipse
                                        
                                        fig, ax = plt.subplots(figsize=(10, 8))
                                        scatter = ax.scatter(
                                            clustered_df[features[0]], 
                                            clustered_df[features[1]],
                                            c=clustered_df['Cluster'], 
                                            cmap='viridis', 
                                            alpha=0.5
                                        )
                                        
                                        # Plot an ellipse for each component
                                        for i in range(n_components):
                                            # Get covariance for this component
                                            covariance = gmm.covariances_[i][:2, :2]
                                            
                                            # Eigenvalues and eigenvectors
                                            v, w = np.linalg.eigh(covariance)
                                            u = w[0] / np.linalg.norm(w[0])
                                            angle = np.arctan2(u[1], u[0])
                                            angle = 180 * angle / np.pi  # convert to degrees
                                            
                                            # Ellipse parameters
                                            v = 2. * np.sqrt(2.) * np.sqrt(v)
                                            
                                            # Scale back to data space
                                            mean = means[i, :2]
                                            v = v * np.array([df[features[0]].std(), df[features[1]].std()])
                                            
                                            # Add ellipses at different confidence levels
                                            for j, (width, height) in enumerate(zip(v, v)):
                                                ell = Ellipse(
                                                    xy=mean, 
                                                    width=width, 
                                                    height=height,
                                                    angle=angle, 
                                                    color=f'C{i}'
                                                )
                                                ell.set_alpha(0.3)
                                                ax.add_artist(ell)
                                        
                                        plt.xlabel(features[0])
                                        plt.ylabel(features[1])
                                        plt.title(f'Gaussian Mixture with Confidence Ellipses')
                                        plt.colorbar(scatter, label='Cluster')
                                        st.pyplot(fig)
                                
                                elif len(features) == 3:
                                    # 3D plot
                                    try:
                                        import plotly.express as px
                                        
                                        fig = px.scatter_3d(
                                            clustered_df, 
                                            x=features[0], 
                                            y=features[1], 
                                            z=features[2],
                                            color='Cluster', 
                                            title=f'Gaussian Mixture ({features[0]}, {features[1]}, {features[2]})'
                                        )
                                        
                                        # Add component means
                                        means_df = pd.DataFrame(means, columns=features)
                                        means_df['Cluster'] = 'Mean'
                                        
                                        fig.add_scatter3d(
                                            x=means_df[features[0]],
                                            y=means_df[features[1]],
                                            z=means_df[features[2]],
                                            mode='markers',
                                            marker=dict(
                                                size=10,
                                                color='red',
                                                symbol='x'
                                            ),
                                            name='Component Means'
                                        )
                                        
                                        st.plotly_chart(fig)
                                    except ImportError:
                                        st.error("Plotly is required for 3D visualization")
                                else:
                                    # For more than 3 features, show pairplot
                                    if len(clustered_df) <= 1000:  # Limit for performance
                                        fig = plt.figure(figsize=(10, 8))
                                        sns.pairplot(
                                            clustered_df,
                                            vars=features[:4],  # Limit to first 4
                                            hue='Cluster',
                                            palette='viridis'
                                        )
                                        st.pyplot(fig)
                                    else:
                                        st.warning("Too many data points for pairplot. Showing scatter matrix instead.")
                                        # Sample for better visualization
                                        sample_df = clustered_df.sample(1000, random_state=42)
                                        fig = plt.figure(figsize=(10, 8))
                                        pd.plotting.scatter_matrix(
                                            sample_df[features[:4]],  # Limit to first 4
                                            c=sample_df['Cluster'],
                                            figsize=(12, 12),
                                            alpha=0.5,
                                            diagonal='kde'
                                        )
                                        st.pyplot(fig)
                                
                                # Model evaluation
                                st.write("**Model Evaluation**")
                                
                                # BIC and AIC
                                st.write(f"BIC: {gmm.bic(X_scaled):.4f}")
                                st.write(f"AIC: {gmm.aic(X_scaled):.4f}")
                                
                                # Silhouette score
                                from sklearn.metrics import silhouette_score
                                
                                silhouette_avg = silhouette_score(X_scaled, clusters)
                                st.write(f"Silhouette Score: {silhouette_avg:.4f}")
                                
                                # Show probabilities
                                st.write("**Probabilities Summary**")
                                prob_df = pd.DataFrame({
                                    'Min Probability': np.min(clustered_df['Probability']),
                                    'Mean Probability': np.mean(clustered_df['Probability']),
                                    'Median Probability': np.median(clustered_df['Probability']),
                                    'Max Probability': np.max(clustered_df['Probability'])
                                }, index=['Probability'])
                                
                                st.dataframe(prob_df)
                                
                                # Histogram of highest probability
                                fig, ax = plt.subplots(figsize=(10, 5))
                                sns.histplot(clustered_df['Probability'], bins=20, ax=ax)
                                plt.title('Distribution of Highest Component Probability')
                                plt.xlabel('Probability')
                                st.pyplot(fig)
        
        elif ml_task == "Dimensionality Reduction":
            st.markdown('<p class="highlight">Dimensionality Reduction</p>', unsafe_allow_html=True)
            
            if len(numeric_cols) < 3:
                st.error("Dimensionality reduction requires at least 3 numeric columns")
            else:
                # Feature selection
                features = st.multiselect(
                    "Select features for dimensionality reduction", 
                    numeric_cols,
                    default=numeric_cols
                )
                
                if len(features) < 3:
                    st.error("Please select at least 3 features")
                else:
                    # Method selection
                    method = st.selectbox(
                        "Dimensionality reduction method",
                        ["PCA", "t-SNE", "UMAP"]
                    )
                    
                    # Parameters based on method
                    if method == "PCA":
                        n_components = st.slider(
                            "Number of components", 
                            2, min(len(features), 10), 2
                        )
                        
                        if st.button("Perform PCA"):
                            with st.spinner("PCA in progress..."):
                                # Get data
                                X = df[features].dropna()
                                
                                # Normalize data
                                from sklearn.preprocessing import StandardScaler
                                scaler = StandardScaler()
                                X_scaled = scaler.fit_transform(X)
                                
                                # Apply PCA
                                from sklearn.decomposition import PCA
                                pca = PCA(n_components=n_components)
                                X_pca = pca.fit_transform(X_scaled)
                                
                                # Create dataframe with results
                                pca_df = pd.DataFrame(
                                    X_pca,
                                    columns=[f'PC{i+1}' for i in range(n_components)]
                                )
                                
                                # Add original index for potential joining
                                pca_df.index = X.index
                                
                                # Show results
                                st.write("**PCA Results**")
                                st.dataframe(pca_df.head())
                                
                                # Explained variance
                                explained_variance = pca.explained_variance_ratio_ * 100
                                cumulative_variance = np.cumsum(explained_variance)
                                
                                variance_df = pd.DataFrame({
                                    'Component': [f'PC{i+1}' for i in range(n_components)],
                                    'Explained Variance (%)': explained_variance,
                                    'Cumulative Variance (%)': cumulative_variance
                                })
                                
                                st.write("**Explained Variance**")
                                st.dataframe(variance_df)
                                
                                # Plot explained variance
                                fig, ax = plt.subplots(figsize=(10, 6))
                                
                                ax.bar(
                                    variance_df['Component'],
                                    variance_df['Explained Variance (%)'],
                                    alpha=0.7
                                )
                                ax.plot(
                                    variance_df['Component'],
                                    variance_df['Cumulative Variance (%)'],
                                    'ro-',
                                    alpha=0.7
                                )
                                
                                plt.ylabel('Explained Variance (%)')
                                plt.title('Explained Variance by Principal Components')
                                plt.axhline(y=90, color='r', linestyle='--', alpha=0.5)
                                plt.xticks(rotation=45)
                                plt.tight_layout()
                                
                                st.pyplot(fig)
                                
                                # Visualization
                                if n_components >= 2:
                                    # 2D scatter plot
                                    fig, ax = plt.subplots(figsize=(10, 8))
                                    
                                    sns.scatterplot(
                                        x='PC1', 
                                        y='PC2',
                                        data=pca_df,
                                        alpha=0.7,
                                        ax=ax
                                    )
                                    
                                    plt.title('PCA: First 2 Principal Components')
                                    plt.tight_layout()
                                    
                                    st.pyplot(fig)
                                
                                if n_components >= 3:
                                    # 3D scatter plot
                                    try:
                                        import plotly.express as px
                                        
                                        fig = px.scatter_3d(
                                            pca_df, 
                                            x='PC1', 
                                            y='PC2', 
                                            z='PC3',
                                            title='PCA: First 3 Principal Components'
                                        )
                                        
                                        st.plotly_chart(fig)
                                    except ImportError:
                                        st.error("Plotly is required for 3D visualization")
                                
                                # Loading scores
                                loadings = pd.DataFrame(
                                    pca.components_.T,
                                    columns=[f'PC{i+1}' for i in range(n_components)],
                                    index=features
                                )
                                
                                st.write("**Loading Scores**")
                                st.dataframe(loadings)
                                
                                # Plot loadings heatmap
                                fig, ax = plt.subplots(figsize=(10, max(8, len(features) * 0.3)))
                                
                                sns.heatmap(
                                    loadings,
                                    cmap='coolwarm',
                                    annot=True,
                                    fmt='.3f',
                                    ax=ax
                                )
                                
                                plt.title('PCA Loading Scores')
                                plt.tight_layout()
                                
                                st.pyplot(fig)
                                
                                # Biplot for first 2 components
                                if st.checkbox("Show biplot (first 2 PCs)"):
                                    # Create biplot
                                    fig, ax = plt.subplots(figsize=(12, 10))
                                    
                                    # Plot observations
                                    plt.scatter(
                                        X_pca[:, 0],
                                        X_pca[:, 1],
                                        alpha=0.5
                                    )
                                    
                                    # Plot feature vectors
                                    for i, feature in enumerate(features):
                                        plt.arrow(
                                            0, 0,
                                            pca.components_[0, i] * 3,
                                            pca.components_[1, i] * 3,
                                            head_width=0.1,
                                            head_length=0.1,
                                            fc='red',
                                            ec='red'
                                        )
                                        plt.text(
                                            pca.components_[0, i] * 3.2,
                                            pca.components_[1, i] * 3.2,
                                            feature,
                                            fontsize=12
                                        )
                                    
                                    # Plot aesthetics
                                    plt.axis('equal')
                                    plt.xlabel(f'PC1 ({explained_variance[0]:.2f}%)')
                                    plt.ylabel(f'PC2 ({explained_variance[1]:.2f}%)')
                                    plt.title('PCA Biplot')
                                    plt.grid(True)
                                    
                                    st.pyplot(fig)
                                
                                # Add download link for PCA results
                                csv = pca_df.to_csv().encode('utf-8')
                                st.download_button(
                                    "Download PCA results as CSV",
                                    csv,
                                    "pca_results.csv",
                                    "text/csv",
                                    key='download-pca-csv'
                                )
                    
                    elif method == "t-SNE":
                        perplexity = st.slider("Perplexity", 5, 50, 30)
                        n_iter = st.slider("Number of iterations", 250, 2000, 1000)
                        
                        if st.button("Perform t-SNE"):
                            with st.spinner("t-SNE in progress..."):
                                # Get data
                                X = df[features].dropna()
                                
                                # Normalize data
                                from sklearn.preprocessing import StandardScaler
                                scaler = StandardScaler()
                                X_scaled = scaler.fit_transform(X)
                                
                                # Apply t-SNE
                                from sklearn.manifold import TSNE
                                
                                tsne = TSNE(
                                    n_components=2,
                                    perplexity=perplexity,
                                    n_iter=n_iter,
                                    random_state=42
                                )
                                
                                X_tsne = tsne.fit_transform(X_scaled)
                                
                                # Create dataframe with results
                                tsne_df = pd.DataFrame(
                                    X_tsne,
                                    columns=['TSNE1', 'TSNE2']
                                )
                                
                                # Add original index for potential joining
                                tsne_df.index = X.index
                                
                                # Show results
                                st.write("**t-SNE Results**")
                                st.dataframe(tsne_df.head())
                                
                                # 2D scatter plot
                                fig, ax = plt.subplots(figsize=(10, 8))
                                
                                sns.scatterplot(
                                    x='TSNE1', 
                                    y='TSNE2',
                                    data=tsne_df,
                                    alpha=0.7,
                                    ax=ax
                                )
                                
                                plt.title('t-SNE Projection')
                                plt.tight_layout()
                                
                                st.pyplot(fig)
                                
                                # Allow coloring by a categorical variable
                                if categoric_cols:
                                    color_col = st.selectbox(
                                        "Color points by", 
                                        ['None'] + categoric_cols
                                    )
                                    
                                    if color_col != 'None':
                                        # Join with categorical column
                                        color_df = df[[color_col]].loc[X.index]
                                        plot_df = tsne_df.join(color_df)
                                        
                                        # Plot with coloring
                                        fig, ax = plt.subplots(figsize=(10, 8))
                                        
                                        sns.scatterplot(
                                            x='TSNE1', 
                                            y='TSNE2',
                                            hue=color_col,
                                            data=plot_df,
                                            alpha=0.7,
                                            ax=ax
                                        )
                                        
                                        plt.title(f't-SNE Projection (colored by {color_col})')
                                        plt.tight_layout()
                                        
                                        st.pyplot(fig)
                                
                                # Add download link for t-SNE results
                                csv = tsne_df.to_csv().encode('utf-8')
                                st.download_button(
                                    "Download t-SNE results as CSV",
                                    csv,
                                    "tsne_results.csv",
                                    "text/csv",
                                    key='download-tsne-csv'
                                )
                    
                    elif method == "UMAP":
                        n_neighbors = st.slider("Number of neighbors", 2, 100, 15)
                        min_dist = st.slider(
                            "Minimum distance", 
                            0.0, 1.0, 0.1, 
                            step=0.05
                        )
                        
                        if st.button("Perform UMAP"):
                            with st.spinner("UMAP in progress..."):
                                try:
                                    import umap
                                    
                                    # Get data
                                    X = df[features].dropna()
                                    
                                    # Normalize data
                                    from sklearn.preprocessing import StandardScaler
                                    scaler = StandardScaler()
                                    X_scaled = scaler.fit_transform(X)
                                    
                                    # Apply UMAP
                                    umap_model = umap.UMAP(
                                        n_neighbors=n_neighbors,
                                        min_dist=min_dist,
                                        n_components=2,
                                        random_state=42
                                    )
                                    
                                    X_umap = umap_model.fit_transform(X_scaled)
                                    
                                    # Create dataframe with results
                                    umap_df = pd.DataFrame(
                                        X_umap,
                                        columns=['UMAP1', 'UMAP2']
                                    )
                                    
                                    # Add original index for potential joining
                                    umap_df.index = X.index
                                    
                                    # Show results
                                    st.write("**UMAP Results**")
                                    st.dataframe(umap_df.head())
                                    
                                    # 2D scatter plot
                                    fig, ax = plt.subplots(figsize=(10, 8))
                                    
                                    sns.scatterplot(
                                        x='UMAP1', 
                                        y='UMAP2',
                                        data=umap_df,
                                        alpha=0.7,
                                        ax=ax
                                    )
                                    
                                    plt.title('UMAP Projection')
                                    plt.tight_layout()
                                    
                                    st.pyplot(fig)
                                    
                                    # Allow coloring by a categorical variable
                                    if categoric_cols:
                                        color_col = st.selectbox(
                                            "Color points by", 
                                            ['None'] + categoric_cols
                                        )
                                        
                                        if color_col != 'None':
                                            # Join with categorical column
                                            color_df = df[[color_col]].loc[X.index]
                                            plot_df = umap_df.join(color_df)
                                            
                                            # Plot with coloring
                                            fig, ax = plt.subplots(figsize=(10, 8))
                                            
                                            sns.scatterplot(
                                                x='UMAP1', 
                                                y='UMAP2',
                                                hue=color_col,
                                                data=plot_df,
                                                alpha=0.7,
                                                ax=ax
                                            )
                                            
                                            plt.title(f'UMAP Projection (colored by {color_col})')
                                            plt.tight_layout()
                                            
                                            st.pyplot(fig)
                                    
                                    # Add download link for UMAP results
                                    csv = umap_df.to_csv().encode('utf-8')
                                    st.download_button(
                                        "Download UMAP results as CSV",
                                        csv,
                                        "umap_results.csv",
                                        "text/csv",
                                        key='download-umap-csv'
                                    )
                                
                                except ImportError:
                                    st.error("Please install the umap-learn package: pip install umap-learn")
        
        elif ml_task == "Model Evaluation":
            st.markdown('<p class="highlight">Model Evaluation</p>', unsafe_allow_html=True)
            
            if st.session_state.ml_model is None:
                st.warning("No trained model found. Please train a model first in the Supervised Learning section.")
            else:
                # Get model info
                model_info = st.session_state.ml_model
                problem_type = model_info['problem_type']
                selected_model = model_info['selected_model']
                features = model_info['features']
                target = model_info['target']
                
                st.write(f"**Model**: {selected_model}")
                st.write(f"**Problem Type**: {problem_type}")
                st.write(f"**Target Variable**: {target}")
                st.write(f"**Features**: {', '.join(features)}")
                
                # Show evaluation metrics
                if problem_type == "Classification":
                    from sklearn.metrics import (
                        accuracy_score, precision_score, recall_score,
                        f1_score, classification_report, confusion_matrix,
                        roc_auc_score, roc_curve, precision_recall_curve
                    )
                    
                    X_test = model_info['X_test']
                    y_test = model_info['y_test']
                    y_pred = model_info['y_pred']
                    pipeline = model_info['pipeline']
                    
                    # Multi-class check
                    multi_class = len(np.unique(y_test)) > 2
                    
                    # Basic metrics
                    accuracy = accuracy_score(y_test, y_pred)
                    
                    if multi_class:
                        precision = precision_score(y_test, y_pred, average='weighted')
                        recall = recall_score(y_test, y_pred, average='weighted')
                        f1 = f1_score(y_test, y_pred, average='weighted')
                    else:
                        precision = precision_score(y_test, y_pred)
                        recall = recall_score(y_test, y_pred)
                        f1 = f1_score(y_test, y_pred)
                    
                    # Display metrics
                    metrics_df = pd.DataFrame({
                        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score'],
                        'Value': [accuracy, precision, recall, f1]
                    })
                    
                    st.write("**Evaluation Metrics**")
                    st.dataframe(metrics_df)
                    
                    # Classification report
                    st.write("**Classification Report**")
                    report = classification_report(y_test, y_pred, output_dict=True)
                    report_df = pd.DataFrame(report).transpose()
                    st.dataframe(report_df)
                    
                    # Confusion matrix
                    st.write("**Confusion Matrix**")
                    cm = confusion_matrix(y_test, y_pred)
                    
                    fig, ax = plt.subplots(figsize=(8, 6))
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
                    plt.xlabel('Predicted')
                    plt.ylabel('True')
                    st.pyplot(fig)
                    
                    # Advanced evaluation options
                    eval_option = st.selectbox(
                        "Advanced evaluation",
                        ["ROC Curve", "Precision-Recall Curve", "Learning Curve", "Validation Curve"]
                    )
                    
                    if eval_option == "ROC Curve":
                        # Check if probability predictions are available
                        if hasattr(pipeline, 'predict_proba'):
                            try:
                                y_prob = pipeline.predict_proba(X_test)
                                
                                if not multi_class:
                                    # Binary classification
                                    y_prob = y_prob[:, 1]
                                    fpr, tpr, _ = roc_curve(y_test, y_prob)
                                    auc = roc_auc_score(y_test, y_prob)
                                    
                                    fig, ax = plt.subplots(figsize=(8, 6))
                                    ax.plot(fpr, tpr, label=f'AUC = {auc:.4f}')
                                    ax.plot([0, 1], [0, 1], 'k--')
                                    plt.xlabel('False Positive Rate')
                                    plt.ylabel('True Positive Rate')
                                    plt.title('ROC Curve')
                                    plt.legend()
                                    st.pyplot(fig)
                                else:
                                    # Multi-class ROC curve (one-vs-rest)
                                    from sklearn.preprocessing import label_binarize
                                    from sklearn.metrics import auc
                                    
                                    classes = np.unique(y_test)
                                    n_classes = len(classes)
                                    
                                    # Binarize the labels
                                    y_test_bin = label_binarize(y_test, classes=classes)
                                    
                                    # Compute ROC curve and ROC area for each class
                                    fpr = {}
                                    tpr = {}
                                    roc_auc = {}
                                    
                                    for i in range(n_classes):
                                        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_prob[:, i])
                                        roc_auc[i] = auc(fpr[i], tpr[i])
                                    
                                    # Plot all ROC curves
                                    fig, ax = plt.subplots(figsize=(8, 6))
                                    
                                    for i in range(n_classes):
                                        plt.plot(
                                            fpr[i], tpr[i],
                                            label=f'Class {classes[i]} (AUC = {roc_auc[i]:.2f})'
                                        )
                                    
                                    plt.plot([0, 1], [0, 1], 'k--')
                                    plt.xlabel('False Positive Rate')
                                    plt.ylabel('True Positive Rate')
                                    plt.title('Multi-class ROC Curve (One-vs-Rest)')
                                    plt.legend()
                                    st.pyplot(fig)
                            except Exception as e:
                                st.error(f"Error generating ROC curve: {e}")
                        else:
                            st.error("This model doesn't support probability predictions, which are required for ROC curves")
                    
                    elif eval_option == "Precision-Recall Curve":
                        # Check if probability predictions are available
                        if hasattr(pipeline, 'predict_proba'):
                            try:
                                y_prob = pipeline.predict_proba(X_test)
                                
                                if not multi_class:
                                    # Binary classification
                                    y_prob = y_prob[:, 1]
                                    precision, recall, thresholds = precision_recall_curve(y_test, y_prob)
                                    
                                    # Calculate average precision
                                    from sklearn.metrics import average_precision_score
                                    ap = average_precision_score(y_test, y_prob)
                                    
                                    fig, ax = plt.subplots(figsize=(8, 6))
                                    ax.plot(recall, precision, label=f'AP = {ap:.4f}')
                                    
                                    # No skill line
                                    no_skill = sum(y_test == 1) / len(y_test)
                                    plt.plot([0, 1], [no_skill, no_skill], 'k--')
                                    
                                    plt.xlabel('Recall')
                                    plt.ylabel('Precision')
                                    plt.title('Precision-Recall Curve')
                                    plt.legend()
                                    st.pyplot(fig)
                                else:
                                    # Multi-class precision-recall curve (one-vs-rest)
                                    from sklearn.preprocessing import label_binarize
                                    from sklearn.metrics import auc, average_precision_score
                                    
                                    classes = np.unique(y_test)
                                    n_classes = len(classes)
                                    
                                    # Binarize the labels
                                    y_test_bin = label_binarize(y_test, classes=classes)
                                    
                                    # Compute precision-recall curve for each class
                                    precision = {}
                                    recall = {}
                                    avg_precision = {}
                                    
                                    for i in range(n_classes):
                                        precision[i], recall[i], _ = precision_recall_curve(y_test_bin[:, i], y_prob[:, i])
                                        avg_precision[i] = average_precision_score(y_test_bin[:, i], y_prob[:, i])
                                    
                                    # Plot all precision-recall curves
                                    fig, ax = plt.subplots(figsize=(8, 6))
                                    
                                    for i in range(n_classes):
                                        plt.plot(
                                            recall[i], precision[i],
                                            label=f'Class {classes[i]} (AP = {avg_precision[i]:.2f})'
                                        )
                                    
                                    plt.xlabel('Recall')
                                    plt.ylabel('Precision')
                                    plt.title('Multi-class Precision-Recall Curve (One-vs-Rest)')
                                    plt.legend()
                                    st.pyplot(fig)
                            except Exception as e:
                                st.error(f"Error generating Precision-Recall curve: {e}")
                        else:
                            st.error("This model doesn't support probability predictions, which are required for Precision-Recall curves")
                    
                    elif eval_option == "Learning Curve":
                        from sklearn.model_selection import learning_curve
                        
                        st.info("Generating learning curve... This may take a moment.")
                        
                        try:
                            pipeline = model_info['pipeline']
                            X = pd.concat([model_info['X_train'], model_info['X_test']])
                            y = pd.concat([model_info['y_train'], model_info['y_test']])
                            
                            # Define train sizes
                            train_sizes = np.linspace(0.1, 1.0, 10)
                            
                            # Calculate learning curve
                            train_sizes, train_scores, test_scores = learning_curve(
                                pipeline, X, y, 
                                train_sizes=train_sizes,
                                cv=5,
                                scoring='accuracy',
                                n_jobs=-1
                            )
                            
                            # Calculate mean and std
                            train_mean = np.mean(train_scores, axis=1)
                            train_std = np.std(train_scores, axis=1)
                            test_mean = np.mean(test_scores, axis=1)
                            test_std = np.std(test_scores, axis=1)
                            
                            # Plot learning curve
                            fig, ax = plt.subplots(figsize=(10, 6))
                            
                            plt.plot(train_sizes, train_mean, 'o-', color='r', label='Training score')
                            plt.plot(train_sizes, test_mean, 'o-', color='g', label='Cross-validation score')
                            
                            # Add error bands
                            plt.fill_between(
                                train_sizes, 
                                train_mean - train_std, 
                                train_mean + train_std, 
                                alpha=0.1, 
                                color='r'
                            )
                            plt.fill_between(
                                train_sizes, 
                                test_mean - test_std, 
                                test_mean + test_std, 
                                alpha=0.1, 
                                color='g'
                            )
                            
                            plt.xlabel('Training set size')
                            plt.ylabel('Accuracy Score')
                            plt.title('Learning Curve')
                            plt.legend(loc='best')
                            plt.grid(True)
                            
                            st.pyplot(fig)
                            
                            # Interpretation
                            st.write("**Interpretation**:")
                            gap = train_mean[-1] - test_mean[-1]
                            
                            if gap > 0.1:
                                st.write("- The model shows signs of **overfitting**. The training score is significantly higher than the cross-validation score.")
                                st.write("- Consider using regularization, reducing model complexity, or getting more training data.")
                            elif test_mean[-1] < 0.6:
                                st.write("- The model shows signs of **underfitting**. Both training and cross-validation scores are low.")
                                st.write("- Consider using a more complex model or adding more features.")
                            else:
                                st.write("- The model shows good fit with minimal overfitting.")
                            
                            if test_mean[-1] - test_mean[0] > 0.05:
                                st.write("- Adding more training data improved model performance, suggesting that collecting more data could help.")
                            else:
                                st.write("- Adding more training data did not significantly improve performance. Getting more data may not help.")
                                
                        except Exception as e:
                            st.error(f"Error generating learning curve: {e}")
                    
                    elif eval_option == "Validation Curve":
                        from sklearn.model_selection import validation_curve
                        
                        # Determine which parameter to use for validation curve
                        if selected_model == "Logistic Regression":
                            param_name = "model__C"
                            param_range = np.logspace(-3, 3, 10)
                            param_label = "Regularization parameter (C)"
                        elif selected_model == "Decision Tree" or selected_model == "Random Forest":
                            param_name = "model__max_depth"
                            param_range = np.array([1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 15, 20])
                            param_label = "Maximum tree depth"
                        elif selected_model == "Gradient Boosting":
                            param_name = "model__learning_rate"
                            param_range = np.logspace(-3, 0, 10)
                            param_label = "Learning rate"
                        elif selected_model == "Support Vector Machine":
                            param_name = "model__C"
                            param_range = np.logspace(-3, 3, 10)
                            param_label = "Regularization parameter (C)"
                        elif selected_model == "K-Nearest Neighbors":
                            param_name = "model__n_neighbors"
                            param_range = np.array([1, 3, 5, 7, 9, 11, 13, 15, 17, 19])
                            param_label = "Number of neighbors"
                        else:
                            st.error(f"Validation curve not implemented for {selected_model}")
                            comparison_type = st.radio("Comparison type",["Split one numeric column by category", "Compare two numeric columns"])

                        st.info(f"Generating validation curve for parameter: {param_name}")
                        
                        try:
                            pipeline = model_info['pipeline']
                            X = pd.concat([model_info['X_train'], model_info['X_test']])
                            y = pd.concat([model_info['y_train'], model_info['y_test']])
                            
                            # Calculate validation curve
                            train_scores, test_scores = validation_curve(
                                pipeline, X, y,
                                param_name=param_name,
                                param_range=param_range,
                                cv=5,
                                scoring='accuracy',
                                n_jobs=-1
                            )
                            
                            # Calculate mean and std
                            train_mean = np.mean(train_scores, axis=1)
                            train_std = np.std(train_scores, axis=1)
                            test_mean = np.mean(test_scores, axis=1)
                            test_std = np.std(test_scores, axis=1)
                            
                            # Plot validation curve
                            fig, ax = plt.subplots(figsize=(10, 6))
                            
                            plt.plot(param_range, train_mean, 'o-', color='r', label='Training score')
                            plt.plot(param_range, test_mean, 'o-', color='g', label='Cross-validation score')
                            
                            # Add error bands
                            plt.fill_between(
                                param_range, 
                                train_mean - train_std, 
                                train_mean + train_std, 
                                alpha=0.1, 
                                color='r'
                            )
                            plt.fill_between(
                                param_range, 
                                test_mean - test_std, 
                                test_mean + test_std, 
                                alpha=0.1, 
                                color='g'
                            )
                            
                            # Check if x-axis should be log scale
                            if np.all(np.diff(np.log10(param_range)) > 0):
                                plt.xscale('log')
                            
                            plt.xlabel(param_label)
                            plt.ylabel('Accuracy Score')
                            plt.title('Validation Curve')
                            plt.legend(loc='best')
                            plt.grid(True)
                            
                            st.pyplot(fig)
                            
                            # Find optimal parameter value
                            best_idx = np.argmax(test_mean)
                            best_param = param_range[best_idx]
                            best_score = test_mean[best_idx]
                            
                            st.write(f"**Optimal parameter value**: {best_param} (score: {best_score:.4f})")
                            
                            # Interpretation
                            st.write("**Interpretation**:")
                            
                            gap = train_mean[best_idx] - test_mean[best_idx]
                            
                            if gap > 0.1:
                                st.write(f"- At the optimal parameter value ({best_param}), the model still shows some overfitting.")
                            else:
                                st.write(f"- At the optimal parameter value ({best_param}), the model has a good fit.")
                            
                            # Check for underfitting/overfitting across the parameter range
                            if np.all(train_mean > test_mean + 0.1):
                                st.write("- The model tends to overfit across most parameter values.")
                            elif np.all(train_mean < 0.6) and np.all(test_mean < 0.6):
                                st.write("- The model tends to underfit across most parameter values.")
                                
                        except Exception as e:
                            st.error(f"Error generating validation curve: {e}")
                
                else:  # Regression
                    from sklearn.metrics import (
                        mean_squared_error, mean_absolute_error,
                        r2_score, explained_variance_score
                    )
                    
                    X_test = model_info['X_test']
                    y_test = model_info['y_test']
                    y_pred = model_info['y_pred']
                    pipeline = model_info['pipeline']
                    
                    # Calculate metrics
                    mse = mean_squared_error(y_test, y_pred)
                    rmse = np.sqrt(mse)
                    mae = mean_absolute_error(y_test, y_pred)
                    r2 = r2_score(y_test, y_pred)
                    evs = explained_variance_score(y_test, y_pred)
                    
                    # Display metrics
                    metrics_df = pd.DataFrame({
                        'Metric': ['MSE', 'RMSE', 'MAE', 'R²', 'Explained Variance'],
                        'Value': [mse, rmse, mae, r2, evs]
                    })
                    
                    st.write("**Evaluation Metrics**")
                    st.dataframe(metrics_df)
                    
                    # Predicted vs Actual plot
                    fig, ax = plt.subplots(figsize=(8, 6))
                    ax.scatter(y_test, y_pred, alpha=0.7)
                    
                    # Add perfect prediction line
                    max_val = max(y_test.max(), y_pred.max())
                    min_val = min(y_test.min(), y_pred.min())
                    ax.plot([min_val, max_val], [min_val, max_val], 'r--')
                    
                    plt.xlabel('Actual')
                    plt.ylabel('Predicted')
                    plt.title('Predicted vs Actual')
                    st.pyplot(fig)
                    
                    # Residual plot
                    residuals = y_test - y_pred
                    
                    fig, ax = plt.subplots(figsize=(8, 6))
                    ax.scatter(y_pred, residuals, alpha=0.7)
                    ax.axhline(y=0, color='r', linestyle='--')
                    
                    plt.xlabel('Predicted')
                    plt.ylabel('Residuals')
                    plt.title('Residual Plot')
                    st.pyplot(fig)
                    
                    # Advanced evaluation options
                    eval_option = st.selectbox(
                        "Advanced evaluation",
                        ["Residual Distribution", "Learning Curve", "Validation Curve"]
                    )
                    
                    if eval_option == "Residual Distribution":
                        fig, ax = plt.subplots(1, 2, figsize=(12, 5))
                        
                        # Histogram of residuals
                        sns.histplot(residuals, kde=True, ax=ax[0])
                        ax[0].set_title('Residual Distribution')
                        ax[0].set_xlabel('Residual')
                        
                        # Q-Q plot
                        from scipy import stats
                        stats.probplot(residuals, plot=ax[1])
                        ax[1].set_title('Q-Q Plot of Residuals')
                        
                        plt.tight_layout()
                        st.pyplot(fig)
                        
                        # Check for normality
                        from scipy.stats import shapiro
                        
                        # Limit to 5000 samples for shapiro-wilk test
                        test_residuals = residuals
                        if len(residuals) > 5000:
                            np.random.seed(42)
                            test_residuals = np.random.choice(residuals, 5000, replace=False)
                        
                        stat, p = shapiro(test_residuals)
                        
                        st.write("**Shapiro-Wilk Test for Normality of Residuals**")
                        st.write(f"W-statistic: {stat:.4f}")
                        st.write(f"p-value: {p:.4f}")
                        
                        if p < 0.05:
                            st.write("The residuals do not appear to be normally distributed (p < 0.05)")
                        else:
                            st.write("The residuals appear to be normally distributed (p >= 0.05)")
                        
                        # Breusch-Pagan test for heteroscedasticity
                        try:
                            from statsmodels.stats.diagnostic import het_breuschpagan
                            
                            # Reshape y_pred for the test
                            X_bp = y_pred.reshape(-1, 1)
                            
                            bp_test = het_breuschpagan(residuals, X_bp)
                            
                            st.write("**Breusch-Pagan Test for Heteroscedasticity**")
                            st.write(f"LM-statistic: {bp_test[0]:.4f}")
                            st.write(f"p-value: {bp_test[1]:.4f}")
                            
                            if bp_test[1] < 0.05:
                                st.write("There is evidence of heteroscedasticity (p < 0.05)")
                            else:
                                st.write("There is no significant evidence of heteroscedasticity (p >= 0.05)")
                        except:
                            st.info("Statsmodels package required for Breusch-Pagan test")
                    
                    elif eval_option == "Learning Curve":
                        from sklearn.model_selection import learning_curve
                        
                        st.info("Generating learning curve... This may take a moment.")
                        
                        try:
                            pipeline = model_info['pipeline']
                            X = pd.concat([model_info['X_train'], model_info['X_test']])
                            y = pd.concat([model_info['y_train'], model_info['y_test']])
                            
                            # Define train sizes
                            train_sizes = np.linspace(0.1, 1.0, 10)
                            
                            # Calculate learning curve
                            train_sizes, train_scores, test_scores = learning_curve(
                                pipeline, X, y, 
                                train_sizes=train_sizes,
                                cv=5,
                                scoring='r2',
                                n_jobs=-1
                            )
                            
                            # Calculate mean and std
                            train_mean = np.mean(train_scores, axis=1)
                            train_std = np.std(train_scores, axis=1)
                            test_mean = np.mean(test_scores, axis=1)
                            test_std = np.std(test_scores, axis=1)
                            
                            # Plot learning curve
                            fig, ax = plt.subplots(figsize=(10, 6))
                            
                            plt.plot(train_sizes, train_mean, 'o-', color='r', label='Training score')
                            plt.plot(train_sizes, test_mean, 'o-', color='g', label='Cross-validation score')
                            
                            # Add error bands
                            plt.fill_between(
                                train_sizes, 
                                train_mean - train_std, 
                                train_mean + train_std, 
                                alpha=0.1, 
                                color='r'
                            )
                            plt.fill_between(
                                train_sizes, 
                                test_mean - test_std, 
                                test_mean + test_std, 
                                alpha=0.1, 
                                color='g'
                            )
                            
                            plt.xlabel('Training set size')
                            plt.ylabel('R² Score')
                            plt.title('Learning Curve')
                            plt.legend(loc='best')
                            plt.grid(True)
                            
                            st.pyplot(fig)
                            
                            # Interpretation
                            st.write("**Interpretation**:")
                            gap = train_mean[-1] - test_mean[-1]
                            
                            if gap > 0.1:
                                st.write("- The model shows signs of **overfitting**. The training score is significantly higher than the cross-validation score.")
                                st.write("- Consider using regularization, reducing model complexity, or getting more training data.")
                            elif test_mean[-1] < 0.6:
                                st.write("- The model shows signs of **underfitting**. Both training and cross-validation scores are low.")
                                st.write("- Consider using a more complex model or adding more features.")
                            else:
                                st.write("- The model shows good fit with minimal overfitting.")
                            
                            if test_mean[-1] - test_mean[0] > 0.05:
                                st.write("- Adding more training data improved model performance, suggesting that collecting more data could help.")
                            else:
                                st.write("- Adding more training data did not significantly improve performance. Getting more data may not help.")
                                
                        except Exception as e:
                            st.error(f"Error generating learning curve: {e}")
                    
                    elif eval_option == "Validation Curve":
                        from sklearn.model_selection import validation_curve
                        
                        # Determine which parameter to use for validation curve
                        if selected_model == "Linear Regression":
                            param_name = "model__fit_intercept"
                            param_range = [True, False]
                            param_label = "Fit Intercept"
                        elif selected_model == "Decision Tree" or selected_model == "Random Forest":
                            param_name = "model__max_depth"
                            param_range = np.array([1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 15, 20])
                            param_label = "Maximum tree depth"
                        elif selected_model == "Gradient Boosting":
                            param_name = "model__learning_rate"
                            param_range = np.logspace(-3, 0, 10)
                            param_label = "Learning rate"
                        elif selected_model == "Support Vector Machine":
                            param_name = "model__C"
                            param_range = np.logspace(-3, 3, 10)
                            param_label = "Regularization parameter (C)"
                        elif selected_model == "K-Nearest Neighbors":
                            param_name = "model__n_neighbors"
                            param_range = np.array([1, 3, 5, 7, 9, 11, 13, 15, 17, 19])
                            param_label = "Number of neighbors"
                        else:
                            importance = pipeline.named_steps['model'].feature_importances_

                            st.error(f"Validation curve not implemented for {selected_model}")
                            # Create DataFrame (using the first N feature names that match the importance length)
                            feature_names[:len(importance)]
                            importance_df = pd.DataFrame({
                                'Feature': feature_names[:len(importance)],
                                'Importance': importance
                            }).sort_values('Importance', ascending=False)

                            st.dataframe(importance_df)

                        
                        st.info(f"Generating validation curve for parameter: {param_name}")
                        
                        try:
                            pipeline = model_info['pipeline']
                            X = pd.concat([model_info['X_train'], model_info['X_test']])
                            y = pd.concat([model_info['y_train'], model_info['y_test']])
                            
                            # Calculate validation curve
                            train_scores, test_scores = validation_curve(
                                pipeline, X, y,
                                param_name=param_name,
                                param_range=param_range,
                                cv=5,
                                scoring='r2',
                                n_jobs=-1
                            )
                            
                            # Calculate mean and std
                            train_mean = np.mean(train_scores, axis=1)
                            train_std = np.std(train_scores, axis=1)
                            test_mean = np.mean(test_scores, axis=1)
                            test_std = np.std(test_scores, axis=1)
                            
                            # Plot validation curve
                            fig, ax = plt.subplots(figsize=(10, 6))
                            
                            plt.plot(param_range, train_mean, 'o-', color='r', label='Training score')
                            plt.plot(param_range, test_mean, 'o-', color='g', label='Cross-validation score')
                            
                            # Add error bands
                            plt.fill_between(
                                param_range, 
                                train_mean - train_std, 
                                train_mean + train_std, 
                                alpha=0.1, 
                                color='r'
                            )
                            plt.fill_between(
                                param_range, 
                                test_mean - test_std, 
                                test_mean + test_std, 
                                alpha=0.1, 
                                color='g'
                            )
                            
                            # Check if x-axis should be log scale
                            if isinstance(param_range, np.ndarray) and param_range.dtype.kind == 'f' and np.all(np.diff(np.log10(param_range)) > 0):
                                plt.xscale('log')
                            
                            plt.xlabel(param_label)
                            plt.ylabel('R² Score')
                            plt.title('Validation Curve')
                            plt.legend(loc='best')
                            plt.grid(True)
                            
                            st.pyplot(fig)
                            
                            # Find optimal parameter value
                            best_idx = np.argmax(test_mean)
                            best_param = param_range[best_idx]
                            best_score = test_mean[best_idx]
                            
                            st.write(f"**Optimal parameter value**: {best_param} (score: {best_score:.4f})")
                            
                            # Interpretation
                            st.write("**Interpretation**:")
                            
                            gap = train_mean[best_idx] - test_mean[best_idx]
                            
                            if gap > 0.1:
                                st.write(f"- At the optimal parameter value ({best_param}), the model still shows some overfitting.")
                            else:
                                st.write(f"- At the optimal parameter value ({best_param}), the model has a good fit.")
                            
                            # Check for underfitting/overfitting across the parameter range
                            if np.all(train_mean > test_mean + 0.1):
                                st.write("- The model tends to overfit across most parameter values.")
                            elif np.all(train_mean < 0.6) and np.all(test_mean < 0.6):
                                st.write("- The model tends to underfit across most parameter values.")
                                
                        except Exception as e:
                            st.error(f"Error generating validation curve: {e}")

# Footer
st.markdown('---')
st.markdown('<p class="footer">Data Analysis Dashboard • Created with Streamlit • Last updated: April 5, 2025</p>', unsafe_allow_html=True)

