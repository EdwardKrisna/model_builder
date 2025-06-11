import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sqlalchemy import create_engine, text
import warnings
import traceback
from typing import Optional, Dict, List, Tuple, Any
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy import stats
import json
from datetime import datetime
import pickle
import io

import base64

import folium
from streamlit_folium import st_folium

# ML imports
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import LabelEncoder
import optuna

warnings.filterwarnings('ignore')

def evaluate(actual, predicted, squared=False, model=None):
    """
    Calculate various regression evaluation metrics, including FSD (Forecast Standard Deviation).

    Parameters:
        actual (array-like): The actual target values.
        predicted (array-like): The predicted target values.
        squared (bool): If True, will exponentiate actual and predicted before metric calculations.
        model (sklearn/base estimator, optional): Not used but for compatibility.

    Returns:
        dict: Dictionary of metrics.
    """
    # Make sure input is array-like
    actual = np.array(actual)
    predicted = np.array(predicted)
    if squared:
        actual = np.exp(actual)
        predicted = np.exp(predicted)

    # Calculate percentage error and absolute percentage error
    pe = (actual - predicted) / actual
    ape = np.abs(pe)
    n = len(actual)

    if n == 0:
        return {k: np.nan for k in ['MAE', 'MSE', 'R2', 'MAPE', 'FSD', 'PE10', 'RT20']}

    # Metrics
    mae = mean_absolute_error(actual, predicted)
    mse = mean_squared_error(actual, predicted)
    r2 = r2_score(actual, predicted)
    mape = np.median(ape)
    fsd = np.std(ape)
    r20 = np.sum(ape >= 0.2)
    r10 = np.sum(ape <= 0.1)
    rt20 = r20 / n
    pe10 = r10 / n

    metrics = {
        'MAE': mae,
        'MSE': mse,
        'R2': r2,
        'MAPE': mape,
        'FSD': fsd,
        'PE10': pe10,
        'RT20': rt20
    }
    return metrics

# Set page config
st.set_page_config(
    page_title="RHR MODEL BUILDER 𓀒 𓀓 𓀔",
    page_icon="🏠",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #3498db;
        margin: 0.5rem 0;
    }
    .success-box {
        background-color: #d4edda;
        color: #155724;
        padding: 0.75rem;
        border-radius: 0.25rem;
        border: 1px solid #c3e6cb;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        color: #856404;
        padding: 0.75rem;
        border-radius: 0.25rem;
        border: 1px solid #ffeaa7;
        margin: 1rem 0;
    }
    .error-box {
        background-color: #f8d7da;
        color: #721c24;
        padding: 0.75rem;
        border-radius: 0.25rem;
        border: 1px solid #f5c6cb;
        margin: 1rem 0;
    }
    .section-divider {
        border-top: 2px solid #e0e0e0;
        margin: 2rem 0;
        padding-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource(show_spinner=False)
def cached_connect_database():
    db_user = st.secrets["database"]["user"]
    db_password = st.secrets["database"]["password"]
    db_host = st.secrets["database"]["host"]
    db_port = st.secrets["database"]["port"]
    db_name = st.secrets["database"]["database"]

    connection_string = f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
    engine = create_engine(connection_string)
    with engine.connect() as conn:
        conn.execute(text("SELECT 1"))
    return engine

@st.cache_data
def cached_load_property_data(_engine):
    query = "SELECT * FROM engineered_property_data"
    df = pd.read_sql(query, _engine)
    return df

@st.cache_data
def cached_clean_data(df, cleaning_options):
    try:
        cleaned_df = df.copy()  # just copy the passed dataframe
        
        # Remove duplicates
        if cleaning_options.get('remove_duplicates', False):
            cleaned_df = cleaned_df.drop_duplicates(keep='first')
        
        # Handle missing values
        if cleaning_options.get('handle_missing', False):
            numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if cleaned_df[col].isnull().any():
                    median_val = cleaned_df[col].median()
                    cleaned_df[col].fillna(median_val, inplace=True)
            categorical_cols = cleaned_df.select_dtypes(include=['object', 'category']).columns
            for col in categorical_cols:
                if cleaned_df[col].isnull().any():
                    mode_val = cleaned_df[col].mode().iloc[0] if not cleaned_df[col].mode().empty else 'Unknown'
                    cleaned_df[col].fillna(mode_val, inplace=True)
        
        # Remove outliers
        if cleaning_options.get('remove_outliers', False) and cleaning_options.get('outlier_column'):
            outlier_col = cleaning_options['outlier_column']
            if outlier_col in cleaned_df.columns:
                Q1 = cleaned_df[outlier_col].quantile(0.25)
                Q3 = cleaned_df[outlier_col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                cleaned_df = cleaned_df[(cleaned_df[outlier_col] >= lower_bound) & (cleaned_df[outlier_col] <= upper_bound)]
        
        return cleaned_df  # return cleaned dataframe only
    
    except Exception as e:
        # You can raise or return None and handle error outside
        raise RuntimeError(f"Data cleaning failed: {str(e)}")


class RealEstateAnalyzer:
    """Flexible Real Estate Analysis System"""
    
    def __init__(self):
        self.engine = None
        self.connection_status = False
        self.original_data = None
        self.current_data = None
        self.model = None
        self.model_data = None
        self.transformed_columns = {}

    def connect_database(self):
        try:
            self.engine = cached_connect_database()
            self.connection_status = True
            return True, "Database connected successfully!"
        except Exception as e:
            self.connection_status = False
            return False, f"Database connection failed: {str(e)}"
    
    def load_property_data(self):
        try:
            self.original_data = cached_load_property_data(self.engine)
            self.current_data = self.original_data.copy()
            return True, f"Loaded {len(self.original_data)} properties successfully!"
        except Exception as e:
            return False, f"Failed to load data: {str(e)}"
    
    def reset_to_original(self):
        """Reset current data to original state"""
        if self.original_data is not None:
            self.current_data = self.original_data.copy()
            self.transformed_columns = {}
            return True, "Data reset to original state"
        return False, "No original data available"
    
    def change_dtype(self, column, new_dtype):
        """Change data type of a column"""
        try:
            if column not in self.current_data.columns:
                return False, f"Column {column} not found"
            
            if new_dtype == 'numeric':
                self.current_data[column] = pd.to_numeric(self.current_data[column], errors='coerce')
            elif new_dtype == 'categorical':
                self.current_data[column] = self.current_data[column].astype('category')
            elif new_dtype == 'string':
                self.current_data[column] = self.current_data[column].astype(str)
            elif new_dtype == 'datetime':
                self.current_data[column] = pd.to_datetime(self.current_data[column], errors='coerce')
            
            return True, f"Changed {column} to {new_dtype}"
            
        except Exception as e:
            return False, f"Failed to change dtype: {str(e)}"
    
    def apply_flexible_filters(self, filters):
        """Apply flexible filters based on column types"""
        try:
            filtered_df = self.current_data.copy()
            
            for column, filter_config in filters.items():
                if column not in filtered_df.columns:
                    continue
                
                filter_type = filter_config['type']
                filter_value = filter_config['value']
                
                if filter_type == 'categorical' and filter_value:
                    filtered_df = filtered_df[filtered_df[column].isin(filter_value)]
                
                elif filter_type == 'numeric_range' and filter_value:
                    min_val, max_val = filter_value
                    filtered_df = filtered_df[
                        (filtered_df[column] >= min_val) & 
                        (filtered_df[column] <= max_val)
                    ]
                
                elif filter_type == 'text_contains' and filter_value:
                    filtered_df = filtered_df[
                        filtered_df[column].str.contains(filter_value, case=False, na=False)
                    ]
            
            self.current_data = filtered_df
            return True, f"Filtered to {len(filtered_df)} properties"
            
        except Exception as e:
            return False, f"Filtering failed: {str(e)}"
    
    def clean_data(self, cleaning_options):
        if self.current_data is None:
            return False, "No data to clean"
        try:
            cleaned_df = cached_clean_data(self.current_data, cleaning_options)
            self.current_data = cleaned_df
            return True, f"Cleaned data: {len(self.current_data)} properties remaining"
        except Exception as e:
            return False, str(e)

    
    def apply_transformations(self, transformations):
        """Apply transformations and create new columns"""
        try:
            transformed_df = self.current_data.copy()
            
            for col, transform in transformations.items():
                if col in transformed_df.columns and transform != 'None':
                    if transform == 'log':
                        new_col = f'ln_{col}'
                        transformed_df[new_col] = np.log(transformed_df[col] + 1)  # Add 1 to handle zeros
                        self.transformed_columns[col] = new_col
                    elif transform == 'squared':
                        new_col = f'{col}_squared'
                        transformed_df[new_col] = transformed_df[col] ** 2
                        self.transformed_columns[col] = new_col
                    elif transform == 'sqrt':
                        new_col = f'sqrt_{col}'
                        transformed_df[new_col] = np.sqrt(np.abs(transformed_df[col]))
                        self.transformed_columns[col] = new_col
            
            self.current_data = transformed_df
            return True, f"Applied {len(transformations)} transformations"
            
        except Exception as e:
            return False, f"Transformation failed: {str(e)}"
    
    def apply_shortcut_filter(self, filter_name):
        """Apply predefined geographic filters"""
        try:
            # Find geographic columns (case insensitive)
            wadmpr_col = None
            wadmkk_col = None
            
            for col in self.current_data.columns:
                col_upper = col
                if 'wadmpr' in col_upper:
                    wadmpr_col = col
                elif 'wadmkk' in col_upper:
                    wadmkk_col = col
            
            if not wadmpr_col or not wadmkk_col:
                return False, "wadmpr or wadmkk columns not found"
            
            filtered_data = self.current_data.copy()
            
            if filter_name == 'bodebek':
                # BODEBEK: Bogor, Depok, Tangerang, Bekasi
                provinces = ['Jawa Barat', 'Banten']
                regencies = [
                    'Bogor', 'Kota Bogor',
                    'Depok', 'Kota Depok', 
                    'Tangerang', 'Kota Tangerang', 'Kota Tangerang Selatan',
                    'Bekasi', 'Kota Bekasi'
                ]
                
                filtered_data = filtered_data[
                    (filtered_data[wadmpr_col].isin(provinces)) &
                    (filtered_data[wadmkk_col].isin(regencies))
                ]
            
            elif filter_name == 'jabodetabek_no_kepulauan_seribu':
                # JABODETABEK without Kepulauan Seribu
                provinces = ['Jawa Barat', 'Banten', 'DKI Jakarta']
                regencies = [
                    'Kota Administrasi Jakarta Selatan', 'Kota Administrasi Jakarta Utara', 
                    'Kota Bekasi', 'Kota Depok', 'Kota Administrasi Jakarta Barat', 
                    'Bogor', 'Kota Administrasi Jakarta Pusat', 'Kota Bogor', 'Bekasi', 
                    'Tangerang', 'Kota Administrasi Jakarta Timur', 'Kota Tangerang', 
                    'Kota Tangerang Selatan'
                ]
                
                filtered_data = filtered_data[
                    (filtered_data[wadmpr_col].isin(provinces)) &
                    (filtered_data[wadmkk_col].isin(regencies))
                ]
            
            elif filter_name == 'jabodetabek':
                # JABODETABEK with Kepulauan Seribu
                provinces = ['Jawa Barat', 'Banten', 'DKI Jakarta']
                regencies = [
                    'Kota Administrasi Jakarta Selatan', 'Kota Administrasi Jakarta Utara', 
                    'Kota Bekasi', 'Kota Depok', 'Kota Administrasi Jakarta Barat', 
                    'Bogor', 'Kota Administrasi Jakarta Pusat', 'Kota Bogor', 'Bekasi', 
                    'Tangerang', 'Kota Administrasi Jakarta Timur', 'Kota Tangerang', 
                    'Kota Tangerang Selatan', 'Administrasi Kepulauan Seribu'
                ]
                
                filtered_data = filtered_data[
                    (filtered_data[wadmpr_col].isin(provinces)) &
                    (filtered_data[wadmkk_col].isin(regencies))
                ]
            
            elif filter_name == 'bandung':
                # Bandung Metropolitan Area
                provinces = ['Jawa Barat']
                regencies = [
                    'Kota Bandung', 'Sumedang', 'Kota Cimahi', 'Bandung', 'Bandung Barat'
                ]
                
                filtered_data = filtered_data[
                    (filtered_data[wadmpr_col].isin(provinces)) &
                    (filtered_data[wadmkk_col].isin(regencies))
                ]
            
            elif filter_name == 'bali':
                # Bali Metropolitan Area
                provinces = ['Bali']
                regencies = [
                    'Kota Denpasar', 'Badung', 'Gianyar', 'Tabanan'
                ]
                
                filtered_data = filtered_data[
                    (filtered_data[wadmpr_col].isin(provinces)) &
                    (filtered_data[wadmkk_col].isin(regencies))
                ]
            
            elif filter_name == 'surabaya':
                # Surabaya Metropolitan Area
                provinces = ['Jawa Timur']
                regencies = [
                    'Kota Surabaya'  # More can be added later
                ]
                
                filtered_data = filtered_data[
                    (filtered_data[wadmpr_col].isin(provinces)) &
                    (filtered_data[wadmkk_col].isin(regencies))
                ]
            
            self.current_data = filtered_data
            return True, f"Applied {filter_name} filter: {len(filtered_data):,} properties"
            
        except Exception as e:
            return False, f"Shortcut filter failed: {str(e)}"
        
    def apply_label_encoding(self, column):    
        """Apply label encoding to a column"""
        try:
            if column not in self.current_data.columns:
                return False, f"Column {column} not found"
            
            # Create encoded column
            label_encoder = LabelEncoder()
            encoded_column = f"{column}_encoded"
            self.current_data[encoded_column] = label_encoder.fit_transform(self.current_data[column].astype(str))
            
            return True, f"Created encoded column: {encoded_column}"
            
        except Exception as e:
            return False, f"Label encoding failed: {str(e)}"
        
    def calculate_vif(self, X_df, drop_const=True):
        """Calculate Variance Inflation Factor"""
        try:
            if drop_const and 'const' in X_df.columns:
                X_df = X_df.drop(columns='const')
            
            vif_data = pd.DataFrame()
            vif_data['feature'] = X_df.columns
            vif_data['VIF'] = [
                variance_inflation_factor(X_df.values, i) for i in range(X_df.shape[1])
            ]
            return vif_data
        except Exception as e:
            st.error(f"VIF calculation failed: {str(e)}")
            return None
    
    def run_ols_model(self, y_column, x_columns):
        """Run OLS regression model with transformed columns"""
        try:
            # Use transformed column names if available
            actual_y = self.transformed_columns.get(y_column, y_column)
            actual_x = [self.transformed_columns.get(col, col) for col in x_columns]
            
            # Prepare the data
            model_vars = [actual_y] + actual_x
            df_model = self.current_data[model_vars].dropna()
            
            # Prepare X and y
            X = df_model[actual_x]
            y = df_model[actual_y]
            
            # Add constant
            X = sm.add_constant(X)
            
            # Fit the model
            model = sm.OLS(y, X).fit(cov_type='HC3')
            
            self.model = model
            self.model_data = df_model
            
            # Calculate VIF
            vif_df = self.calculate_vif(X, drop_const=True)
            
            return True, f"OLS model fitted successfully with {len(df_model)} observations", vif_df
            
        except Exception as e:
            return False, f"OLS modeling failed: {str(e)}", None
    
    def get_model_results(self):
        """Get comprehensive model results"""
        if self.model is None:
            return None
        
        results = {
            'summary': str(self.model.summary()),
            'rsquared': self.model.rsquared,
            'rsquared_adj': self.model.rsquared_adj,
            'fvalue': self.model.fvalue,
            'f_pvalue': self.model.f_pvalue,
            'aic': self.model.aic,
            'bic': self.model.bic,
            'params': self.model.params,
            'pvalues': self.model.pvalues,
            'conf_int': self.model.conf_int(),
            'residuals': self.model.resid,
            'fitted_values': self.model.fittedvalues,
            'actual_values': self.model.model.endog
        }
        
        return results
    
    def goval_machine_learning(self, X, y, algorithm, group=None, n_splits=10, random_state=101, min_sample=3):
        """
        Group-based machine learning with evaluation. Falls back to standard CV if group=None.
        """
        data = self.current_data

        # --- If group is None: standard KFold cross-validation ---
        if group is None:
            from sklearn.model_selection import KFold

            evaluation_results = {'Fold': [], 'R2': [], 'FSD': [], 'PE10': [], 'RT20': []}
            train_results = {'R2': [], 'FSD': [], 'PE10': [], 'RT20': []}

            global_train_metrics = {'R2': 0, 'FSD': 0, 'PE10': 0, 'RT20': 0}
            global_test_metrics = {'R2': 0, 'FSD': 0, 'PE10': 0, 'RT20': 0}

            kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

            for fold, (train_idx, test_idx) in enumerate(kf.split(data[X])):
                X_train, X_test = data.iloc[train_idx][X], data.iloc[test_idx][X]
                y_train, y_test = data.iloc[train_idx][y], data.iloc[test_idx][y]

                model = algorithm.fit(X_train, y_train)
                y_train_pred = model.predict(X_train)
                y_test_pred = model.predict(X_test)

                train_metrics = evaluate(y_train, y_train_pred, squared=True)
                test_metrics = evaluate(y_test, y_test_pred, squared=True)

                for metric in global_train_metrics.keys():
                    global_train_metrics[metric] += train_metrics[metric]
                    global_test_metrics[metric] += test_metrics[metric]

                train_results['R2'].append(train_metrics['R2'])
                train_results['FSD'].append(train_metrics['FSD'])
                train_results['PE10'].append(train_metrics['PE10'])
                train_results['RT20'].append(train_metrics['RT20'])

                evaluation_results['Fold'].append(f"Fold-{fold + 1}")
                evaluation_results['R2'].append(test_metrics['R2'])
                evaluation_results['FSD'].append(test_metrics['FSD'])
                evaluation_results['PE10'].append(test_metrics['PE10'])
                evaluation_results['RT20'].append(test_metrics['RT20'])

            for metric in global_train_metrics.keys():
                global_train_metrics[metric] /= n_splits
                global_test_metrics[metric] /= n_splits

            evaluation_df = pd.DataFrame(evaluation_results)
            train_results_df = pd.DataFrame(train_results)

            return (
                model, evaluation_df, train_results_df,
                global_train_metrics, global_test_metrics,
                y_test, y_test_pred, False
            )

        # --- If group is provided: use your original group-based logic ---
        else:
            evaluation_results = {'Fold': [], 'R2': [], 'FSD': [], 'PE10': [], 'RT20': []}
            train_results = {'R2': [], 'FSD': [], 'PE10': [], 'RT20': []}

            global_train_metrics = {'R2': 0, 'FSD': 0, 'PE10': 0, 'RT20': 0}
            global_test_metrics = {'R2': 0, 'FSD': 0, 'PE10': 0, 'RT20': 0}

            for fold in range(n_splits):
                X_train_, X_test_ = [], []
                y_train_, y_test_ = [], []

                for group_value in data[group].unique():
                    data_group = data[data[group] == group_value]

                    if len(data_group) > min_sample:
                        X_group = data_group[X]
                        y_group = data_group[y]
                        X_train, X_test, y_train, y_test = train_test_split(
                            X_group, y_group, test_size=0.33, random_state=random_state + fold
                        )
                        X_train_.append(X_train)
                        X_test_.append(X_test)
                        y_train_.append(y_train)
                        y_test_.append(y_test)
                    else:
                        X_train_.append(data_group[X])
                        y_train_.append(data_group[y])

                X_train_all = pd.concat(X_train_)
                X_test_all = pd.concat(X_test_)
                y_train_all = pd.concat(y_train_)
                y_test_all = pd.concat(y_test_)

                # Train the model
                model = algorithm.fit(X_train_all, y_train_all)

                # Predict on train and test sets
                y_train_pred = model.predict(X_train_all)
                y_test_pred = model.predict(X_test_all)

                # Evaluate train and test scores
                train_metrics = evaluate(y_train_all, y_train_pred, squared=True)
                test_metrics = evaluate(y_test_all, y_test_pred, squared=True)

                for metric in global_train_metrics.keys():
                    global_train_metrics[metric] += train_metrics[metric]
                    global_test_metrics[metric] += test_metrics[metric]

                train_results['R2'].append(train_metrics['R2'])
                train_results['FSD'].append(train_metrics['FSD'])
                train_results['PE10'].append(train_metrics['PE10'])
                train_results['RT20'].append(train_metrics['RT20'])

                evaluation_results['Fold'].append(f"Fold-{fold + 1}")
                evaluation_results['R2'].append(test_metrics['R2'])
                evaluation_results['FSD'].append(test_metrics['FSD'])
                evaluation_results['PE10'].append(test_metrics['PE10'])
                evaluation_results['RT20'].append(test_metrics['RT20'])

            for metric in global_train_metrics.keys():
                global_train_metrics[metric] /= n_splits
                global_test_metrics[metric] /= n_splits

            evaluation_df = pd.DataFrame(evaluation_results)
            train_results_df = pd.DataFrame(train_results)

            return (
                model, evaluation_df, train_results_df,
                global_train_metrics, global_test_metrics,
                y_test_all, y_test_pred, False
            )

# Initialize session state
if 'analyzer' not in st.session_state:
    st.session_state.analyzer = RealEstateAnalyzer()

if 'processing_step' not in st.session_state:
    st.session_state.processing_step = 'overview'

# Get analyzer from session state
analyzer = st.session_state.analyzer

# Main App Header
st.markdown("""
<h1 style="display: flex; align-items: center; gap: 10px; line-height: 1;">
  <img src="https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExdWUybHNwbXV3cjd5eG9vbXBxcGs4Y3g1Y29neDdwNDQ2emdhYjd3cSZlcD12MV9naWZzX3NlYXJjaCZjdD1n/uSGobo6DnKmfqYygyk/giphy.gif" 
       alt="GIF" 
       style="height: 90px;">
  <span style="display: inline-block; vertical-align: middle;">RHR MODEL BUILDER 𓀒 𓀓 𓀔</span>
</h1>
""", unsafe_allow_html=True)


# Navigation buttons
st.markdown("### 🧭 Analysis Workflow")
workflow_steps = [
    ('overview', '📊 Data Overview'),
    ('dtype', '🔧 Data Types'),
    ('filter', '🔍 Filtering'),
    ('clean', '🧹 Cleaning'),
    ('transform', '⚡ Transform'),
    ('model', '📈 OLS Model'),
    ('ml', '🤖 ML Models')
]

cols = st.columns(len(workflow_steps))
for i, (step_key, step_name) in enumerate(workflow_steps):
    with cols[i]:
        if st.button(step_name, key=f"nav_{step_key}", 
                    type="primary" if st.session_state.processing_step == step_key else "secondary"):
            st.session_state.processing_step = step_key

# Auto-connect to database on first load
if not analyzer.connection_status:
    with st.spinner("Connecting to database..."):
        success, message = analyzer.connect_database()
        if success:
            st.markdown(f'<div class="success-box">✅ {message}</div>', unsafe_allow_html=True)
            # Auto-load data
            with st.spinner("Loading property data..."):
                data_success, data_message = analyzer.load_property_data()
                if data_success:
                    st.markdown(f'<div class="success-box">📊 {data_message}</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="error-box">❌ {data_message}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="error-box">❌ {message}</div>', unsafe_allow_html=True)
            st.stop()

# Display current data status
if analyzer.current_data is not None:
    st.markdown("### 📋 Current Data Status")
    # Create 5 columns: 4 for metrics, 1 for download button
    col1, col2, col3, col4, col5 = st.columns([1,1,1,1,1.5])
    
    with col1:
        st.metric("Properties", f"{len(analyzer.current_data):,}")
    with col2:
        st.metric("Columns", len(analyzer.current_data.columns))
    with col3:
        missing_pct = (analyzer.current_data.isnull().sum().sum() / 
                      (analyzer.current_data.shape[0] * analyzer.current_data.shape[1]) * 100)
        st.metric("Completeness", f"{100-missing_pct:.1f}%")
    with col4:
        st.metric("Memory", f"{analyzer.current_data.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
    
    if 'excel_data' not in st.session_state:
        st.session_state.excel_data = None

    with col5:
        if st.button("💾 Prepare Download Excel"):
            with st.spinner("Preparing Excel data..."):
                towrite = io.BytesIO()
                with pd.ExcelWriter(towrite, engine='xlsxwriter') as writer:
                    analyzer.current_data.to_excel(writer, index=False, sheet_name='CurrentData')
                towrite.seek(0)
                st.session_state.excel_data = towrite.read()
            st.success("Excel data ready for download! Click Download 👇🏼")

        if st.session_state.excel_data is not None:
            st.download_button(
                label="Download Data (.xlsx)",
                data=st.session_state.excel_data,
                file_name=f"current_data_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

# Reset button
if st.button("🔄 Reset to Original Data", help="Reset all changes and start fresh"):
    success, message = analyzer.reset_to_original()
    if success:
        st.success(message)
        st.rerun()

st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

# Step-based interface
if st.session_state.processing_step == 'overview':
    st.markdown('## <img src="https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExaXZod3R3NnJ2cW93MjkycXJ3dTRxeHluYXlkemhwdnVyZTFmOWhibyZlcD12MV9naWZzX3RyZW5kaW5nJmN0PWc/0GtVKtagi2GvWuY3vm/giphy.gif" alt="data gif" style="height:96px; vertical-align:middle;"> Data Overview', unsafe_allow_html=True)

    if analyzer.current_data is not None:
        # Data preview
        st.markdown("### 📋 Data Preview")
        st.dataframe(analyzer.current_data.head(10), use_container_width=True)
        
        # Basic statistics for numeric columns
        numeric_data = analyzer.current_data.select_dtypes(include=[np.number])
        if not numeric_data.empty:
            st.markdown("### 📈 Numeric Columns Statistics")
            st.dataframe(numeric_data.describe(), use_container_width=True)
        
        # Data info
        st.markdown("### ℹ️ Column Information")
        info_df = pd.DataFrame({
            'Column': analyzer.current_data.columns,
            'Data Type': [str(dtype) for dtype in analyzer.current_data.dtypes],
            'Non-Null Count': analyzer.current_data.count(),
            'Null Count': analyzer.current_data.isnull().sum(),
            'Unique Values': [analyzer.current_data[col].nunique() for col in analyzer.current_data.columns]
        })
        st.dataframe(info_df, use_container_width=True)

elif st.session_state.processing_step == 'dtype':
    st.markdown('## <img src="https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExeDltcWZoZ3dsNTVzZm5xMWR5bXExbGx0cG14eWdudGdpanJtZjBnMCZlcD12MV9naWZzX3NlYXJjaCZjdD1n/xCCqt6qDewWf6zriPX/giphy.gif" alt="data gif" style="height:72px; vertical-align:middle;"> Data Type Management', unsafe_allow_html=True)
    
    if analyzer.current_data is not None:
        st.markdown("### Change Column Data Types")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            selected_column = st.selectbox("Select Column", analyzer.current_data.columns)
        
        with col2:
            current_dtype = str(analyzer.current_data[selected_column].dtype)
            st.info(f"Current type: {current_dtype}")
            
            new_dtype = st.selectbox("New Data Type", 
                                   ['numeric', 'categorical', 'string', 'datetime'])
        
        with col3:
            st.write("")
            st.write("")
            if st.button("Apply Type Change", type="primary"):
                success, message = analyzer.change_dtype(selected_column, new_dtype)
                if success:
                    st.success(message)
                    st.rerun()
                else:
                    st.error(message)
        
        # Show current data types
        st.markdown("### Current Data Types")
        dtype_df = pd.DataFrame({
            'Column': analyzer.current_data.columns,
            'Data Type': [str(dtype) for dtype in analyzer.current_data.dtypes],
            'Sample Values': [str(analyzer.current_data[col].dropna().iloc[0]) if not analyzer.current_data[col].dropna().empty else 'N/A' 
                            for col in analyzer.current_data.columns]
        })
        st.dataframe(dtype_df, use_container_width=True)

elif st.session_state.processing_step == 'filter':
    st.markdown('## <img src="https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExd3NmZDV4d3Q2Zm00dXNuZ3J5bDZ4ZmVvYWFhaW1wbWdsNGJxeTZ5OCZlcD12MV9naWZzX3NlYXJjaCZjdD1n/eEjf3t9MeTXZM0d91u/giphy.gif" alt="data gif" style="height:72px; vertical-align:middle;"> Geographic & Data Filtering', unsafe_allow_html=True)

    if analyzer.current_data is not None:
        # Shortcut Geographic Filters
        st.markdown("### 🚀 Quick Geographic Filters")
        st.info("Pre-defined metropolitan area filters for quick analysis")
        
        shortcut_filters = {
            'bodebek': '🏙️ BODEBEK (Bogor, Depok, Tangerang, Bekasi)',
            'jabodetabek': '🌆 JABODETABEK (Jakarta + surrounding areas)',
            'jabodetabek_no_kepulauan_seribu': '🌆 JABODETABEK (No Kepulauan Seribu)',
            'bandung': '🏔️ Bandung Metropolitan Area',
            'bali': '🏝️ Bali Metropolitan Area',
            'surabaya': '🏢 Surabaya Metropolitan Area'
        }
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button(shortcut_filters['bodebek'], use_container_width=True):
                success, message = analyzer.apply_shortcut_filter('bodebek')
                if success:
                    st.success(message)
                    st.session_state.filter_applied = True
                    st.rerun()
                else:
                    st.error(message)
            
            if st.button(shortcut_filters['jabodetabek'], use_container_width=True):
                success, message = analyzer.apply_shortcut_filter('jabodetabek')
                if success:
                    st.success(message)
                    st.session_state.filter_applied = True
                    st.rerun()
                else:
                    st.error(message)
        
        with col2:
            if st.button(shortcut_filters['jabodetabek_no_kepulauan_seribu'], use_container_width=True):
                success, message = analyzer.apply_shortcut_filter('jabodetabek_no_kepulauan_seribu')
                if success:
                    st.success(message)
                    st.session_state.filter_applied = True
                    st.rerun()
                else:
                    st.error(message)
            
            if st.button(shortcut_filters['bandung'], use_container_width=True):
                success, message = analyzer.apply_shortcut_filter('bandung')
                if success:
                    st.success(message)
                    st.session_state.filter_applied = True
                    st.rerun()
                else:
                    st.error(message)
        
        with col3:
            if st.button(shortcut_filters['bali'], use_container_width=True):
                success, message = analyzer.apply_shortcut_filter('bali')
                if success:
                    st.success(message)
                    st.session_state.filter_applied = True
                    st.rerun()
                else:
                    st.error(message)
            
            if st.button(shortcut_filters['surabaya'], use_container_width=True):
                success, message = analyzer.apply_shortcut_filter('surabaya')
                if success:
                    st.success(message)
                    st.session_state.filter_applied = True
                    st.rerun()
                else:
                    st.error(message)
        
        st.markdown("---")
        
        # Geographic Filtering (Priority Section)
        st.markdown("### 🗺️ Manual Geographic Filtering")
        
        geo_columns = ['wadmpr', 'wadmkk', 'wadmkc', 'wadmkd']
        available_geo_cols = []
        
        # Find available geographic columns (case insensitive)
        for col in analyzer.current_data.columns:
            col_upper = col
            for geo in geo_columns:
                if geo in col_upper:
                    available_geo_cols.append(col)
                    break
        
        selected_filters = {}  # Initialize selected_filters here
        
        if available_geo_cols:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                # Find wadmpr column
                wadmpr_col = None
                for col in available_geo_cols:
                    if 'wadmpr' in col:
                        wadmpr_col = col
                        break
                
                if wadmpr_col:
                    wadmpr_options = sorted(analyzer.current_data[wadmpr_col].dropna().unique())
                    selected_wadmpr = st.multiselect(
                        f"Select Province ({wadmpr_col})", 
                        wadmpr_options,
                        default=wadmpr_options[:5] if len(wadmpr_options) > 5 else wadmpr_options,
                        key="geo_wadmpr"
                    )
                    if selected_wadmpr:
                        selected_filters[wadmpr_col] = {'type': 'categorical', 'value': selected_wadmpr}
            
            with col2:
                # Find wadmkk column
                wadmkk_col = None
                for col in available_geo_cols:
                    if 'wadmkk' in col:
                        wadmkk_col = col
                        break
                
                if wadmkk_col:
                    # Filter wadmkk based on selected wadmpr
                    if wadmpr_col and selected_wadmpr:
                        available_wadmkk = analyzer.current_data[
                            analyzer.current_data[wadmpr_col].isin(selected_wadmpr)
                        ][wadmkk_col].dropna().unique()
                    else:
                        available_wadmkk = analyzer.current_data[wadmkk_col].dropna().unique()
                    
                    available_wadmkk = sorted(available_wadmkk)
                    selected_wadmkk = st.multiselect(
                        f"Select Regency/City ({wadmkk_col})", 
                        available_wadmkk,
                        default=available_wadmkk[:10] if len(available_wadmkk) > 10 else available_wadmkk,
                        key="geo_wadmkk"
                    )
                    if selected_wadmkk:
                        selected_filters[wadmkk_col] = {'type': 'categorical', 'value': selected_wadmkk}
            
            with col3:
                # Find wadmkc column
                wadmkc_col = None
                for col in available_geo_cols:
                    if 'wadmkc' in col:
                        wadmkc_col = col
                        break
                
                if wadmkc_col:
                    # Filter wadmkc based on selected wadmkk
                    if wadmkk_col and selected_wadmkk:
                        available_wadmkc = analyzer.current_data[
                            analyzer.current_data[wadmkk_col].isin(selected_wadmkk)
                        ][wadmkc_col].dropna().unique()
                    else:
                        available_wadmkc = analyzer.current_data[wadmkc_col].dropna().unique()
                    
                    available_wadmkc = sorted(available_wadmkc)
                    selected_wadmkc = st.multiselect(
                        f"Select District ({wadmkc_col})", 
                        available_wadmkc,
                        default=available_wadmkc[:10] if len(available_wadmkc) > 10 else available_wadmkc,
                        key="geo_wadmkc"
                    )
                    if selected_wadmkc:
                        selected_filters[wadmkc_col] = {'type': 'categorical', 'value': selected_wadmkc}
        
        # Additional Flexible Filtering
        st.markdown("### 🔍 Additional Filters")
        
        # Select columns to filter (exclude geographic columns already handled)
        non_geo_columns = [col for col in analyzer.current_data.columns 
                          if not any(geo in col for geo in geo_columns)]
        filter_columns = st.multiselect("Select Additional Columns to Filter", 
                                       non_geo_columns, 
                                       key="additional_filter_columns")
        
        for col in filter_columns:
            st.markdown(f"#### Filter: {col}")
            col_dtype = analyzer.current_data[col].dtype
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Determine filter type based on data type and uniqueness
                unique_count = analyzer.current_data[col].nunique()
                
                if col_dtype in ['object', 'category'] or unique_count <= 20:
                    # Categorical filter
                    st.write("**Filter Type: Categorical**")
                    unique_values = sorted(analyzer.current_data[col].dropna().unique())
                    selected_values = st.multiselect(f"Select {col} values", 
                                                   unique_values, 
                                                   default=unique_values,
                                                   key=f"cat_filter_{col}")
                    if selected_values:
                        selected_filters[col] = {'type': 'categorical', 'value': selected_values}
                
                elif np.issubdtype(col_dtype, np.number):
                    # Numeric range filter
                    st.write("**Filter Type: Numeric Range**")
                    min_val = float(analyzer.current_data[col].min())
                    max_val = float(analyzer.current_data[col].max())
                    
                    range_values = st.slider(f"{col} range", 
                                           min_value=min_val, 
                                           max_value=max_val, 
                                           value=(min_val, max_val),
                                           key=f"num_filter_{col}")
                    
                    if range_values != (min_val, max_val):
                        selected_filters[col] = {'type': 'numeric_range', 'value': range_values}
                
                else:
                    # Text contains filter
                    st.write("**Filter Type: Text Contains**")
                    text_filter = st.text_input(f"Text to search in {col}", 
                                               key=f"text_filter_{col}")
                    if text_filter:
                        selected_filters[col] = {'type': 'text_contains', 'value': text_filter}
            
            with col2:
                # Show column statistics
                st.write("**Column Statistics:**")
                st.write(f"- Data Type: {col_dtype}")
                st.write(f"- Unique Values: {unique_count:,}")
                st.write(f"- Missing Values: {analyzer.current_data[col].isnull().sum():,}")
                
                if np.issubdtype(col_dtype, np.number):
                    st.write(f"- Min: {analyzer.current_data[col].min():.2f}")
                    st.write(f"- Max: {analyzer.current_data[col].max():.2f}")
                    st.write(f"- Mean: {analyzer.current_data[col].mean():.2f}")
        
        # Apply filters
        if selected_filters and st.button("🔍 Apply Manual Filters", type="primary"):
            with st.spinner("Applying filters..."):
                success, message = analyzer.apply_flexible_filters(selected_filters)
                if success:
                    st.success(message)
                    st.session_state.filter_applied = True
                    st.rerun()
                else:
                    st.error(message)
        
        # Show active filters
        if selected_filters:
            st.markdown("### Active Manual Filters")
            for col, filter_config in selected_filters.items():
                if any(geo in col for geo in geo_columns):
                    st.write(f"🗺️ **{col}:** {len(filter_config['value'])} selected")
                else:
                    st.write(f"**{col}:** {filter_config['type']} = {filter_config['value']}")
        
        # Map Visualization (Only after filtering)
        if st.session_state.get('filter_applied', False) or len(analyzer.current_data) < len(analyzer.original_data):
            st.markdown("---")
            st.markdown("### 🗺️ Property Location Map")
            
            # Check for latitude and longitude columns
            lat_col = None
            lon_col = None
            
            for col in analyzer.current_data.columns:
                col_lower = col.lower()
                if 'lat' in col_lower and not lat_col:
                    lat_col = col
                elif any(term in col_lower for term in ['lon', 'lng']) and not lon_col:
                    lon_col = col
            
            if lat_col and lon_col and 'hpm' in analyzer.current_data.columns:
                if st.button("🗺️ Show Property Map", type="secondary"):
                    try:
                        # Prepare map data
                        map_data = analyzer.current_data[[lat_col, lon_col, 'hpm']].copy()
                        map_data = map_data.dropna()
                        
                        # Convert to numeric
                        map_data[lat_col] = pd.to_numeric(map_data[lat_col], errors='coerce')
                        map_data[lon_col] = pd.to_numeric(map_data[lon_col], errors='coerce')
                        map_data['hpm'] = pd.to_numeric(map_data['hpm'], errors='coerce')
                        
                        # Remove invalid coordinates
                        map_data = map_data.dropna()
                        map_data = map_data[
                            (map_data[lat_col] >= -90) & (map_data[lat_col] <= 90) &
                            (map_data[lon_col] >= -180) & (map_data[lon_col] <= 180)
                        ]
                        
                        if not map_data.empty:
                            # Create quantiles for HPM
                            map_data['hpm_quantile'] = pd.qcut(map_data['hpm'], q=5, labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'])
                            
                            # Color mapping (Viridis)
                            color_map = {
                                'Q1': '#440154',  # Dark purple
                                'Q2': '#31688e',  # Dark blue
                                'Q3': '#35b779',  # Green
                                'Q4': '#fde725',  # Yellow
                                'Q5': '#dcf44c'   # Light yellow
                            }
                            
                            map_data['color'] = map_data['hpm_quantile'].map(color_map)
                            
                            # Create the map
                            fig = go.Figure()
                            
                            # Add points for each quantile
                            for quantile in ['Q1', 'Q2', 'Q3', 'Q4', 'Q5']:
                                quantile_data = map_data[map_data['hpm_quantile'] == quantile]
                                if not quantile_data.empty:
                                    fig.add_trace(go.Scattermapbox(
                                        lat=quantile_data[lat_col],
                                        lon=quantile_data[lon_col],
                                        mode='markers',
                                        marker=dict(
                                            size=8,
                                            color=color_map[quantile],
                                            opacity=0.8
                                        ),
                                        text=[f"HPM: {val:,.0f}<br>Quantile: {quantile}" 
                                             for val in quantile_data['hpm']],
                                        hovertemplate='<b>%{text}</b><br>Lat: %{lat:.4f}<br>Lon: %{lon:.4f}<extra></extra>',
                                        name=f'{quantile} (HPM: {quantile_data["hpm"].min():,.0f} - {quantile_data["hpm"].max():,.0f})'
                                    ))
                            
                            # Map layout
                            center_lat = map_data[lat_col].mean()
                            center_lon = map_data[lon_col].mean()
                            
                            fig.update_layout(
                                mapbox=dict(
                                    style="open-street-map",
                                    center=dict(lat=center_lat, lon=center_lon),
                                    zoom=10
                                ),
                                height=600,
                                margin=dict(l=0, r=0, t=30, b=0),
                                title=f"Property HPM Distribution - {len(map_data):,} properties (Quantiles, Viridis Colormap)",
                                legend=dict(
                                    yanchor="top",
                                    y=0.99,
                                    xanchor="left",
                                    x=0.01
                                )
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Map statistics
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("Properties Mapped", f"{len(map_data):,}")
                            with col2:
                                st.metric("HPM Range", f"{map_data['hpm'].min():,.0f} - {map_data['hpm'].max():,.0f}")
                            with col3:
                                st.metric("Median HPM", f"{map_data['hpm'].median():,.0f}")
                            with col4:
                                st.metric("Map Center", f"{center_lat:.3f}, {center_lon:.3f}")
                            
                            # Quantile statistics
                            st.markdown("**HPM Quantile Statistics:**")
                            quantile_stats = map_data.groupby('hpm_quantile')['hpm'].agg(['count', 'min', 'max', 'mean']).round(0)
                            quantile_stats.columns = ['Count', 'Min HPM', 'Max HPM', 'Mean HPM']
                            st.dataframe(quantile_stats, use_container_width=True)
                            
                        else:
                            st.warning("No valid coordinate data available for mapping")
                        
                    except Exception as e:
                        st.error(f"Map generation failed: {str(e)}")
        
        else:
            st.info("Apply a filter first to enable map visualization")
        
        if not selected_filters:
            st.info("No manual filters applied - use quick filters above or add manual filters below")

elif st.session_state.processing_step == 'clean':
    st.markdown('## <img src="https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExajh0eW9iZTMxZDZpMGxzbTgxanVicXM4b2YybW5zdDR2cHQzMjFnMiZlcD12MV9naWZzX3NlYXJjaCZjdD1n/NV4cSrRYXXwfUcYnua/giphy.gif" alt="data gif" style="height:72px; vertical-align:middle;"> Data Cleaning', unsafe_allow_html=True)

    if analyzer.current_data is not None:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Basic Cleaning:**")
            remove_duplicates = st.checkbox("Remove duplicate records (keep first)", value=False)
            handle_missing = st.checkbox("Handle missing values", value=False)
        
        with col2:
            st.markdown("**Outlier Removal:**")
            remove_outliers = st.checkbox("Remove outliers", value=False)
            outlier_column = None
            if remove_outliers:
                numeric_cols = analyzer.current_data.select_dtypes(include=[np.number]).columns
                outlier_column = st.selectbox("Column for outlier detection", numeric_cols)
        
        # Cleaning options dictionary
        cleaning_options = {
            'remove_duplicates': remove_duplicates,
            'handle_missing': handle_missing,
            'remove_outliers': remove_outliers,
            'outlier_column': outlier_column
        }
        
        # Apply cleaning button
        if st.button("🧹 Apply Data Cleaning", type="primary"):
            with st.spinner("Cleaning data..."):
                success, message = analyzer.clean_data(cleaning_options)
                if success:
                    st.success(message)
                    st.rerun()
                else:
                    st.error(message)

elif st.session_state.processing_step == 'transform':
    st.markdown('## <img src="https://media.giphy.com/media/v1.Y2lkPWVjZjA1ZTQ3Y25yOXB5MDFqNGlmdmJnenFqandjMzl6YnJscnRseDlzN2poZG1wMiZlcD12MV9naWZzX3NlYXJjaCZjdD1n/EjLTU9HAnnskywtJ9j/giphy.gif" alt="data gif" style="height:9px; vertical-align:middle;"> Variable Transformations', unsafe_allow_html=True)
    
    if analyzer.current_data is not None:
        st.markdown("### Apply Transformations")
        st.info("Select columns to transform. New transformed columns will be added to your dataset and available for OLS modeling.")
        
        # Get numeric columns
        numeric_columns = analyzer.current_data.select_dtypes(include=[np.number]).columns.tolist()
        
        if not numeric_columns:
            st.warning("No numeric columns available for transformation")
        else:
            # Shortcut for distance columns
            st.markdown("### 🚀 Quick Distance Transformations")
            distance_columns = [col for col in numeric_columns if 'distance_to_' in col.lower()]
            
            if distance_columns:
                st.info(f"Found {len(distance_columns)} distance columns: {', '.join(distance_columns)}")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    distance_transform = st.selectbox("Transform all distance columns", 
                                                    ['log', 'squared', 'sqrt'], 
                                                    key="distance_transform_select")
                
                with col2:
                    st.write("")
                    st.write("")
                    if st.button("⚡ Transform All Distance Columns", type="secondary"):
                        try:
                            distance_transformations = {col: distance_transform for col in distance_columns}
                            success, message = analyzer.apply_transformations(distance_transformations)
                            if success:
                                st.success(f"✅ Applied {distance_transform} transformation to {len(distance_columns)} distance columns")
                                for col in distance_columns:
                                    new_col = analyzer.transformed_columns.get(col, 'Unknown')
                                    st.success(f"🔹 Created: {new_col}")
                                st.rerun()
                            else:
                                st.error(message)
                        except Exception as e:
                            st.error(f"Batch transformation failed: {str(e)}")
                
                st.markdown("**Distance Columns Found:**")
                for col in distance_columns:
                    st.write(f"- {col}")
                
                st.markdown("---")
            
            # Individual transformations
            st.markdown("### Individual Column Transformations")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Column selection
                st.markdown("**Select Column to Transform:**")
                selected_column = st.selectbox("Choose column", numeric_columns, key="transform_column_select")
                
                # Transformation type
                st.markdown("**Select Transformation:**")
                transform_type = st.selectbox("Choose transformation", 
                                            ['log', 'squared', 'sqrt'], 
                                            key="transform_type_select")
                
                # Apply single transformation
                if st.button("⚡ Apply Transformation", type="primary"):
                    if selected_column and transform_type:
                        transformations = {selected_column: transform_type}
                        with st.spinner(f"Applying {transform_type} transformation to {selected_column}..."):
                            success, message = analyzer.apply_transformations(transformations)
                            if success:
                                st.success(f"✅ {message}")
                                st.success(f"🔹 Created new column: {analyzer.transformed_columns.get(selected_column, 'Unknown')}")
                                st.rerun()
                            else:
                                st.error(message)
            
            with col2:
                # Preview of selected column
                if selected_column:
                    st.markdown(f"**Preview of {selected_column}:**")
                    col_data = analyzer.current_data[selected_column].dropna()
                    
                    st.write(f"- Count: {len(col_data):,}")
                    st.write(f"- Min: {col_data.min():.2f}")
                    st.write(f"- Max: {col_data.max():.2f}")
                    st.write(f"- Mean: {col_data.mean():.2f}")
                    st.write(f"- Median: {col_data.median():.2f}")
                    
                    # Show what the transformation would look like
                    if transform_type:
                        st.markdown(f"**{transform_type.title()} Transformation Preview:**")
                        if transform_type == 'log':
                            preview_data = np.log(col_data + 1)
                            new_col_name = f"ln_{selected_column}"
                        elif transform_type == 'squared':
                            preview_data = col_data ** 2
                            new_col_name = f"{selected_column}_squared"
                        elif transform_type == 'sqrt':
                            preview_data = np.sqrt(np.abs(col_data))
                            new_col_name = f"sqrt_{selected_column}"
                        
                        st.write(f"- New column name: **{new_col_name}**")
                        st.write(f"- Min: {preview_data.min():.2f}")
                        st.write(f"- Max: {preview_data.max():.2f}")
                        st.write(f"- Mean: {preview_data.mean():.2f}")
        
        # Show current transformations
        if analyzer.transformed_columns:
            st.markdown("### 🔄 Current Transformations")
            st.success(f"You have {len(analyzer.transformed_columns)} transformed columns available for modeling:")
            
            transformation_df = pd.DataFrame([
                {'Original Column': original, 'Transformed Column': transformed, 'Available for OLS': '✅', 'Available for ML': '✅'}
                for original, transformed in analyzer.transformed_columns.items()
            ])
            st.dataframe(transformation_df, use_container_width=True)
            
            # Option to remove transformations
            st.markdown("**Remove Transformations:**")
            cols_to_remove = st.multiselect("Select transformed columns to remove", 
                                          list(analyzer.transformed_columns.values()),
                                          key="remove_transforms")
            
            if cols_to_remove and st.button("🗑️ Remove Selected Transformations", type="secondary"):
                try:
                    # Remove columns from dataframe
                    analyzer.current_data = analyzer.current_data.drop(columns=cols_to_remove, errors='ignore')
                    
                    # Remove from transformed_columns tracking
                    to_remove_original = []
                    for original, transformed in analyzer.transformed_columns.items():
                        if transformed in cols_to_remove:
                            to_remove_original.append(original)
                    
                    for original in to_remove_original:
                        del analyzer.transformed_columns[original]
                    
                    st.success(f"Removed {len(cols_to_remove)} transformed columns")
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"Error removing transformations: {str(e)}")
        else:
            st.info("No transformations applied yet. Transform some variables to make them available for OLS and ML modeling.")

elif st.session_state.processing_step == 'model':
    st.markdown('## <img src="https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExd3phdHl3Zm0zZTZoZ2RkdHc5Z3NncHIzaXpjdWI4bmw1YzluMm0ydiZlcD12MV9naWZzX3NlYXJjaCZjdD1n/9ADoZQgs0tyww/giphy.gif" alt="data gif" style="height:72px; vertical-align:middle;"> OLS Regression Analysis', unsafe_allow_html=True)
    
    if analyzer.current_data is not None:
        # Model configuration
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Model Variables:**")
            
            # Get all numeric columns (including transformed ones)
            numeric_columns = analyzer.current_data.select_dtypes(include=[np.number]).columns.tolist()
            
            # Target variable (Y)
            y_column = st.selectbox("Dependent Variable (Y)", numeric_columns, 
                                   index=numeric_columns.index('hpm') if 'hpm' in numeric_columns else 0)
            
            # Independent variables (X)
            available_x_cols = [col for col in numeric_columns if col != y_column]
            x_columns = st.multiselect("Independent Variables (X)", available_x_cols,
                                      default=available_x_cols[:5] if len(available_x_cols) >= 5 else available_x_cols)
        
        with col2:
            st.markdown("**Model Information:**")
            if analyzer.transformed_columns:
                st.write("**Available Transformations:**")
                for original, transformed in analyzer.transformed_columns.items():
                    st.write(f"- {original} → {transformed}")
            
            st.write(f"**Sample size:** {len(analyzer.current_data):,} observations")
            if x_columns:
                # Show actual column names that will be used
                actual_y = analyzer.transformed_columns.get(y_column, y_column)
                actual_x = [analyzer.transformed_columns.get(col, col) for col in x_columns]
                st.write(f"**Model formula:** {actual_y} ~ {' + '.join(actual_x)}")
        
        # Run OLS Model button
        if st.button("📈 Run OLS Regression", type="primary") and x_columns:
            with st.spinner("Fitting OLS model..."):
                success, message, vif_df = analyzer.run_ols_model(y_column, x_columns)
                
                if success:
                    st.success(message)
                    
                    # Display model results
                    results = analyzer.get_model_results()
                    
                    if results:
                        # Model metrics
                        st.markdown("### 📊 Model Results")
                        
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("R-squared", f"{results['rsquared']:.4f}")
                        with col2:
                            st.metric("Adj. R-squared", f"{results['rsquared_adj']:.4f}")
                        with col3:
                            st.metric("F-statistic", f"{results['fvalue']:.2f}")
                        with col4:
                            st.metric("AIC", f"{results['aic']:.2f}")
                        
                        # VIF Results
                        if vif_df is not None:
                            st.markdown("### 📊 Variance Inflation Factors (VIF)")
                            st.dataframe(vif_df.style.format({'VIF': '{:.2f}'}), use_container_width=True)
                            
                            # VIF interpretation
                            high_vif = vif_df[vif_df['VIF'] > 5]
                            if not high_vif.empty:
                                st.warning(f"High VIF detected (>5): {', '.join(high_vif['feature'].tolist())}")
                        
                        # Model summary
                        with st.expander("📋 Detailed Model Summary", expanded=False):
                            st.text(results['summary'])
                        
                        # Coefficients table
                        st.markdown("### 📊 Model Coefficients")
                        coef_df = pd.DataFrame({
                            'Coefficient': results['params'],
                            'P-value': results['pvalues'],
                            'CI_Lower': results['conf_int'][0],
                            'CI_Upper': results['conf_int'][1]
                        })
                        coef_df['Significant'] = coef_df['P-value'] < 0.05
                        st.dataframe(coef_df.style.format({
                            'Coefficient': '{:.4f}',
                            'P-value': '{:.4f}',
                            'CI_Lower': '{:.4f}',
                            'CI_Upper': '{:.4f}'
                        }), use_container_width=True)
                        
                        # Model diagnostics plots
                        st.markdown("### 📈 Model Diagnostics")
                        
                        # Create diagnostic plots using matplotlib style
                        dataset_name = f"Real Estate Analysis - {len(analyzer.model_data)} observations"
                        
                        # Create three plots side by side
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            # Plot 1: Predicted vs Actual (using seaborn style)
                            fig1, ax1 = plt.subplots(figsize=(6, 4))
                            y_pred = results['fitted_values']
                            y_actual = results['actual_values']
                            
                            sns.regplot(x=y_pred, y=y_actual, line_kws={"color": "red"}, ax=ax1)
                            ax1.set_xlabel('Predicted Values')
                            ax1.set_ylabel('Actual Values')
                            ax1.set_title(f'{dataset_name}\nPredicted vs Actual')
                            plt.tight_layout()
                            st.pyplot(fig1)
                        
                        with col2:
                            # Plot 2: Residuals vs Fitted
                            fig2, ax2 = plt.subplots(figsize=(6, 4))
                            residuals = results['residuals']
                            fitted_values = results['fitted_values']
                            
                            sns.scatterplot(x=fitted_values, y=residuals, alpha=0.7, ax=ax2)
                            ax2.axhline(0, color='gray', linestyle='--')
                            ax2.set_xlabel('Fitted Values')
                            ax2.set_ylabel('Residuals')
                            ax2.set_title(f'{dataset_name}\nResiduals vs Fitted')
                            plt.tight_layout()
                            st.pyplot(fig2)
                        
                        with col3:
                            # Plot 3: Residual Normality (KDE Plot)
                            fig3, ax3 = plt.subplots(figsize=(6, 4))
                            sns.kdeplot(residuals, fill=True, color='blue', ax=ax3)
                            ax3.set_title(f'{dataset_name}\nResidual Normality Plot (KDE)')
                            ax3.set_xlabel('Residuals')
                            plt.tight_layout()
                            st.pyplot(fig3)
                        
                        # Additional diagnostic statistics
                        st.markdown("### 📊 Diagnostic Statistics")
                        diag_col1, diag_col2, diag_col3, diag_col4 = st.columns(4)
                        
                        with diag_col1:
                            st.metric("Mean Residual", f"{residuals.mean():.6f}")
                        with diag_col2:
                            st.metric("Std Residual", f"{residuals.std():.4f}")
                        with diag_col3:
                            # Durbin-Watson test
                            try:
                                dw_stat = sm.stats.stattools.durbin_watson(residuals)
                                st.metric("Durbin-Watson", f"{dw_stat:.4f}")
                            except:
                                st.metric("Durbin-Watson", "N/A")
                        with diag_col4:
                            # Jarque-Bera test for normality
                            try:
                                jb_result = sm.stats.jarque_bera(residuals)
                                jb_pvalue = jb_result[1]
                                st.metric("Jarque-Bera p-value", f"{jb_pvalue:.4f}")
                            except:
                                st.metric("Jarque-Bera p-value", "N/A")
                        
                        # Export options
                        st.markdown("### 💾 Export Results")
                        
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            # Export model summary
                            actual_y = analyzer.transformed_columns.get(y_column, y_column)
                            actual_x = [analyzer.transformed_columns.get(col, col) for col in x_columns]
                            
                            model_summary = {
                                'timestamp': datetime.now().isoformat(),
                                'model_formula': f"{actual_y} ~ {' + '.join(actual_x)}",
                                'original_variables': {
                                    'y': y_column,
                                    'x': x_columns
                                },
                                'transformed_variables': {
                                    'y': actual_y,
                                    'x': actual_x
                                },
                                'sample_size': len(analyzer.model_data),
                                'r_squared': results['rsquared'],
                                'adj_r_squared': results['rsquared_adj'],
                                'f_statistic': results['fvalue'],
                                'f_pvalue': results['f_pvalue'],
                                'aic': results['aic'],
                                'bic': results['bic']
                            }
                            
                            st.download_button(
                                label="📊 Download Model Summary",
                                data=json.dumps(model_summary, indent=2),
                                file_name=f"model_results_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
                                mime="application/json"
                            )
                        
                        with col2:
                            # Export coefficients
                            st.download_button(
                                label="📋 Download Coefficients",
                                data=coef_df.to_csv(),
                                file_name=f"model_coefficients_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                                mime="text/csv"
                            )
                        
                        with col3:
                            # Export VIF if available
                            if vif_df is not None:
                                st.download_button(
                                    label="📊 Download VIF",
                                    data=vif_df.to_csv(),
                                    file_name=f"vif_results_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                                    mime="text/csv"
                                )
                else:
                    st.error(message)
        
elif st.session_state.processing_step == 'ml':
    st.markdown('## <img src="https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExdmo4MXdyeGNsc2V2MGs0ZTFvNXA1YzlycWw2MDhxZm5hZW8ybjNmYSZlcD12MV9naWZzX3NlYXJjaCZjdD1n/3o7TKpVj5lCPd3CtPi/giphy.gif" alt="data gif" style="height:72px; vertical-align:middle;"> Machine Learning Models', unsafe_allow_html=True)
    
    if analyzer.current_data is not None:
        # Model Configuration
        st.markdown("### 🎯 Model Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Model Variables:**")
            
            # Get all numeric columns (including transformed ones)
            numeric_columns = analyzer.current_data.select_dtypes(include=[np.number]).columns.tolist()
            
            # Target variable (Y)
            ml_y_column = st.selectbox("Target Variable (Y)", numeric_columns, 
                                      key="ml_y_select")
            
            # Independent variables (X)
            available_x_cols = [col for col in numeric_columns if col != ml_y_column]
            ml_x_columns = st.multiselect("Feature Variables (X)", available_x_cols,
                                         default=available_x_cols[:5] if len(available_x_cols) >= 5 else available_x_cols,
                                         key="ml_x_select")
        
        with col2:
            st.markdown("**Group Configuration:**")
            
            # Group column selection
            use_group = st.checkbox("Use Group-Based Cross-Validation", value=False)
            group_column = None
            
            if use_group:
                # Find geographic columns (case insensitive)
                geo_candidates = []
                for col in analyzer.current_data.columns:
                    col_upper = col
                    if any(geo in col_upper for geo in ['wadmpr', 'wadmkk', 'wadmkc', 'wadmkd']):
                        geo_candidates.append(col)
                
                # Also include encoded columns
                encoded_candidates = [col for col in analyzer.current_data.columns if '_encoded' in col.lower()]
                
                all_group_candidates = geo_candidates + encoded_candidates
                
                if all_group_candidates:
                    group_column = st.selectbox("Select Group Column", all_group_candidates)
                else:
                    st.warning("No geographic or encoded columns found for grouping")
                    use_group = False
            
            # Label Encoding Option
            st.markdown("**Label Encoding:**")
            categorical_cols = analyzer.current_data.select_dtypes(include=['object', 'category']).columns.tolist()
            
            if categorical_cols:
                encode_column = st.selectbox("Create Encoded Column", ['None'] + categorical_cols)
                
                if encode_column != 'None' and st.button("🔤 Create Encoded Column"):
                    success, message = analyzer.apply_label_encoding(encode_column)
                    if success:
                        st.success(message)
                        st.rerun()
                    else:
                        st.error(message)
        
        # Hyperparameter Configuration
        st.markdown("### ⚙️ Hyperparameter Configuration")
        
        tab1, tab2 = st.tabs(["🎯 Optuna Optimization", "✏️ Manual Parameters"])
        
        with tab1:
            st.markdown("**Optuna Hyperparameter Optimization**")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                objective_metric = st.selectbox("Optimization Objective", 
                                              ['R2', 'PE10', 'RT20', 'FSD'], 
                                              index=0)
                n_trials = st.number_input("Number of Trials", min_value=10, max_value=200, value=50)
            
            with col2:
                min_sample = st.number_input("Minimum Sample per Group", min_value=1, max_value=10, value=3)
                n_splits = st.number_input("Cross-Validation Folds", min_value=3, max_value=20, value=10)
            
            with col3:
                random_state = st.number_input("Random State", min_value=1, max_value=1000, value=101)
            
            # Optuna optimization
            if st.button("🔍 Run Optuna Optimization", type="primary") and ml_x_columns:
                
                def objective(trial):
                    # Random Forest hyperparameters
                    n_estimators = trial.suggest_int('n_estimators', 50, 500)
                    max_depth = trial.suggest_int('max_depth', 3, 20)
                    min_samples_split = trial.suggest_int('min_samples_split', 2, 10)
                    min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 5)
                    max_features = trial.suggest_categorical('max_features', ['sqrt', 'log2', None])
                    
                    # Create model
                    rf_model = RandomForestRegressor(
                        n_estimators=n_estimators,
                        max_depth=max_depth,
                        min_samples_split=min_samples_split,
                        min_samples_leaf=min_samples_leaf,
                        max_features=max_features,
                        random_state=random_state
                    )
                    
                    # Run evaluation
                    try:
                        _, _, _, _, global_test_metrics, _, _, _ = analyzer.goval_machine_learning(
                            ml_x_columns, ml_y_column, rf_model, 
                            group_column if use_group else None,
                            n_splits, random_state, min_sample
                        )
                        
                        # Return objective based on selected metric
                        if objective_metric == 'R2':
                            return global_test_metrics['R2']  # Maximize
                        elif objective_metric == 'PE10':
                            return global_test_metrics['PE10']  # Maximize
                        elif objective_metric == 'RT20':
                            return -global_test_metrics['RT20']  # Minimize (so negative to maximize)
                        elif objective_metric == 'FSD':
                            return -global_test_metrics['FSD']  # Minimize (so negative to maximize)
                        
                    except Exception as e:
                        st.error(f"Optuna trial failed: {e}")
                        st.code(traceback.format_exc())
                        return -999
                
                with st.spinner(f"Running Optuna optimization for {n_trials} trials..."):
                    try:
                        # Create study
                        direction = 'maximize'
                        study = optuna.create_study(direction=direction)
                        study.optimize(objective, n_trials=n_trials)
                        
                        # Store best parameters in session state
                        st.session_state.best_params = study.best_params
                        st.session_state.best_value = study.best_value
                        
                        st.success(f"✅ Optimization completed!")
                        st.write("**Best Parameters:**")
                        st.json(study.best_params)
                        st.write(f"**Best {objective_metric}:** {study.best_value:.4f}")
                        
                    except Exception as e:
                        st.error(f"Optuna optimization failed: {str(e)}")
        
        with tab2:
            st.markdown("**Manual Random Forest Parameters**")
            
            # Check if we have best params from Optuna
            if 'best_params' in st.session_state:
                st.info("💡 Optuna found these optimal parameters. You can use them or modify below:")
                st.json(st.session_state.best_params)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                manual_n_estimators = st.text_input("n_estimators", value="100")
                manual_max_depth = st.text_input("max_depth", value="10")
            
            with col2:
                manual_min_samples_split = st.text_input("min_samples_split", value="2")
                manual_min_samples_leaf = st.text_input("min_samples_leaf", value="1")
            
            with col3:
                manual_max_features = st.selectbox("max_features", ['sqrt', 'log2', 'None'], index=0)
                manual_random_state = st.text_input("random_state", value=str(random_state))
        
        # Model Training
        st.markdown("### 🚀 Model Training & Evaluation")
        
        if st.button("🤖 Train Random Forest Model", type="primary") and ml_x_columns:
            
            try:
                # Parse manual parameters
                try:
                    n_est = int(manual_n_estimators)
                    max_d = int(manual_max_depth) if manual_max_depth != 'None' else None
                    min_split = int(manual_min_samples_split)
                    min_leaf = int(manual_min_samples_leaf)
                    max_feat = manual_max_features if manual_max_features != 'None' else None
                    rand_state = int(manual_random_state)
                except ValueError:
                    st.error("Please enter valid integer values for parameters")
                    st.stop()
                
                # Create Random Forest model
                rf_model = RandomForestRegressor(
                    n_estimators=n_est,
                    max_depth=max_d,
                    min_samples_split=min_split,
                    min_samples_leaf=min_leaf,
                    max_features=max_feat,
                    random_state=rand_state
                )
                
                with st.spinner("Training Random Forest model..."):
                    # Run model training and evaluation
                    final_model, evaluation_df, train_results_df, global_train_metrics, global_test_metrics, y_test_last, y_pred_last, is_log_transformed = analyzer.goval_machine_learning(
                        ml_x_columns, ml_y_column, rf_model,
                        group_column if use_group else None,
                        n_splits, rand_state, min_sample
                    )
                    
                    # Store model in session state
                    st.session_state.ml_model = final_model
                    st.session_state.ml_evaluation_df = evaluation_df
                    st.session_state.ml_train_df = train_results_df
                    st.session_state.ml_global_train = global_train_metrics
                    st.session_state.ml_global_test = global_test_metrics
                    st.session_state.ml_y_test_last = y_test_last
                    st.session_state.ml_y_pred_last = y_pred_last
                    st.session_state.ml_is_log_transformed = is_log_transformed
                
                st.success("✅ Model training completed!")
                
                # Display Results
                st.markdown("### 📊 Model Results")
                
                # Global metrics
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**🏋️ Training Metrics (Average)**")
                    for metric, value in global_train_metrics.items():
                        st.metric(f"Train {metric}", f"{value:.4f}")
                
                with col2:
                    st.markdown("**🎯 Test Metrics (Average)**")
                    for metric, value in global_test_metrics.items():
                        st.metric(f"Test {metric}", f"{value:.4f}")
                
                # Detailed Results
                st.markdown("### 📈 Detailed Cross-Validation Results")
                
                tab1, tab2, tab3 = st.tabs(["📊 Test Results", "🏋️ Train Results", "📈 Visualization"])
                
                with tab1:
                    st.markdown("**Test Results by Fold:**")
                    st.dataframe(evaluation_df, use_container_width=True)
                    
                    st.markdown("**Test Results Summary:**")
                    st.dataframe(evaluation_df.describe(), use_container_width=True)
                
                with tab2:
                    st.markdown("**Train Results Summary:**")
                    st.dataframe(train_results_df.describe(), use_container_width=True)
                
                with tab3:
                    # Actual vs Predicted plot
                    fig, ax = plt.subplots(figsize=(8, 6))
                    
                    if is_log_transformed:
                        y_actual_plot = np.exp(y_test_last)
                        y_pred_plot = np.exp(y_pred_last)
                    else:
                        y_actual_plot = y_test_last
                        y_pred_plot = y_pred_last
                    
                    sns.scatterplot(x=y_actual_plot, y=y_pred_plot, alpha=0.6, ax=ax)
                    ax.plot([y_actual_plot.min(), y_actual_plot.max()], 
                           [y_actual_plot.min(), y_actual_plot.max()], 'r--', lw=2)
                    ax.set_xlabel('Actual Values')
                    ax.set_ylabel('Predicted Values')
                    ax.set_title('Actual vs Predicted (Last Fold)')
                    plt.tight_layout()
                    st.pyplot(fig)
                
                # Model Export
                st.markdown("### 💾 Model Export")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    # Download model
                    model_filename = f"rf_model_{datetime.now().strftime('%Y%m%d_%H%M')}.pkl"
                    model_bytes = pickle.dumps(final_model)
                    
                    st.download_button(
                        label="📦 Download Model (.pkl)",
                        data=model_bytes,
                        file_name=model_filename,
                        mime="application/octet-stream"
                    )
                
                with col2:
                    # Download evaluation results
                    eval_csv = evaluation_df.to_csv(index=False)
                    st.download_button(
                        label="📊 Download Results (.csv)",
                        data=eval_csv,
                        file_name=f"ml_evaluation_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                        mime="text/csv"
                    )
                
                with col3:
                    # Download model info
                    model_info = {
                        'timestamp': datetime.now().isoformat(),
                        'algorithm': 'RandomForestRegressor',
                        'target_variable': ml_y_column,
                        'features': ml_x_columns,
                        'parameters': {
                            'n_estimators': n_est,
                            'max_depth': max_d,
                            'min_samples_split': min_split,
                            'min_samples_leaf': min_leaf,
                            'max_features': max_feat,
                            'random_state': rand_state
                        },
                        'group_column': group_column if use_group else None,
                        'n_splits': n_splits,
                        'min_sample': min_sample,
                        'is_log_transformed': is_log_transformed,
                        'global_test_metrics': global_test_metrics,
                        'global_train_metrics': global_train_metrics
                    }
                    
                    st.download_button(
                        label="📋 Download Model Info (.json)",
                        data=json.dumps(model_info, indent=2),
                        file_name=f"ml_model_info_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
                        mime="application/json"
                    )
                
            except Exception as e:
                st.error(f"Model training failed: {str(e)}")
                st.code(traceback.format_exc())
        
        elif not ml_x_columns:
            st.warning("Please select at least one feature variable to train the model")
        
        # Show current model status
        if 'ml_model' in st.session_state:
            st.markdown("---")
            st.success("🤖 Random Forest model is ready!")
            
            # Quick model info
            with st.expander("📋 Current Model Info"):
                st.write(f"**Algorithm:** Random Forest Regressor")
                if 'ml_global_test' in st.session_state:
                    metrics = st.session_state.ml_global_test
                    st.write(f"**Test R²:** {metrics['R2']:.4f}")
                    st.write(f"**Test PE10:** {metrics['PE10']:.4f}")
                    st.write(f"**Test RT20:** {metrics['RT20']:.4f}")
                    st.write(f"**Test FSD:** {metrics['FSD']:.4f}")
    
    else:
        st.warning("Please load and process data first")

# Data preview section (always available at bottom)
if analyzer.current_data is not None:
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    
    with st.expander("👁️ Current Data Preview", expanded=False):
        # Show current data state
        st.markdown(f"**Current Data Shape:** {analyzer.current_data.shape[0]:,} rows × {analyzer.current_data.shape[1]} columns")
        
        if analyzer.transformed_columns:
            st.markdown("**Transformed Columns:**")
            for original, transformed in analyzer.transformed_columns.items():
                st.write(f"- {original} → {transformed}")
        
        # Data preview
        st.dataframe(analyzer.current_data.head(20), use_container_width=True)
        
        # Quick statistics for numeric columns
        numeric_data = analyzer.current_data.select_dtypes(include=[np.number])
        if not numeric_data.empty:
            st.markdown("**Numeric Columns Statistics:**")
            st.dataframe(numeric_data.describe(), use_container_width=True)

# Sidebar with current status and quick actions
with st.sidebar:
    st.markdown("## 📊 Current Status")
    
    if analyzer.current_data is not None:
        st.success(f"✅ Data Loaded: {len(analyzer.current_data):,} properties")
        
        # Show current step
        current_step_name = next((name for key, name in workflow_steps if key == st.session_state.processing_step), "Unknown")
        st.info(f"📍 Current Step: {current_step_name}")
        
        # Show transformations if any
        if analyzer.transformed_columns:
            st.markdown("**🔄 Active Transformations:**")
            for original, transformed in analyzer.transformed_columns.items():
                st.write(f"- {original} → {transformed}")
        
        # Show model status
        if analyzer.model is not None:
            st.success("✅ OLS Model Ready")
            st.write(f"R² = {analyzer.model.rsquared:.3f}")
        else:
            st.warning("⚠️ No Model Fitted")
    else:
        st.error("❌ No Data Loaded")
    
    st.markdown("---")
    
    # Quick actions
    st.markdown("## ⚡ Quick Actions")
    
    if st.button("🔄 Reset All", help="Reset to original data"):
        success, message = analyzer.reset_to_original()
        if success:
            st.success(message)
            st.rerun()
    
    if analyzer.current_data is not None:
        # Export current data
        st.download_button(
            label="💾 Export Current Data (.csv)",
            data=analyzer.current_data.to_csv(index=False),
            file_name=f"processed_data_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv"
        )
    
    # Session info
    session_info = {
        'timestamp': datetime.now().isoformat(),
        'data_shape': analyzer.current_data.shape if analyzer.current_data is not None else None,
        'current_step': st.session_state.processing_step,
        'transformations': analyzer.transformed_columns,
        'has_model': analyzer.model is not None
    }
    
    st.download_button(
        label="📄 Export Session Info (.json)",
        data=json.dumps(session_info, indent=2),
        file_name=f"session_info_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
        mime="application/json"
    )

# Footer
st.markdown("---")
st.markdown("""
<h1 style="display: flex; align-items: center;">
    <img src="https://kjpp.rhr.co.id/wp-content/uploads/2020/12/LOGO_KJPP_RHR_1_resize.png" 
         alt="Logo" style="height:48px; margin-right: 20px;">
    <span style="font-weight: bold; font-size: 1.5rem;"></span>
</h1>
""", unsafe_allow_html=True)
