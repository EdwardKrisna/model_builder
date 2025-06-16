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
# st.markdown(
#     "<div style='text-align:center; font-weight:bold; font-size: 3rem; margin-bottom: 1rem;'>RHR MODEL BUILDER 𓀒 𓀓 𓀔</div>", 
#     unsafe_allow_html=True
# )

fun_mode = st.sidebar.checkbox("Surprise!", value=False)

if fun_mode:
    st.markdown(
        """
        <style>
        .full-width-gif {
            width: 100%;
            max-height: 200px;
            object-fit: cover;
            margin: 0;
            padding: 0;
            display: block;
        }
        </style>
        <img src="https://media.giphy.com/media/v1.Y2lkPWVjZjA1ZTQ3aXdydWZwbjN1aTY5NDhvZzRwYzhwcnlrb2p3NmRsYng4cHB3OWZvMyZlcD12MV9naWZzX3JlbGF0ZWQmY3Q9Zw/NKEt9elQ5cR68/giphy.gif" 
             alt="Fun Mode GIF" 
             class="full-width-gif" />
        """,
        unsafe_allow_html=True,
    )

@st.cache_resource(show_spinner=False, ttl=3600)  # Add TTL to refresh connection
def cached_connect_database():
    try:
        db_user = st.secrets["database"]["user"]
        db_password = st.secrets["database"]["password"]
        db_host = st.secrets["database"]["host"]
        db_port = st.secrets["database"]["port"]
        db_name = st.secrets["database"]["database"]
        
        # Add connection pooling and timeout parameters
        connection_string = f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}?connect_timeout=10&application_name=streamlit_app"
        
        engine = create_engine(
            connection_string,
            pool_size=5,
            max_overflow=10,
            pool_pre_ping=True,
            pool_recycle=3600
        )
        
        # Test connection with timeout
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return engine
    except Exception as e:
        st.error(f"Database connection failed: {e}")
        return None

@st.cache_data(ttl=1800, show_spinner=False)  # Cache for 30 minutes
def cached_load_property_data(_engine):
    """Load data with optimized query and chunking"""
    if _engine is None:
        return pd.DataFrame()
    
    try:
        # Use more specific query to reduce data transfer
        query = """
        SELECT * FROM engineered_property_data 
        WHERE hpm BETWEEN 50000 AND 200000000
        ORDER BY hpm
        LIMIT 100000;  -- Add reasonable limit
        """
        
        # Use chunking for large datasets
        df_chunks = []
        chunk_size = 10000
        
        for chunk in pd.read_sql(query, _engine, chunksize=chunk_size):
            df_chunks.append(chunk)
            if len(df_chunks) * chunk_size >= 200000:  # Limit total rows
                break
        
        if df_chunks:
            df = pd.concat(df_chunks, ignore_index=True)
        else:
            df = pd.DataFrame()
            
        return df
    except Exception as e:
        st.error(f"Data loading failed: {e}")
        return pd.DataFrame()

@st.cache_data(show_spinner=False, max_entries=3)  # Limit cache entries
def cached_clean_data(df, cleaning_options):
    """Optimized data cleaning with better error handling"""
    if df.empty:
        return df
    
    try:
        # Create a hash of cleaning options to improve cache efficiency
        cleaning_hash = hash(str(sorted(cleaning_options.items())))
        
        cleaned_df = df.copy()
        
        # Remove duplicates (vectorized operation)
        if cleaning_options.get('remove_duplicates', False):
            initial_count = len(cleaned_df)
            cleaned_df = cleaned_df.drop_duplicates(keep='first')
            print(f"Removed {initial_count - len(cleaned_df)} duplicates")
        
        # Handle missing values (optimized)
        if cleaning_options.get('handle_missing', False):
            # Use more efficient fillna operations
            numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                medians = cleaned_df[numeric_cols].median()
                cleaned_df[numeric_cols] = cleaned_df[numeric_cols].fillna(medians)
            
            categorical_cols = cleaned_df.select_dtypes(include=['object', 'category']).columns
            if len(categorical_cols) > 0:
                modes = cleaned_df[categorical_cols].mode().iloc[0] if not cleaned_df[categorical_cols].empty else 'Unknown'
                cleaned_df[categorical_cols] = cleaned_df[categorical_cols].fillna(modes)
        
        # Remove outliers (optimized)
        if cleaning_options.get('remove_outliers', False) and cleaning_options.get('outlier_column'):
            outlier_col = cleaning_options['outlier_column']
            if outlier_col in cleaned_df.columns:
                # Use quantile method which is faster
                Q1 = cleaned_df[outlier_col].quantile(0.25)
                Q3 = cleaned_df[outlier_col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                # Use boolean indexing (faster than query)
                mask = (cleaned_df[outlier_col] >= lower_bound) & (cleaned_df[outlier_col] <= upper_bound)
                cleaned_df = cleaned_df[mask]
        
        return cleaned_df
        
    except Exception as e:
        st.error(f"Data cleaning failed: {str(e)}")
        return df

@st.cache_data(show_spinner=False)
def get_data_metrics(data_shape, data_memory_usage_sum):
    """Cache expensive data status calculations"""
    return {
        'missing_pct': None,  # Calculate only when needed
        'memory_mb': data_memory_usage_sum / 1024**2
    }



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
            # Start with current data
            filtered_df = self.current_data.copy()
            
            # Debug logging
            print(f"Starting with {len(filtered_df)} records")
            
            # Apply filters in logical order: Province → Regency → District → Other
            filter_order = ['wadmpr', 'wadmkk', 'wadmkc', 'wadmkd']
            other_filters = {k: v for k, v in filters.items() if k not in filter_order}
            
            # Apply geographic filters first in hierarchy order
            for geo_col in filter_order:
                if geo_col in filters:
                    filter_config = filters[geo_col]
                    if filter_config['type'] == 'categorical' and filter_config['value']:
                        before_count = len(filtered_df)
                        filtered_df = filtered_df[filtered_df[geo_col].isin(filter_config['value'])]
                        after_count = len(filtered_df)
                        print(f"After {geo_col} filter: {before_count} → {after_count}")
                        
                        # If no records left after geographic filtering, stop
                        if len(filtered_df) == 0:
                            print(f"No records left after {geo_col} filter!")
                            break
            
            # Apply other filters
            for column, filter_config in other_filters.items():
                if column not in filtered_df.columns:
                    continue
                    
                filter_type = filter_config['type']
                filter_value = filter_config['value']
                
                before_count = len(filtered_df)
                
                if filter_type == 'categorical' and filter_value:
                    filtered_df = filtered_df[filtered_df[column].isin(filter_value)]
                    
                elif filter_type == 'numeric_range' and filter_value:
                    min_val, max_val = filter_value
                    
                    # Handle infinity properly
                    if min_val == -np.inf:
                        min_val = filtered_df[column].min()
                    if max_val == np.inf:
                        max_val = filtered_df[column].max()
                    
                    mask = (filtered_df[column] >= min_val) & (filtered_df[column] <= max_val)
                    filtered_df = filtered_df[mask]
                
                after_count = len(filtered_df)
                print(f"After {column} filter: {before_count} → {after_count}")
            
            self.current_data = filtered_df
            if 'st' in globals():
                st.session_state.data_changed = True
            return True, f"Filtered to {len(filtered_df)} properties"
            
        except Exception as e:
            print(f"Filter error: {str(e)}")
            return False, f"Filtering failed: {str(e)}"
    
    def clean_data(self, cleaning_options):
        if self.current_data is None:
            return False, "No data to clean"
        try:
            cleaned_df = cached_clean_data(self.current_data, cleaning_options)
            self.current_data = cleaned_df
            if 'st' in globals():
                st.session_state.data_changed = True
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
            if 'st' in globals():
                st.session_state.data_changed = True
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
    
    def validate_geographic_filters(self, filters):
        """Validate that geographic filters are compatible"""
        
        geo_filters = {}
        for col, config in filters.items():
            if any(geo in col for geo in ['wadmpr', 'wadmkk', 'wadmkc', 'wadmkd']):
                geo_filters[col] = config
        
        if len(geo_filters) <= 1:
            return True, "Single or no geographic filter - OK"
        
        # Test if the combination yields reasonable results
        test_df = self.current_data.copy()
        
        for col, config in geo_filters.items():
            if config['type'] == 'categorical' and config['value']:
                before = len(test_df)
                test_df = test_df[test_df[col].isin(config['value'])]
                after = len(test_df)
                
                reduction_pct = ((before - after) / before) * 100 if before > 0 else 0
                
                # Warn if any single geographic filter eliminates >90% of data
                if reduction_pct > 90:
                    return False, f"Geographic filter '{col}' eliminates {reduction_pct:.1f}% of data - likely incompatible"
        
        final_count = len(test_df)
        original_count = len(self.current_data)
        total_reduction = ((original_count - final_count) / original_count) * 100
        
        # Warn if combined geographic filters eliminate >95% of data
        if total_reduction > 95:
            return False, f"Combined geographic filters eliminate {total_reduction:.1f}% of data - likely incompatible"
        
        return True, f"Geographic filters are compatible - {final_count:,} records will remain"

        
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
            
            # Optional: ensure constant is added for correct VIF behavior
            X_df = sm.add_constant(X_df, has_constant='add')

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
if fun_mode:
    st.markdown(
        """
        <div style='text-align: center; font-weight: bold; font-size: 3rem; margin-bottom: 1rem;'>
            RHR MODEL BUILDER 𓀒 𓀓 𓀔
        </div>
        """,
        unsafe_allow_html=True
    )
else:
    st.markdown(
        """
        <div style='text-align: center; font-weight: bold; font-size: 3rem; margin-bottom: 1rem;'>
            RHR MODEL BUILDER 
        </div>
        """,
        unsafe_allow_html=True
    )

# Navigation buttons
st.markdown("### 🧭 Analysis Workflow")
workflow_steps = [
    ('overview', '📊 Data Overview'),
    ('dtype', '🔧 Data Types'),
    ('filter', '🔍 Filtering'),
    ('clean', '🧹 Cleaning'),
    ('transform', '⚡ Transform'),
    ('model', '📈 OLS Model'),
    ('ml', '🤖 ML Models'),
    ('hybrid', '🔗 Hybrid Model')
]

cols = st.columns(len(workflow_steps))
for i, (step_key, step_name) in enumerate(workflow_steps):
    with cols[i]:
        if st.button(step_name, key=f"nav_{step_key}", 
                    type="primary" if st.session_state.processing_step == step_key else "secondary"):
            st.session_state.processing_step = step_key

# Auto-connect to database on first load
if 'data_initialized' not in st.session_state:
    st.session_state.data_initialized = False

if not st.session_state.data_initialized:
    with st.spinner("Connecting to database..."):
        success, message = analyzer.connect_database()
        if success:
            st.markdown(f'<div class="success-box">✅ {message}</div>', unsafe_allow_html=True)
            # Auto-load data
            with st.spinner("Loading property data..."):
                data_success, data_message = analyzer.load_property_data()
                if data_success:
                    st.markdown(f'<div class="success-box">📊 {data_message}</div>', unsafe_allow_html=True)
                    st.session_state.data_initialized = True  # Mark as initialized
                else:
                    st.markdown(f'<div class="error-box">❌ {data_message}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="error-box">❌ {message}</div>', unsafe_allow_html=True)
            st.stop()

# Display current data status
if analyzer.current_data is not None:
    st.markdown("### 📋 Current Data Status")
    col1, col2, col3, col4, col5 = st.columns([1,1,1,1,1.5])
    
    # Use cached metrics
    if 'cached_data_metrics' not in st.session_state or st.session_state.get('data_changed', True):
        # Only calculate when data actually changes
        memory_sum = analyzer.current_data.memory_usage(deep=True).sum()
        missing_sum = analyzer.current_data.isnull().sum().sum()
        total_cells = analyzer.current_data.shape[0] * analyzer.current_data.shape[1]
        
        st.session_state.cached_data_metrics = {
            'memory_mb': memory_sum / 1024**2,
            'missing_pct': (missing_sum / total_cells * 100) if total_cells > 0 else 0
        }
        st.session_state.data_changed = False
    
    metrics = st.session_state.cached_data_metrics
    
    with col1:
        st.metric("Properties", f"{len(analyzer.current_data):,}")
    with col2:
        st.metric("Columns", len(analyzer.current_data.columns))
    with col3:
        st.metric("Completeness", f"{100-metrics['missing_pct']:.1f}%")
    with col4:
        st.metric("Memory", f"{metrics['memory_mb']:.1f} MB")
    
    # Excel download (unchanged)
    with col5:
        if 'excel_data' not in st.session_state:
            st.session_state.excel_data = None

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
        # Invalidate all caches
        st.session_state.data_changed = True
        st.session_state.show_overview_stats = False
        st.session_state.show_dtype_table = False
        if 'cached_data_metrics' in st.session_state:
            del st.session_state.cached_data_metrics
        st.success(message)
        st.rerun()

st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

# Step-based interface
if st.session_state.processing_step == 'overview':
    if fun_mode:
        st.markdown('## <img src="https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExaXZod3R3NnJ2cW93MjkycXJ3dTRxeHluYXlkemhwdnVyZTFmOWhibyZlcD12MV9naWZzX3RyZW5kaW5nJmN0PWc/0GtVKtagi2GvWuY3vm/giphy.gif" alt="data gif" style="height:96px; vertical-align:middle;"> Data Overview', unsafe_allow_html=True)
    else:
        st.markdown('## Data Overview')

    if analyzer.current_data is not None:
        # Data preview (lightweight)
        st.markdown("### 📋 Data Preview")
        st.dataframe(analyzer.current_data.head(10), use_container_width=True)
        
        # LAZY LOAD HEAVY OPERATIONS - Only show when requested
        if st.button("📊 Show Detailed Statistics") or st.session_state.get('show_overview_stats', False):
            st.session_state.show_overview_stats = True
            
            # Basic statistics for numeric columns (now only when requested)
            numeric_data = analyzer.current_data.select_dtypes(include=[np.number])
            if not numeric_data.empty:
                st.markdown("### 📈 Numeric Columns Statistics")
                st.dataframe(numeric_data.describe(), use_container_width=True)
            
            # Data info (now only when requested)
            st.markdown("### ℹ️ Column Information")
            info_df = pd.DataFrame({
                'Column': analyzer.current_data.columns,
                'Data Type': [str(dtype) for dtype in analyzer.current_data.dtypes],
                'Non-Null Count': analyzer.current_data.count(),
                'Null Count': analyzer.current_data.isnull().sum(),
                'Unique Values': [analyzer.current_data[col].nunique() for col in analyzer.current_data.columns]
            })
            st.dataframe(info_df, use_container_width=True)
        
        elif not st.session_state.get('show_overview_stats', False):
            st.info("👆 Click 'Show Detailed Statistics' to view numeric summaries and column information")

elif st.session_state.processing_step == 'dtype':
    if fun_mode:
        st.markdown('## <img src="https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExeDltcWZoZ3dsNTVzZm5xMWR5bXExbGx0cG14eWdudGdpanJtZjBnMCZlcD12MV9naWZzX3NlYXJjaCZjdD1n/xCCqt6qDewWf6zriPX/giphy.gif" alt="data gif" style="height:72px; vertical-align:middle;"> Data Type Management', unsafe_allow_html=True)
    else:
        st.markdown('## Data Type Management')

    if analyzer.current_data is not None:
        st.markdown("### Change Column Data Types")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            selected_column = st.selectbox("Select Column", analyzer.current_data.columns)
        
        with col2:
            current_dtype = str(analyzer.current_data[selected_column].dtype)
            st.info(f"Current type: {current_dtype}")
            new_dtype = st.selectbox("New Data Type", ['numeric', 'categorical', 'string', 'datetime'])
        
        with col3:
            st.write("")
            st.write("")
            if st.button("Apply Type Change", type="primary"):
                success, message = analyzer.change_dtype(selected_column, new_dtype)
                if success:
                    # Mark data as changed to refresh cache
                    st.session_state.data_changed = True
                    st.success(message)
                    st.rerun()
                else:
                    st.error(message)
        
        # LAZY LOAD DATA TYPES TABLE
        if st.button("📋 Show All Data Types") or st.session_state.get('show_dtype_table', False):
            st.session_state.show_dtype_table = True
            
            st.markdown("### Current Data Types")
            # More efficient sample value extraction
            sample_values = []
            for col in analyzer.current_data.columns:
                non_null_series = analyzer.current_data[col].dropna()
                if len(non_null_series) > 0:
                    sample_values.append(str(non_null_series.iloc[0]))
                else:
                    sample_values.append('N/A')
            
            dtype_df = pd.DataFrame({
                'Column': analyzer.current_data.columns,
                'Data Type': [str(dtype) for dtype in analyzer.current_data.dtypes],
                'Sample Values': sample_values
            })
            st.dataframe(dtype_df, use_container_width=True)
        
        elif not st.session_state.get('show_dtype_table', False):
            st.info("👆 Click 'Show All Data Types' to view complete data type information")

elif st.session_state.processing_step == 'filter':
    if fun_mode:
        st.markdown('## <img src="https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExd3NmZDV4d3Q2Zm00dXNuZ3J5bDZ4ZmVvYWFhaW1wbWdsNGJxeTZ5OCZlcD12MV9naWZzX3NlYXJjaCZjdD1n/eEjf3t9MeTXZM0d91u/giphy.gif" alt="data gif" style="height:72px; vertical-align:middle;"> Geographic & Data Filtering', unsafe_allow_html=True)
    else:
        st.markdown('## Geographic & Data Filtering')

    if analyzer.current_data is not None:
        # Initialize selected_filters dictionary
        # Initialize selected_filters dictionary
        selected_filters = {}

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
                    st.rerun()
                else:
                    st.error(message)
            
            if st.button(shortcut_filters['jabodetabek'], use_container_width=True):
                success, message = analyzer.apply_shortcut_filter('jabodetabek')
                if success:
                    st.success(message)
                    st.rerun()
                else:
                    st.error(message)

        with col2:
            if st.button(shortcut_filters['jabodetabek_no_kepulauan_seribu'], use_container_width=True):
                success, message = analyzer.apply_shortcut_filter('jabodetabek_no_kepulauan_seribu')
                if success:
                    st.success(message)
                    st.rerun()
                else:
                    st.error(message)
            
            if st.button(shortcut_filters['bandung'], use_container_width=True):
                success, message = analyzer.apply_shortcut_filter('bandung')
                if success:
                    st.success(message)
                    st.rerun()
                else:
                    st.error(message)

        with col3:
            if st.button(shortcut_filters['bali'], use_container_width=True):
                success, message = analyzer.apply_shortcut_filter('bali')
                if success:
                    st.success(message)
                    st.rerun()
                else:
                    st.error(message)
            
            if st.button(shortcut_filters['surabaya'], use_container_width=True):
                success, message = analyzer.apply_shortcut_filter('surabaya')
                if success:
                    st.success(message)
                    st.rerun()
                else:
                    st.error(message)

        st.markdown("---")

        # Geographic Filtering (Priority Section)
        st.markdown("### 🗺️ Smart Geographic Filtering")

        # Clear geographic filters button
        if st.button("🗑️ Clear Geographic Filters"):
            # Clear the multiselect keys to reset selections
            for key in ['geo_wadmpr', 'geo_wadmkk', 'geo_wadmkc']:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()

        # Find geographic columns
        geo_columns = ['wadmpr', 'wadmkk', 'wadmkc', 'wadmkd']
        available_geo_cols = []

        for col in analyzer.current_data.columns:
            for geo in geo_columns:
                if geo in col:
                    available_geo_cols.append(col)
                    break

        if available_geo_cols:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                # Province selection
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
                        key="geo_wadmpr"
                    )
                    if selected_wadmpr:
                        selected_filters[wadmpr_col] = {'type': 'categorical', 'value': selected_wadmpr}
            
            with col2:
                # Regency selection (filtered by selected provinces)
                wadmkk_col = None
                for col in available_geo_cols:
                    if 'wadmkk' in col:
                        wadmkk_col = col
                        break
                
                if wadmkk_col:
                    # Filter regencies based on selected provinces
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
                        key="geo_wadmkk"
                    )
                    if selected_wadmkk:
                        selected_filters[wadmkk_col] = {'type': 'categorical', 'value': selected_wadmkk}
            
            with col3:
                # District selection (filtered by selected regencies)
                wadmkc_col = None
                for col in available_geo_cols:
                    if 'wadmkc' in col:
                        wadmkc_col = col
                        break
                
                if wadmkc_col:
                    # Filter districts based on selected regencies
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
                        key="geo_wadmkc"
                    )
                    if selected_wadmkc:
                        selected_filters[wadmkc_col] = {'type': 'categorical', 'value': selected_wadmkc}

        # Show current geographic selection summary
        geographic_filters = {k: v for k, v in selected_filters.items() 
                            if any(geo in k for geo in ['wadmpr', 'wadmkk', 'wadmkc', 'wadmkd'])}

        if geographic_filters:
            st.info("📍 **Current Geographic Selection:**")
            for col, config in geographic_filters.items():
                level = "Province" if "wadmpr" in col else "Regency" if "wadmkk" in col else "District" if "wadmkc" in col else "Village"
                st.write(f"- **{level}:** {len(config['value'])} selected")
            
            # Preview geographic filtering result
            preview_geo_df = analyzer.current_data.copy()
            for col, config in geographic_filters.items():
                preview_geo_df = preview_geo_df[preview_geo_df[col].isin(config['value'])]
            
            geo_reduction = len(analyzer.current_data) - len(preview_geo_df)
            if geo_reduction > 0:
                st.success(f"✅ Geographic filters will reduce data by {geo_reduction:,} records → {len(preview_geo_df):,} remaining")
            else:
                st.info("ℹ️ Geographic filters don't reduce the current dataset")
        else:
            st.info("ℹ️ No geographic filters selected")

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
                    # Numeric range filter - using the improved version
                    st.write("**Filter Type: Numeric Range**")
                    
                    col_data = analyzer.current_data[col].dropna()
                    data_min = float(col_data.min())
                    data_max = float(col_data.max())
                    data_mean = float(col_data.mean())
                    data_median = float(col_data.median())
                    
                    # Show current data statistics
                    st.info(f"📊 Range: {data_min:,.0f} - {data_max:,.0f} | Mean: {data_mean:,.0f} | Median: {data_median:,.0f}")
                    
                    # Use number_input for better UX
                    min_input = st.number_input(
                        f"Minimum {col}",
                        min_value=float(data_min),
                        max_value=float(data_max),
                        value=float(data_min),
                        key=f"min_input_{col}"
                    )
                    
                    max_input = st.number_input(
                        f"Maximum {col}",
                        min_value=float(data_min),
                        max_value=float(data_max),
                        value=float(data_max),
                        key=f"max_input_{col}"
                    )
                    
                    # Validation and preview
                    if min_input > max_input:
                        st.error("❌ Minimum cannot be greater than maximum.")
                    else:
                        # Preview how many records will be filtered
                        filtered_count = len(col_data[(col_data >= min_input) & (col_data <= max_input)])
                        total_count = len(col_data)
                        percentage = (filtered_count / total_count) * 100 if total_count > 0 else 0
                        
                        st.success(f"✅ Filter Preview: {filtered_count:,} / {total_count:,} records ({percentage:.1f}%)")
                        
                        # Only apply filter if it actually changes the range
                        if min_input != data_min or max_input != data_max:
                            selected_filters[col] = {
                                'type': 'numeric_range', 
                                'value': (float(min_input), float(max_input))
                            }

            with col2:
                # Show column statistics
                st.write("**Column Statistics:**")
                st.write(f"- Data Type: {col_dtype}")
                st.write(f"- Unique Values: {unique_count:,}")
                st.write(f"- Missing Values: {analyzer.current_data[col].isnull().sum():,}")

                if np.issubdtype(col_dtype, np.number):
                    col_data = analyzer.current_data[col]
                    st.write("**Sample Values:**")
                    st.write(col_data.nlargest(5).tolist())

        # Apply filters
        if selected_filters and st.button("🔍 Apply Manual Filters", type="primary"):
            with st.spinner("Applying filters..."):
                success, message = analyzer.apply_flexible_filters(selected_filters)
                
                if success:
                    st.success(message)
                    st.rerun()
                else:
                    st.error(f"❌ Filter failed: {message}")

        # Show active filters
        if selected_filters:
            st.markdown("### Active Manual Filters")
            for col, filter_config in selected_filters.items():
                if any(geo in col for geo in geo_columns):
                    st.write(f"🗺️ **{col}:** {len(filter_config['value'])} selected")
                else:
                    st.write(f"**{col}:** {filter_config['type']} = {filter_config['value']}")

        if not selected_filters:
            st.info("No manual filters applied - use quick filters above or add manual filters below")

        
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
    if fun_mode:
        st.markdown('## <img src="https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExajh0eW9iZTMxZDZpMGxzbTgxanVicXM4b2YybW5zdDR2cHQzMjFnMiZlcD12MV9naWZzX3NlYXJjaCZjdD1n/NV4cSrRYXXwfUcYnua/giphy.gif" alt="data gif" style="height:96px; vertical-align:middle;"> Data Cleaning', unsafe_allow_html=True)
    else:
        st.markdown('## Data Cleaning')

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
    if fun_mode:
        st.markdown('## <img src="https://media.giphy.com/media/v1.Y2lkPWVjZjA1ZTQ3Y25yOXB5MDFqNGlmdmJnenFqandjMzl6YnJscnRseDlzN2poZG1wMiZlcD12MV9naWZzX3NlYXJjaCZjdD1n/EjLTU9HAnnskywtJ9j/giphy.gif" alt="data gif" style="height:96px; vertical-align:middle;"> Variable Transformations', unsafe_allow_html=True)
    else:
        st.markdown('## Variable Transformations')

    if analyzer.current_data is not None:
        st.markdown("### Apply Transformations")
        st.info("Select columns to transform. New transformed columns will be added to your dataset and available for OLS modeling.")
        
        # Get numeric columns
        numeric_columns = analyzer.current_data.select_dtypes(include=[np.number]).columns.tolist()
        
        if not numeric_columns:
            st.warning("No numeric columns available for transformation")
        else:
            # Shortcut for distance columns
            st.markdown("### 🚀 Quick Distance + (HPM & Luas Tanah) Transformations")
            distance_columns = [col for col in numeric_columns if 'distance_to_' in col.lower()]
            distance_columns.append('hpm')
            distance_columns.append('luas_tanah')
            
            if distance_columns:
                st.info(f"Found {len(distance_columns)} distance columns: {', '.join(distance_columns)}")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    distance_transform = st.selectbox("Transform all distance + (HPM & Luas Tanah) columns", 
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
    if fun_mode:
        st.markdown('## <img src="https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExd3phdHl3Zm0zZTZoZ2RkdHc5Z3NncHIzaXpjdWI4bmw1YzluMm0ydiZlcD12MV9naWZzX3NlYXJjaCZjdD1n/9ADoZQgs0tyww/giphy.gif" alt="data gif" style="height:72px; vertical-align:middle;"> OLS Regression Analysis', unsafe_allow_html=True)
    else:
        st.markdown('## OLS Regression Analysis')

    if analyzer.current_data is not None:
        # Model configuration
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Model Variables:**")
            
            # Get all numeric columns (including transformed ones)
            numeric_columns = analyzer.current_data.select_dtypes(include=[np.number]).columns.tolist()
            
            # Target variable (Y)
            y_column = st.selectbox("Dependent Variable (Y)", numeric_columns, 
                                   index=numeric_columns.index('ln_hpm') if 'ln_hpm' in numeric_columns else 0)
            
            # Independent variables (X)
            available_x_cols = [col for col in numeric_columns if col != y_column]
            st.markdown("**Independent Variables (X):**")
            x_columns = []
            for col in available_x_cols:
                if st.checkbox(col, value=(col in available_x_cols[:5] if len(available_x_cols) >= 5 else True), key=f"x_var_{col}"):
                    x_columns.append(col)

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

                        # Model summary
                        with st.expander("📋 Detailed Model Summary", expanded=False):
                            st.code(results['summary'])
                        
                        col1, col2,= st.columns(2)
                        with col1:
                            st.metric("R-squared", f"{results['rsquared']:.4f}")
                        with col2:
                            st.metric("Adj. R-squared", f"{results['rsquared_adj']:.4f}")
                        
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
                        
                        # VIF Results
                        if vif_df is not None:
                            st.markdown("### 📊 Variance Inflation Factors (VIF)")
                            st.dataframe(vif_df.style.format({'VIF': '{:.2f}'}), use_container_width=True)
                            
                            # VIF interpretation
                            high_vif = vif_df[vif_df['VIF'] > 5]
                            if not high_vif.empty:
                                st.warning(f"High VIF detected (>5): {', '.join(high_vif['feature'].tolist())}")
                        
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
    if fun_mode:
        st.markdown('## <img src="https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExaWZoeTByeWI1YmdsMHU3dnJ3ejNnem04MmM4Zjh5eThvbG10ZjFiaCZlcD12MV9naWZzX3NlYXJjaCZjdD1n/Gf1RA1jNSpbbuDE40m/giphy.gif" alt="data gif" style="height:96px; vertical-align:middle;"> Machine Learning Models', unsafe_allow_html=True)
    else:
        st.markdown('## Machine Learning Models')

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

elif st.session_state.processing_step == 'hybrid':
    if fun_mode:
        st.markdown('## <img src="https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExbXA4M2Z5aGp3eXpwZGVlZmVwZjR4cjFrbmJvdHY0ODJxZWtvdDZmaiZlcD12MV9naWZzX3NlYXJjaCZjdD1n/3oKIPEqDGUULpEU0aQ/giphy.gif" alt="hybrid gif" style="height:96px; vertical-align:middle;"> Hybrid OLS + Random Forest Model', unsafe_allow_html=True)
    else:
        st.markdown('## Hybrid OLS + Random Forest Model')
    
    # Explanation of the hybrid approach
    st.info("🔗 **Hybrid Approach**: Combines the strengths of linear and nonlinear modeling")
    
    with st.expander("📚 How the Hybrid Model Works", expanded=False):
        st.markdown("""
        **Step-by-Step Process:**
        
        1. **Fit OLS Model**: `y_pred_ols = OLS(X, y)` - Captures main linear relationships
        2. **Calculate Residuals**: `residuals = y_actual - y_pred_ols` - What OLS couldn't explain  
        3. **Train RF on Residuals**: `RF.fit(X, residuals)` - Captures nonlinear patterns in residuals
        4. **Final Prediction**: `y_final = y_pred_ols + rf_pred_residuals` - Combines both models
        
        **Result**: Linear interpretability + Nonlinear flexibility
        """)
    
    if analyzer.current_data is not None:
        # Model Configuration
        st.markdown("### 🎯 Model Configuration")
        
        # Get all numeric columns (including transformed ones)
        numeric_columns = analyzer.current_data.select_dtypes(include=[np.number]).columns.tolist()
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Target variable (Y)
            hybrid_y_column = st.selectbox("Target Variable (Y)", numeric_columns, 
                                          index=numeric_columns.index('ln_hpm') if 'ln_hpm' in numeric_columns else 0,
                                          key="hybrid_y_select")
        
        with col2:
            # Cross-validation settings
            hybrid_n_splits = st.number_input("CV Folds", min_value=3, max_value=20, value=5, key="hybrid_cv_folds")
        
        # Independent variables (X) - using checkbox approach for full names
        st.markdown("**Independent Variables (X):**")
        available_x_cols = [col for col in numeric_columns if col != hybrid_y_column]
        
        # Create columns for checkboxes (3 columns to save space)
        checkbox_cols = st.columns(3)
        hybrid_x_columns = []
        
        for i, col in enumerate(available_x_cols):
            with checkbox_cols[i % 3]:
                if st.checkbox(col, value=(i < 5), key=f"hybrid_x_{col}"):  # Default first 5 selected
                    hybrid_x_columns.append(col)
        
        if hybrid_x_columns:
            st.success(f"✅ Selected {len(hybrid_x_columns)} features for hybrid model")
        
        # Hyperparameter Configuration
        st.markdown("### ⚙️ Random Forest Parameters (for Residuals)")
        
        param_col1, param_col2, param_col3 = st.columns(3)
        
        with param_col1:
            hybrid_n_estimators = st.number_input("n_estimators", min_value=10, max_value=500, value=100, key="hybrid_n_est")
            hybrid_max_depth = st.number_input("max_depth", min_value=3, max_value=30, value=10, key="hybrid_max_depth")
        
        with param_col2:
            hybrid_min_samples_split = st.number_input("min_samples_split", min_value=2, max_value=20, value=5, key="hybrid_min_split")
            hybrid_min_samples_leaf = st.number_input("min_samples_leaf", min_value=1, max_value=10, value=2, key="hybrid_min_leaf")
        
        with param_col3:
            hybrid_max_features = st.selectbox("max_features", ['sqrt', 'log2', None], index=0, key="hybrid_max_feat")
            hybrid_random_state = st.number_input("random_state", min_value=1, max_value=1000, value=42, key="hybrid_random")
        
        # Train Hybrid Model
        st.markdown("### 🚀 Train Hybrid Model")
        
        if st.button("🔗 Train Hybrid OLS + RF Model", type="primary") and hybrid_x_columns:
            try:
                with st.spinner("Training hybrid model..."):
                    # Prepare data
                    model_vars = [hybrid_y_column] + hybrid_x_columns
                    df_model = analyzer.current_data[model_vars].dropna()
                    
                    X = df_model[hybrid_x_columns]
                    y = df_model[hybrid_y_column]
                    
                    st.info(f"Training on {len(df_model):,} observations with {len(hybrid_x_columns)} features")
                    
                    # Cross-validation setup
                    from sklearn.model_selection import KFold
                    kf = KFold(n_splits=hybrid_n_splits, shuffle=True, random_state=hybrid_random_state)
                    
                    # Results storage
                    results = {
                        'Fold': [],
                        'OLS_R2': [], 'OLS_MAPE': [], 'OLS_FSD': [], 'OLS_PE10': [], 'OLS_RT20': [],
                        'Hybrid_R2': [], 'Hybrid_MAPE': [], 'Hybrid_FSD': [], 'Hybrid_PE10': [], 'Hybrid_RT20': [],
                        'RF_Residual_R2': []  # Just for diagnostic purposes
                    }
                    
                    fold_predictions = []
                    progress_bar = st.progress(0)
                    
                    # Cross-validation loop
                    for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
                        progress_bar.progress((fold + 1) / hybrid_n_splits)
                        
                        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
                        
                        # STEP 1: Fit OLS Model
                        X_train_ols = sm.add_constant(X_train)
                        X_test_ols = sm.add_constant(X_test)
                        
                        ols_model = sm.OLS(y_train, X_train_ols).fit()
                        ols_pred_train = ols_model.predict(X_train_ols)
                        ols_pred_test = ols_model.predict(X_test_ols)
                        
                        # STEP 2: Calculate OLS Residuals
                        residuals_train = y_train - ols_pred_train
                        residuals_test = y_test - ols_pred_test  # True residuals for evaluation
                        
                        # STEP 3: Train Random Forest on Residuals
                        rf_model = RandomForestRegressor(
                            n_estimators=hybrid_n_estimators,
                            max_depth=hybrid_max_depth,
                            min_samples_split=hybrid_min_samples_split,
                            min_samples_leaf=hybrid_min_samples_leaf,
                            max_features=hybrid_max_features,
                            random_state=hybrid_random_state
                        )
                        
                        # Train RF to predict residuals
                        rf_model.fit(X_train, residuals_train)
                        rf_pred_residuals_test = rf_model.predict(X_test)
                        
                        # STEP 4: Make Final Hybrid Prediction
                        hybrid_pred_test = ols_pred_test + rf_pred_residuals_test
                        
                        # STEP 5: Evaluate Models (Only OLS and Hybrid on actual target)
                        ols_metrics = evaluate(y_test, ols_pred_test, squared=False)
                        hybrid_metrics = evaluate(y_test, hybrid_pred_test, squared=False)
                        
                        # Diagnostic: How well does RF predict the true residuals?
                        rf_residual_r2 = r2_score(residuals_test, rf_pred_residuals_test)
                        
                        # Store results
                        results['Fold'].append(f"Fold-{fold + 1}")
                        
                        # OLS metrics
                        results['OLS_R2'].append(ols_metrics['R2'])
                        results['OLS_MAPE'].append(ols_metrics['MAPE'])
                        results['OLS_FSD'].append(ols_metrics['FSD'])
                        results['OLS_PE10'].append(ols_metrics['PE10'])
                        results['OLS_RT20'].append(ols_metrics['RT20'])
                        
                        # Hybrid metrics
                        results['Hybrid_R2'].append(hybrid_metrics['R2'])
                        results['Hybrid_MAPE'].append(hybrid_metrics['MAPE'])
                        results['Hybrid_FSD'].append(hybrid_metrics['FSD'])
                        results['Hybrid_PE10'].append(hybrid_metrics['PE10'])
                        results['Hybrid_RT20'].append(hybrid_metrics['RT20'])
                        
                        # Diagnostic metric
                        results['RF_Residual_R2'].append(rf_residual_r2)
                        
                        # Store predictions for plotting (last fold)
                        if fold == hybrid_n_splits - 1:
                            fold_predictions = {
                                'y_actual': y_test,
                                'ols_pred': ols_pred_test,
                                'hybrid_pred': hybrid_pred_test,
                                'actual_residuals': residuals_test,
                                'rf_pred_residuals': rf_pred_residuals_test
                            }
                    
                    progress_bar.empty()
                    
                    # Calculate average metrics
                    avg_metrics = {
                        'OLS': {
                            'R2': np.mean(results['OLS_R2']),
                            'MAPE': np.mean(results['OLS_MAPE']),
                            'FSD': np.mean(results['OLS_FSD']),
                            'PE10': np.mean(results['OLS_PE10']),
                            'RT20': np.mean(results['OLS_RT20'])
                        },
                        'Hybrid': {
                            'R2': np.mean(results['Hybrid_R2']),
                            'MAPE': np.mean(results['Hybrid_MAPE']),
                            'FSD': np.mean(results['Hybrid_FSD']),
                            'PE10': np.mean(results['Hybrid_PE10']),
                            'RT20': np.mean(results['Hybrid_RT20'])
                        },
                        'RF_Residual_R2_avg': np.mean(results['RF_Residual_R2'])
                    }
                    
                    # Store in session state
                    st.session_state.hybrid_results = results
                    st.session_state.hybrid_avg_metrics = avg_metrics
                    st.session_state.hybrid_predictions = fold_predictions
                    st.session_state.hybrid_model_info = {
                        'target': hybrid_y_column,
                        'features': hybrid_x_columns,
                        'n_observations': len(df_model),
                        'n_splits': hybrid_n_splits,
                        'rf_params': {
                            'n_estimators': hybrid_n_estimators,
                            'max_depth': hybrid_max_depth,
                            'min_samples_split': hybrid_min_samples_split,
                            'min_samples_leaf': hybrid_min_samples_leaf,
                            'max_features': hybrid_max_features,
                            'random_state': hybrid_random_state
                        }
                    }
                
                st.success("✅ Hybrid model training completed!")
                
                # Performance Improvement Analysis
                improvement_r2 = avg_metrics['Hybrid']['R2'] - avg_metrics['OLS']['R2']
                improvement_mape = avg_metrics['OLS']['MAPE'] - avg_metrics['Hybrid']['MAPE']
                improvement_fsd = avg_metrics['OLS']['FSD'] - avg_metrics['Hybrid']['FSD']
                
                if improvement_r2 > 0.01:  # Meaningful improvement
                    st.success(f"🎉 **Significant Improvement!** Hybrid model improved R² by {improvement_r2:.4f} ({improvement_r2/avg_metrics['OLS']['R2']*100:.1f}% relative improvement)")
                elif improvement_r2 > 0.001:
                    st.info(f"📊 **Modest Improvement**: Hybrid model improved R² by {improvement_r2:.4f}")
                else:
                    st.warning(f"⚠️ **Limited Improvement**: R² change = {improvement_r2:.4f}. Linear relationships may dominate this dataset.")
                
                # Display Results
                st.markdown("### 📊 Model Comparison Results")
                
                # Main metrics comparison
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown("**🔵 OLS Only**")
                    st.metric("R²", f"{avg_metrics['OLS']['R2']:.4f}")
                    st.metric("MAPE", f"{avg_metrics['OLS']['MAPE']:.4f}")
                    st.metric("FSD", f"{avg_metrics['OLS']['FSD']:.4f}")
                    st.metric("PE10", f"{avg_metrics['OLS']['PE10']:.4f}")
                    st.metric("RT20", f"{avg_metrics['OLS']['RT20']:.4f}")
                
                with col2:
                    st.markdown("**🟡 Hybrid (OLS + RF)**")
                    st.metric("R²", f"{avg_metrics['Hybrid']['R2']:.4f}", 
                             delta=f"{improvement_r2:+.4f}")
                    st.metric("MAPE", f"{avg_metrics['Hybrid']['MAPE']:.4f}", 
                             delta=f"{-improvement_mape:+.4f}")
                    st.metric("FSD", f"{avg_metrics['Hybrid']['FSD']:.4f}",
                             delta=f"{-improvement_fsd:+.4f}")
                    st.metric("PE10", f"{avg_metrics['Hybrid']['PE10']:.4f}",
                             delta=f"{avg_metrics['Hybrid']['PE10'] - avg_metrics['OLS']['PE10']:+.4f}")
                    st.metric("RT20", f"{avg_metrics['Hybrid']['RT20']:.4f}",
                             delta=f"{avg_metrics['Hybrid']['RT20'] - avg_metrics['OLS']['RT20']:+.4f}")
                
                with col3:
                    st.markdown("**🔍 Diagnostics**")
                    st.metric("RF Residual R²", f"{avg_metrics['RF_Residual_R2_avg']:.4f}")
                    st.info("This shows how well Random Forest captures the patterns in OLS residuals")
                    
                    if avg_metrics['RF_Residual_R2_avg'] > 0.1:
                        st.success("✅ RF found meaningful patterns in residuals")
                    else:
                        st.warning("⚠️ RF found limited patterns in residuals")
                
                # Detailed results table
                st.markdown("### 📈 Detailed Cross-Validation Results")
                
                # Create clean results dataframe
                display_results = pd.DataFrame({
                    'Fold': results['Fold'],
                    'OLS_R²': results['OLS_R2'],
                    'OLS_MAPE': results['OLS_MAPE'],
                    'OLS_FSD': results['OLS_FSD'],
                    'Hybrid_R²': results['Hybrid_R2'],
                    'Hybrid_MAPE': results['Hybrid_MAPE'],
                    'Hybrid_FSD': results['Hybrid_FSD'],
                    'R²_Improvement': [h - o for h, o in zip(results['Hybrid_R2'], results['OLS_R2'])],
                    'RF_Residual_R²': results['RF_Residual_R2']
                })
                
                st.dataframe(display_results.style.format({
                    'OLS_R²': '{:.4f}',
                    'OLS_MAPE': '{:.4f}',
                    'OLS_FSD': '{:.4f}',
                    'Hybrid_R²': '{:.4f}',
                    'Hybrid_MAPE': '{:.4f}',
                    'Hybrid_FSD': '{:.4f}',
                    'R²_Improvement': '{:+.4f}',
                    'RF_Residual_R²': '{:.4f}'
                }), use_container_width=True)
                
                # Visualization
                st.markdown("### 📊 Model Performance Visualization")
                
                # Create comparison plots
                fig, axes = plt.subplots(2, 2, figsize=(12, 10))
                
                # Plot 1: OLS vs Actual
                axes[0,0].scatter(fold_predictions['y_actual'], fold_predictions['ols_pred'], alpha=0.6, color='blue')
                axes[0,0].plot([fold_predictions['y_actual'].min(), fold_predictions['y_actual'].max()], 
                              [fold_predictions['y_actual'].min(), fold_predictions['y_actual'].max()], 'r--', lw=2)
                axes[0,0].set_xlabel('Actual Values')
                axes[0,0].set_ylabel('OLS Predictions')
                axes[0,0].set_title(f'OLS Only\nR² = {avg_metrics["OLS"]["R2"]:.3f}')
                
                # Plot 2: Hybrid vs Actual
                axes[0,1].scatter(fold_predictions['y_actual'], fold_predictions['hybrid_pred'], alpha=0.6, color='orange')
                axes[0,1].plot([fold_predictions['y_actual'].min(), fold_predictions['y_actual'].max()], 
                              [fold_predictions['y_actual'].min(), fold_predictions['y_actual'].max()], 'r--', lw=2)
                axes[0,1].set_xlabel('Actual Values')
                axes[0,1].set_ylabel('Hybrid Predictions')
                axes[0,1].set_title(f'Hybrid (OLS + RF)\nR² = {avg_metrics["Hybrid"]["R2"]:.3f}')
                
                # Plot 3: RF Residual Prediction vs Actual Residuals
                axes[1,0].scatter(fold_predictions['actual_residuals'], fold_predictions['rf_pred_residuals'], 
                                 alpha=0.6, color='green')
                axes[1,0].plot([fold_predictions['actual_residuals'].min(), fold_predictions['actual_residuals'].max()], 
                              [fold_predictions['actual_residuals'].min(), fold_predictions['actual_residuals'].max()], 'r--', lw=2)
                axes[1,0].set_xlabel('Actual OLS Residuals')
                axes[1,0].set_ylabel('RF Predicted Residuals')
                axes[1,0].set_title(f'RF Residual Prediction\nR² = {avg_metrics["RF_Residual_R2_avg"]:.3f}')
                
                # Plot 4: Improvement per fold
                folds = range(1, len(results['Fold']) + 1)
                improvements = [h - o for h, o in zip(results['Hybrid_R2'], results['OLS_R2'])]
                axes[1,1].bar(folds, improvements, alpha=0.7, color='purple')
                axes[1,1].axhline(y=0, color='black', linestyle='-', alpha=0.3)
                axes[1,1].set_xlabel('Fold')
                axes[1,1].set_ylabel('R² Improvement')
                axes[1,1].set_title('Hybrid Improvement by Fold')
                axes[1,1].set_xticks(folds)
                
                plt.tight_layout()
                st.pyplot(fig)
                
                # Train final models on full dataset for saving
                st.markdown("### 💾 Save Trained Models")
                
                with st.spinner("Training final models on full dataset..."):
                    # Prepare full dataset
                    X_full = df_model[hybrid_x_columns]
                    y_full = df_model[hybrid_y_column]
                    
                    # Train final OLS model
                    X_full_ols = sm.add_constant(X_full)
                    final_ols_model = sm.OLS(y_full, X_full_ols).fit()
                    ols_pred_full = final_ols_model.predict(X_full_ols)
                    residuals_full = y_full - ols_pred_full
                    
                    # Train final RF model on full dataset residuals
                    final_rf_model = RandomForestRegressor(
                        n_estimators=hybrid_n_estimators,
                        max_depth=hybrid_max_depth,
                        min_samples_split=hybrid_min_samples_split,
                        min_samples_leaf=hybrid_min_samples_leaf,
                        max_features=hybrid_max_features,
                        random_state=hybrid_random_state
                    )
                    final_rf_model.fit(X_full, residuals_full)
                    
                    # Store final models in session state
                    st.session_state.final_ols_model = final_ols_model
                    st.session_state.final_rf_model = final_rf_model
                    st.session_state.hybrid_feature_names = hybrid_x_columns
                
                st.success("✅ Final models trained and ready for download!")
                
                # Model download section
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    # Download OLS model
                    ols_model_bytes = pickle.dumps(st.session_state.final_ols_model)
                    st.download_button(
                        label="📦 Download OLS Model (.pkl)",
                        data=ols_model_bytes,
                        file_name=f"hybrid_ols_model_{datetime.now().strftime('%Y%m%d_%H%M')}.pkl",
                        mime="application/octet-stream",
                        help="Download the trained OLS model (Step 1 of hybrid)"
                    )
                
                with col2:
                    # Download RF model
                    rf_model_bytes = pickle.dumps(st.session_state.final_rf_model)
                    st.download_button(
                        label="📦 Download RF Model (.pkl)",
                        data=rf_model_bytes,
                        file_name=f"hybrid_rf_model_{datetime.now().strftime('%Y%m%d_%H%M')}.pkl",
                        mime="application/octet-stream",
                        help="Download the Random Forest model trained on OLS residuals (Step 2 of hybrid)"
                    )
                
                with col3:
                    # Download both models as a package
                    hybrid_package = {
                        'ols_model': st.session_state.final_ols_model,
                        'rf_model': st.session_state.final_rf_model,
                        'feature_names': hybrid_x_columns,
                        'target_name': hybrid_y_column,
                        'model_info': st.session_state.hybrid_model_info,
                        'performance_metrics': avg_metrics,
                        'usage_instructions': {
                            'step1': 'Load both models',
                            'step2': 'Get OLS prediction: ols_pred = ols_model.predict(X_with_constant)',
                            'step3': 'Get RF residual prediction: rf_pred = rf_model.predict(X)',
                            'step4': 'Final prediction: final_pred = ols_pred + rf_pred'
                        }
                    }
                    
                    hybrid_package_bytes = pickle.dumps(hybrid_package)
                    st.download_button(
                        label="📦 Download Hybrid Package (.pkl)",
                        data=hybrid_package_bytes,
                        file_name=f"hybrid_model_package_{datetime.now().strftime('%Y%m%d_%H%M')}.pkl",
                        mime="application/octet-stream",
                        help="Download complete hybrid model package with usage instructions"
                    )
                
                # Usage instructions
                with st.expander("📖 How to Use Downloaded Models", expanded=False):
                    st.markdown("""
                    **Using the Hybrid Model Package:**
                    
                    ```python
                    import pickle
                    import pandas as pd
                    import statsmodels.api as sm
                    
                    # Load the hybrid package
                    with open('hybrid_model_package.pkl', 'rb') as f:
                        package = pickle.load(f)
                    
                    ols_model = package['ols_model']
                    rf_model = package['rf_model']
                    feature_names = package['feature_names']
                    
                    # Make predictions on new data
                    def predict_hybrid(new_data):
                        # Ensure features are in correct order
                        X_new = new_data[feature_names]
                        
                        # Step 1: OLS prediction (add constant)
                        X_new_ols = sm.add_constant(X_new)
                        ols_pred = ols_model.predict(X_new_ols)
                        
                        # Step 2: RF residual prediction
                        rf_pred = rf_model.predict(X_new)
                        
                        # Step 3: Combine predictions
                        final_pred = ols_pred + rf_pred
                        
                        return final_pred
                    
                    # Example usage
                    # new_predictions = predict_hybrid(your_new_data)
                    ```
                    
                    **Individual Models:**
                    - **OLS Model**: Use `ols_model.predict(X_with_constant)` 
                    - **RF Model**: Use `rf_model.predict(X)` on same features to get residual predictions
                    """)
                
                # Export options
                st.markdown("### 💾 Export Hybrid Model Results")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    # Export detailed results
                    st.download_button(
                        label="📊 Download Results (.csv)",
                        data=display_results.to_csv(index=False),
                        file_name=f"hybrid_model_results_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                        mime="text/csv"
                    )
                
                with col2:
                    # Export model info
                    model_info = st.session_state.hybrid_model_info.copy()
                    model_info['avg_metrics'] = avg_metrics
                    model_info['improvements'] = {
                        'r2_improvement': improvement_r2,
                        'mape_improvement': improvement_mape,
                        'fsd_improvement': improvement_fsd,
                        'relative_r2_improvement_pct': (improvement_r2/avg_metrics['OLS']['R2']*100) if avg_metrics['OLS']['R2'] > 0 else 0
                    }
                    model_info['timestamp'] = datetime.now().isoformat()
                    
                    st.download_button(
                        label="📋 Download Model Info (.json)",
                        data=json.dumps(model_info, indent=2),
                        file_name=f"hybrid_model_info_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
                        mime="application/json"
                    )
                
                with col3:
                    # Export comparison summary
                    comparison_summary = {
                        'timestamp': datetime.now().isoformat(),
                        'model_comparison': {
                            'OLS': avg_metrics['OLS'],
                            'Hybrid': avg_metrics['Hybrid']
                        },
                        'improvement_analysis': {
                            'r2_improvement': improvement_r2,
                            'mape_improvement': improvement_mape,
                            'relative_r2_improvement_pct': (improvement_r2/avg_metrics['OLS']['R2']*100) if avg_metrics['OLS']['R2'] > 0 else 0,
                            'rf_residual_r2': avg_metrics['RF_Residual_R2_avg'],
                            'recommendation': 'Use Hybrid' if improvement_r2 > 0.01 else 'Consider Hybrid' if improvement_r2 > 0.001 else 'OLS Sufficient'
                        },
                        'model_config': st.session_state.hybrid_model_info
                    }
                    
                    st.download_button(
                        label="📈 Download Analysis (.json)",
                        data=json.dumps(comparison_summary, indent=2),
                        file_name=f"hybrid_analysis_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
                        mime="application/json"
                    )
                    
            except Exception as e:
                st.error(f"Hybrid model training failed: {str(e)}")
                st.code(traceback.format_exc())
        
        elif not hybrid_x_columns:
            st.warning("Please select at least one feature variable to train the hybrid model")
        
        # Show current hybrid model status
        if 'hybrid_results' in st.session_state:
            st.markdown("---")
            st.success("🔗 Hybrid model training completed!")
            
            # Quick model comparison
            with st.expander("📋 Current Hybrid Model Summary", expanded=True):
                avg_metrics = st.session_state.hybrid_avg_metrics
                
                comparison_df = pd.DataFrame({
                    'Model': ['OLS Only', 'Hybrid (OLS + RF)'],
                    'R²': [avg_metrics['OLS']['R2'], avg_metrics['Hybrid']['R2']],
                    'MAPE': [avg_metrics['OLS']['MAPE'], avg_metrics['Hybrid']['MAPE']],
                    'FSD': [avg_metrics['OLS']['FSD'], avg_metrics['Hybrid']['FSD']],
                    'PE10': [avg_metrics['OLS']['PE10'], avg_metrics['Hybrid']['PE10']],
                    'RT20': [avg_metrics['OLS']['RT20'], avg_metrics['Hybrid']['RT20']]
                })
                
                st.dataframe(comparison_df.style.format({
                    'R²': '{:.4f}',
                    'MAPE': '{:.4f}',
                    'FSD': '{:.4f}',
                    'PE10': '{:.4f}',
                    'RT20': '{:.4f}'
                }), use_container_width=True)
                
                improvement = avg_metrics['Hybrid']['R2'] - avg_metrics['OLS']['R2']
                if improvement > 0.01:
                    st.success(f"🎉 **Strong improvement**: R² increased by {improvement:.4f}")
                elif improvement > 0.001:
                    st.info(f"📊 **Modest improvement**: R² increased by {improvement:.4f}")
                else:
                    st.info(f"📏 **Linear model performs well**: R² change = {improvement:.4f}")
    
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
    if fun_mode:
        st.markdown("""
        <h1 style="
            display: flex; 
            flex-direction: column;   /* stack vertically */
            align-items: center; 
            gap: 10px; 
            line-height: 1;
        ">
            <img src="https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExdWUybHNwbXV3cjd5eG9vbXBxcGs4Y3g1Y29neDdwNDQ2emdhYjd3cSZlcD12MV9naWZzX3NlYXJjaCZjdD1n/uSGobo6DnKmfqYygyk/giphy.gif" 
                alt="GIF" 
                style="height: 120px;">
            <span style="display: inline-block; vertical-align: middle;">Current Status</span>
        </h1>
    """, unsafe_allow_html=True)
    else:
        st.markdown('## Current Status')
    
    
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
    st.markdown("## ⚡ Quick Actions!")
    
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

# # Footer
# st.markdown("---")
# st.markdown("""
# <h1 style="display: flex; align-items: center;">
#     <img src="https://kjpp.rhr.co.id/wp-content/uploads/2020/12/LOGO_KJPP_RHR_1_resize.png" 
#          alt="Logo" style="height:48px; margin-right: 20px;">
#     <span style="font-weight: bold; font-size: 1.5rem;"></span>
# </h1>
# """, unsafe_allow_html=True)
