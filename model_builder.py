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
import math
from scipy.stats import randint, uniform, skew

import base64

import folium
from streamlit_folium import st_folium

# ML imports
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import LabelEncoder
import optuna
from sklearn.ensemble import GradientBoostingRegressor
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import onnx

warnings.filterwarnings('ignore')

def initialize_session_state():
    """Initialize session state variables"""
    defaults = {
        'processing_step': 'selection',
        'advanced_step': 'ml',
        'data_changed': False,
        'show_overview_stats': False,
        'show_dtype_table': False,
        'show_saved_vars': False
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

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

fun_mode = st.sidebar.checkbox("DONT CHECK THIS BOX !!!!!!!!", value=False)

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

@st.cache_resource(show_spinner=False, ttl=3600)
def cached_connect_database():
    try:
        db_user = st.secrets["database"]["user"]
        db_password = st.secrets["database"]["password"]
        db_host = st.secrets["database"]["host"]
        db_port = st.secrets["database"]["port"]
        db_name = st.secrets["database"]["database"]
        
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

@st.cache_data(ttl=1800, show_spinner=False)
def cached_load_property_data(_engine):
    """Load data with optimized query and chunking"""
    if _engine is None:
        return pd.DataFrame()
    
    try:
        # Estimate memory usage first
        count_query = """
        SELECT COUNT(*) FROM engineered_property_data 
        WHERE hpm BETWEEN 50000 AND 200000000;
        """
        with _engine.connect() as conn:
            result = conn.execute(text(count_query))
            total_count = result.scalar()
        
        # Warn if too large
        if total_count > 50000:
            st.warning(f"⚠️ Large dataset detected ({total_count:,} records). Loading first 50,000 for performance.")
            limit = 50000
        else:
            limit = min(total_count, 100000)
        
        query = f"""
        SELECT * FROM engineered_property_data 
        WHERE hpm BETWEEN 50000 AND 200000000
        ORDER BY hpm
        LIMIT {limit};
        """
        
        # Use chunking for large datasets
        df_chunks = []
        chunk_size = 10000
        
        for chunk in pd.read_sql(query, _engine, chunksize=chunk_size):
            df_chunks.append(chunk)
        
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
            subset_cols = ['alamat', 'longitude', 'latitude', 'luas_tanah','hpm']
            initial_count = len(cleaned_df)
            cleaned_df = cleaned_df.drop_duplicates(subset=subset_cols, keep='first')
            print(f"Removed {initial_count - len(cleaned_df)} duplicates based on {subset_cols}")
        
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
        if cleaning_options.get('remove_outliers', False) and cleaning_options.get('outlier_column') and cleaning_options.get('group_column'):
            outlier_col = cleaning_options['outlier_column']
            group_col = cleaning_options['group_column']

            if outlier_col in cleaned_df.columns and group_col in cleaned_df.columns:
                # Prepare lists to store valid indexes (non-outliers)
                valid_idx = []

                # Parameters for trimming and skew threshold
                p10_quantile = 0.1
                p90_quantile = 0.9
                skew_threshold = 0.5

                # Process group-wise
                for group in cleaned_df[group_col].unique():
                    group_df = cleaned_df[cleaned_df[group_col] == group]
                    values = group_df[outlier_col]

                    # Calculate 10th and 90th quantiles
                    p10 = np.quantile(values, p10_quantile)
                    p90 = np.quantile(values, p90_quantile)

                    # Trim data
                    trimmed_vals = values[(values > p10) & (values < p90)]

                    if len(trimmed_vals) == 0:
                        # If no trimmed data, keep all by default
                        valid_idx.extend(group_df.index.tolist())
                        continue

                    std = np.std(trimmed_vals)
                    mean = np.mean(trimmed_vals)
                    median = np.median(trimmed_vals)
                    skew_score = skew(trimmed_vals, axis=0, bias=True)

                    if std == 0:
                        # No variation, keep all
                        valid_idx.extend(group_df.index.tolist())
                        continue

                    center = median if abs(skew_score) > skew_threshold else mean

                    bottom = (center - p10) / std
                    lower_border = center - math.ceil(bottom) * std

                    upper = (p90 - center) / std
                    upper_border = center + math.ceil(upper) * std

                    # Select non-outliers for this group
                    group_valid_idx = group_df[
                        (group_df[outlier_col] >= lower_border) & (group_df[outlier_col] <= upper_border)
                    ].index.tolist()

                    valid_idx.extend(group_valid_idx)

                # Filter cleaned_df to keep only non-outliers
                cleaned_df = cleaned_df.loc[valid_idx]
        
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
    
    def load_filtered_property_data(self, filter_type=None):
        """Load data with database-level filtering for better performance"""
        try:
            # Base query with common filters
            base_query = """
            SELECT * FROM engineered_property_data 
            WHERE hpm BETWEEN 50000 AND 200000000
            """
            
            # Add geographic filters at database level
            if filter_type == 'bodebek':
                geographic_filter = """
                AND wadmpr IN ('Jawa Barat', 'Banten')
                AND wadmkk IN ('Bogor', 'Kota Bogor', 'Depok', 'Kota Depok', 
                            'Tangerang', 'Kota Tangerang', 'Kota Tangerang Selatan',
                            'Bekasi', 'Kota Bekasi')
                """
            elif filter_type == 'jabodetabek':
                geographic_filter = """
                AND wadmpr IN ('Jawa Barat', 'Banten', 'DKI Jakarta')
                AND wadmkk IN ('Kota Administrasi Jakarta Selatan', 'Kota Administrasi Jakarta Utara', 
                            'Kota Bekasi', 'Kota Depok', 'Kota Administrasi Jakarta Barat', 
                            'Bogor', 'Kota Administrasi Jakarta Pusat', 'Kota Bogor', 'Bekasi', 
                            'Tangerang', 'Kota Administrasi Jakarta Timur', 'Kota Tangerang', 
                            'Kota Tangerang Selatan', 'Administrasi Kepulauan Seribu')
                """
            elif filter_type == 'jabodetabek_no_kepulauan_seribu':
                geographic_filter = """
                AND wadmpr IN ('Jawa Barat', 'Banten', 'DKI Jakarta')
                AND wadmkk IN ('Kota Administrasi Jakarta Selatan', 'Kota Administrasi Jakarta Utara', 
                            'Kota Bekasi', 'Kota Depok', 'Kota Administrasi Jakarta Barat', 
                            'Bogor', 'Kota Administrasi Jakarta Pusat', 'Kota Bogor', 'Bekasi', 
                            'Tangerang', 'Kota Administrasi Jakarta Timur', 'Kota Tangerang', 
                            'Kota Tangerang Selatan')
                """
            elif filter_type == 'bandung':
                geographic_filter = """
                AND wadmpr IN ('Jawa Barat')
                AND wadmkk IN ('Kota Bandung', 'Sumedang', 'Kota Cimahi', 'Bandung', 'Bandung Barat')
                """
            elif filter_type == 'bali':
                geographic_filter = """
                AND wadmpr IN ('Bali')
                AND wadmkk IN ('Kota Denpasar', 'Badung', 'Gianyar', 'Tabanan')
                """
            elif filter_type == 'surabaya':
                geographic_filter = """
                AND wadmpr IN ('Jawa Timur')
                AND wadmkk IN ('Kota Surabaya')
                """
            else:
                geographic_filter = ""
            
            # Combine query
            final_query = base_query + geographic_filter + " ORDER BY hpm LIMIT 50000;"
            
            # Execute query
            with self.engine.connect() as conn:
                # Get count first for user feedback
                count_query = base_query.replace("SELECT *", "SELECT COUNT(*)") + geographic_filter
                result = conn.execute(text(count_query))
                total_count = result.scalar()
                
                if total_count > 50000:
                    st.warning(f"⚠️ Found {total_count:,} records. Loading first 50,000 for performance.")
                else:
                    st.success(f"✅ Loading {total_count:,} records matching your criteria.")
            
            # Load the filtered data
            df = pd.read_sql(final_query, self.engine)
            
            self.original_data = df
            self.current_data = df.copy()
            
            return True, f"Loaded {len(df):,} properties for {filter_type or 'all data'}"
            
        except Exception as e:
            return False, f"Failed to load filtered data: {str(e)}"
    
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
                        # Check if already exists
                        if new_col in transformed_df.columns:
                            st.warning(f"Column {new_col} already exists. Overwriting.")
                        transformed_df[new_col] = np.log(transformed_df[col] + 1)
                        self.transformed_columns[col] = new_col
                    elif transform == 'squared':
                        new_col = f'{col}_squared'
                        if new_col in transformed_df.columns:
                            st.warning(f"Column {new_col} already exists. Overwriting.")
                        transformed_df[new_col] = transformed_df[col] ** 2
                        self.transformed_columns[col] = new_col
                    elif transform == 'sqrt':
                        new_col = f'sqrt_{col}'
                        if new_col in transformed_df.columns:
                            st.warning(f"Column {new_col} already exists. Overwriting.")
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

    
    def load_custom_filtered_data(self, provinces=None, regencies=None, districts=None, 
                            hpm_min=50000, hpm_max=200000000, limit=50000):
        """Load data with custom database-level filtering"""
        try:
            query_parts = ["SELECT * FROM engineered_property_data WHERE 1=1"]
            
            # HPM filter
            query_parts.append(f"AND hpm BETWEEN {hpm_min} AND {hpm_max}")
            
            # Geographic filters with proper SQL escaping
            if provinces:
                # Escape single quotes in province names
                escaped_provinces = [p.replace("'", "''") for p in provinces]
                province_list = "', '".join(escaped_provinces)
                query_parts.append(f"AND wadmpr IN ('{province_list}')")
            
            if regencies:
                escaped_regencies = [r.replace("'", "''") for r in regencies]
                regency_list = "', '".join(escaped_regencies)
                query_parts.append(f"AND wadmkk IN ('{regency_list}')")
                
            if districts:
                escaped_districts = [d.replace("'", "''") for d in districts]
                district_list = "', '".join(escaped_districts)
                query_parts.append(f"AND wadmkc IN ('{district_list}')")
            
            # Add ordering and limit
            query_parts.append(f"ORDER BY hpm LIMIT {limit}")
            
            final_query = " ".join(query_parts)
            
            # Get count first for user feedback
            count_query = final_query.replace("SELECT *", "SELECT COUNT(*)").replace(f"ORDER BY hpm LIMIT {limit}", "")
            
            with self.engine.connect() as conn:
                result = conn.execute(text(count_query))
                total_count = result.scalar()
                
                if total_count == 0:
                    return False, "No records found matching your criteria. Please adjust your filters."
                elif total_count > limit:
                    st.warning(f"⚠️ Found {total_count:,} records. Loading first {limit:,} for performance.")
                else:
                    st.info(f"✅ Found {total_count:,} records matching your criteria.")
            
            # Load the filtered data
            df = pd.read_sql(final_query, self.engine)
            
            self.original_data = df
            self.current_data = df.copy()
            
            return True, f"Loaded {len(df):,} properties with custom filters"
            
        except Exception as e:
            return False, f"Custom filtering failed: {str(e)}"

    def get_unique_geographic_values(self, column, parent_filter=None):
        """Get unique values for geographic columns with optional parent filtering"""
        try:
            base_query = f"SELECT DISTINCT {column} FROM engineered_property_data WHERE {column} IS NOT NULL"
            
            if parent_filter:
                if column == 'wadmkk' and 'wadmpr' in parent_filter:
                    provinces = parent_filter['wadmpr']
                    escaped_provinces = [p.replace("'", "''") for p in provinces]
                    province_list = "', '".join(escaped_provinces)
                    base_query += f" AND wadmpr IN ('{province_list}')"
                elif column == 'wadmkc' and 'wadmkk' in parent_filter:
                    regencies = parent_filter['wadmkk']
                    escaped_regencies = [r.replace("'", "''") for r in regencies]
                    regency_list = "', '".join(escaped_regencies)
                    base_query += f" AND wadmkk IN ('{regency_list}')"
            
            base_query += f" ORDER BY {column}"
            
            with self.engine.connect() as conn:
                result = conn.execute(text(base_query))
                values = [row[0] for row in result.fetchall()]
            
            return values
            
        except Exception as e:
            st.error(f"Failed to load {column} options: {str(e)}")
            return []

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
    
    def save_ols_variables(self, y_column, x_columns):
        """Save OLS variables for reuse in ML models"""
        self.saved_ols_variables = {
            'y_column': y_column,
            'x_columns': x_columns,
            'timestamp': datetime.now().isoformat()
        }
        return True, "OLS variables saved successfully!"

    def get_saved_ols_variables(self):
        """Get saved OLS variables"""
        return getattr(self, 'saved_ols_variables', None)
    
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
    
    def export_model_onnx(self, model, feature_names, model_name):
        """Export sklearn model to ONNX format with feature info"""
        try:
            # Define initial input shape (None for batch size, len for features)
            initial_type = [('float_input', FloatTensorType([None, len(feature_names)]))]
            
            # Convert to ONNX
            onnx_model = convert_sklearn(model, initial_types=initial_type)
            
            # Create feature info JSON
            feature_info = {
                'model_name': model_name,
                'feature_names': feature_names,
                'feature_count': len(feature_names),
                'model_type': str(type(model).__name__),
                'timestamp': datetime.now().isoformat(),
                'instructions': {
                    'usage': 'Load ONNX model with onnxruntime and use feature_names for input ordering',
                    'input_shape': [None, len(feature_names)],
                    'input_type': 'float32'
                }
            }
            
            return onnx_model.SerializeToString(), json.dumps(feature_info, indent=2)
            
        except Exception as e:
            return None, f"ONNX export failed: {str(e)}"
    
    def prepare_model_exports(self, model, feature_names, model_name):
        """Prepare ONNX model and feature JSON for download"""
        try:
            # Define initial input shape
            initial_type = [('float_input', FloatTensorType([None, len(feature_names)]))]
            
            # Convert to ONNX
            onnx_model = convert_sklearn(model, initial_types=initial_type)
            
            # Create feature info JSON
            feature_info = {
                'model_name': model_name,
                'feature_names': feature_names,
                'feature_count': len(feature_names),
                'model_type': str(type(model).__name__),
                'timestamp': datetime.now().isoformat(),
                'instructions': {
                    'usage': 'Load ONNX model with onnxruntime and use feature_names for input ordering',
                    'input_shape': [None, len(feature_names)],
                    'input_type': 'float32'
                }
            }
            
            return onnx_model.SerializeToString(), json.dumps(feature_info, indent=2), True
            
        except Exception as e:
            return None, f"ONNX export failed: {str(e)}", False

initialize_session_state()

# Initialize session state
if 'analyzer' not in st.session_state:
    st.session_state.analyzer = RealEstateAnalyzer()

if 'processing_step' not in st.session_state:
    st.session_state.processing_step = 'selection'

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
    ('selection', '📍 Data Selection'),      # NEW - first step
    ('overview', '📊 Data Overview'),
    ('dtype', '🔧 Data Types'),
    ('filter', '🔍 Additional Filters'),     # Renamed
    ('clean', '🧹 Cleaning'),
    ('transform', '⚡ Transform'),
    ('model', '📈 OLS Model'),
    ('advanced', '🤖 Advanced Models')
]

# Disable other navigation if no data loaded
if analyzer.current_data is None and st.session_state.processing_step != 'selection':
    st.session_state.processing_step = 'selection'
    st.warning("⚠️ Please select data first")
    st.rerun()

cols = st.columns(len(workflow_steps))
for i, (step_key, step_name) in enumerate(workflow_steps):
    with cols[i]:
        if st.button(step_name, key=f"nav_{step_key}", 
                    type="primary" if st.session_state.processing_step == step_key else "secondary"):
            st.session_state.processing_step = step_key

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
if st.button("🔄 Reset & Reselect Data", help="Clear all data and start fresh"):
    # Clear everything
    analyzer.current_data = None
    analyzer.original_data = None
    analyzer.model = None
    analyzer.transformed_columns = {}
    
    # Reset to selection step
    st.session_state.processing_step = 'selection'
    
    # Clear caches
    if 'cached_data_metrics' in st.session_state:
        del st.session_state.cached_data_metrics
    
    st.success("Data cleared - ready for new selection")
    st.rerun()

st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

# Step-based interface
if st.session_state.processing_step == 'selection':
    if fun_mode:
        st.markdown('## <img src="https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExZ3I3MnhtZjR3b2FvZ2RkNHowZ3ZyaTVwN21yOG1reW1xYnhmbGFqZCZlcD12MV9naWZzX3NlYXJjaCZjdD1n/f2MC2O2bzX8d9DVnwf/giphy.gif" alt="selection gif" style="height:96px; vertical-align:middle;"> Data Selection', unsafe_allow_html=True)
    else:
        st.markdown('## 📍 Data Selection')
    
    # Auto-connect to database only when needed
    if not analyzer.connection_status:
        with st.spinner("Connecting to database..."):
            success, message = analyzer.connect_database()
            if success:
                st.success(message)
            else:
                st.error(message)
                st.stop()

    # Add this check before any database operations
    if analyzer.engine is None:
        st.error("Database connection failed. Please check your connection settings.")
        st.stop()
    
    st.info("🎯 **Start by selecting your data scope** - this makes analysis faster and more focused")
    
    # Show current data status
    if analyzer.current_data is not None:
        st.warning(f"⚠️ **Data already loaded:** {len(analyzer.current_data):,} properties")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🗑️ Clear Data & Reselect", type="secondary", use_container_width=True):
                # Clear data
                analyzer.current_data = None
                analyzer.original_data = None
                analyzer.model = None
                analyzer.transformed_columns = {}
                
                # Clear caches to free memory
                st.cache_data.clear()
                
                st.rerun()
        
        with col2:
            if st.button("➡️ Continue with Current Data", type="primary", use_container_width=True):
                st.session_state.processing_step = 'overview'
                st.rerun()
        
        st.markdown("---")
    
    # Quick Geographic Filters (same as your current filter section but for selection)
    st.markdown("### 🚀 Quick Geographic Selection")
    st.info("Pre-defined metropolitan areas for quick data loading")

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
        if st.button(shortcut_filters['bodebek'], use_container_width=True, key="sel_bodebek"):
            with st.spinner("Loading BODEBEK data..."):
                success, message = analyzer.load_filtered_property_data('bodebek')  # Only loads BODEBEK data
                if success:
                    st.success(f"✅ {message}")
                    st.session_state.processing_step = 'overview'
                    st.rerun()
                else:
                    st.error(message)
        
        if st.button(shortcut_filters['jabodetabek'], use_container_width=True, key="sel_jabodetabek"):
            with st.spinner("Loading JABODETABEK data..."):
                success, message = analyzer.load_filtered_property_data('jabodetabek')  # Only loads JABODETABEK data
                if success:
                    st.success(f"✅ {message}")
                    st.session_state.processing_step = 'overview'
                    st.rerun()
                else:
                    st.error(message)

    with col2:
        if st.button(shortcut_filters['jabodetabek_no_kepulauan_seribu'], use_container_width=True, key="sel_jabodetabek_no"):
            with st.spinner("Loading JABODETABEK (No Kepulauan Seribu) data..."):
                success, message = analyzer.load_filtered_property_data('jabodetabek_no_kepulauan_seribu')
                if success:
                    st.success(f"✅ {message}")
                    st.session_state.processing_step = 'overview'
                    st.rerun()
                else:
                    st.error(message)
        
        if st.button(shortcut_filters['bandung'], use_container_width=True, key="sel_bandung"):
            with st.spinner("Loading Bandung data..."):
                success, message = analyzer.load_filtered_property_data('bandung')
                if success:
                    st.success(f"✅ {message}")
                    st.session_state.processing_step = 'overview'
                    st.rerun()
                else:
                    st.error(message)

    with col3:
        if st.button(shortcut_filters['bali'], use_container_width=True, key="sel_bali"):
            with st.spinner("Loading Bali data..."):
                success, message = analyzer.load_filtered_property_data('bali')
                if success:
                    st.success(f"✅ {message}")
                    st.session_state.processing_step = 'overview'
                    st.rerun()
                else:
                    st.error(message)
        
        if st.button(shortcut_filters['surabaya'], use_container_width=True, key="sel_surabaya"):
            with st.spinner("Loading Surabaya data..."):
                success, message = analyzer.load_filtered_property_data('surabaya')
                if success:
                    st.success(f"✅ {message}")
                    st.session_state.processing_step = 'overview'
                    st.rerun()
                else:
                    st.error(message)

    st.markdown("---")
    
    # Load All Data Option
    st.markdown("### 📊 Load Full Dataset")
    st.warning("⚠️ **Not recommended for large datasets** - may be slow")
    
    if st.button("📥 Load All Data (Up to 100k records)", type="secondary", use_container_width=True):
        with st.spinner("Loading full dataset..."):
            success, message = analyzer.load_property_data()
            if success:
                st.success(message)
                st.session_state.processing_step = 'overview'
                st.rerun()
            else:
                st.error(message)
    
    st.markdown("---")

    # Advanced Custom Filter Interface
    st.markdown("### 🎛️ Advanced Custom Filters")
    st.info("💡 **Pro Mode**: Build your own custom query with precise geographic and price filters")

    with st.expander("🔧 Custom Filter Builder", expanded=False):
        
        # Geographic Filters Section
        st.markdown("#### 🗺️ Geographic Selection")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**Province (wadmpr)**")
            if st.button("🔄 Load Provinces", key="load_provinces"):
                with st.spinner("Loading province list..."):
                    province_options = analyzer.get_unique_geographic_values('wadmpr')
                    st.session_state.province_options = province_options
            
            if 'province_options' in st.session_state:
                selected_provinces = st.multiselect(
                    "Select Provinces",
                    st.session_state.province_options,
                    key="custom_provinces",
                    help="Choose one or more provinces"
                )
            else:
                selected_provinces = []
                st.info("Click 'Load Provinces' to see available options")
        
        with col2:
            st.markdown("**Regency/City (wadmkk)**")
            if selected_provinces and st.button("🔄 Load Regencies", key="load_regencies"):
                with st.spinner("Loading regency list..."):
                    regency_options = analyzer.get_unique_geographic_values(
                        'wadmkk', 
                        {'wadmpr': selected_provinces}
                    )
                    st.session_state.regency_options = regency_options
            
            if 'regency_options' in st.session_state and selected_provinces:
                selected_regencies = st.multiselect(
                    "Select Regencies/Cities",
                    st.session_state.regency_options,
                    key="custom_regencies",
                    help="Choose regencies within selected provinces"
                )
            else:
                selected_regencies = []
                if not selected_provinces:
                    st.info("Select provinces first")
                else:
                    st.info("Click 'Load Regencies' to see options")
        
        with col3:
            st.markdown("**District (wadmkc)**")
            if selected_regencies and st.button("🔄 Load Districts", key="load_districts"):
                with st.spinner("Loading district list..."):
                    district_options = analyzer.get_unique_geographic_values(
                        'wadmkc',
                        {'wadmkk': selected_regencies}
                    )
                    st.session_state.district_options = district_options
            
            if 'district_options' in st.session_state and selected_regencies:
                selected_districts = st.multiselect(
                    "Select Districts",
                    st.session_state.district_options,
                    key="custom_districts",
                    help="Choose districts within selected regencies"
                )
            else:
                selected_districts = []
                if not selected_regencies:
                    st.info("Select regencies first")
                else:
                    st.info("Click 'Load Districts' to see options")
        
        # Price Range Section
        st.markdown("#### 💰 Price Range (HPM)")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            hpm_min = st.number_input(
                "Minimum HPM",
                min_value=1000,
                max_value=1000000000,
                value=50000,
                step=10000,
                key="custom_hpm_min",
                help="Minimum price per square meter"
            )
        
        with col2:
            hpm_max = st.number_input(
                "Maximum HPM", 
                min_value=1000,
                max_value=1000000000,
                value=200000000,
                step=10000,
                key="custom_hpm_max",
                help="Maximum price per square meter"
            )
        
        with col3:
            result_limit = st.selectbox(
                "Result Limit",
                [10000, 25000, 50000, 100000],
                index=2,
                key="custom_limit",
                help="Maximum number of records to load"
            )
        
        # Validation
        if hpm_min >= hpm_max:
            st.error("❌ Minimum HPM must be less than Maximum HPM")
            custom_filters_valid = False
        else:
            custom_filters_valid = True
        
        # Preview Section
        st.markdown("#### 👀 Filter Preview")
        
        filter_summary = []
        if selected_provinces:
            filter_summary.append(f"**Provinces:** {len(selected_provinces)} selected")
        if selected_regencies:
            filter_summary.append(f"**Regencies:** {len(selected_regencies)} selected")
        if selected_districts:
            filter_summary.append(f"**Districts:** {len(selected_districts)} selected")
        
        filter_summary.append(f"**HPM Range:** {hpm_min:,} - {hpm_max:,}")
        filter_summary.append(f"**Result Limit:** {result_limit:,}")
        
        if filter_summary:
            for item in filter_summary:
                st.write(f"• {item}")
        
        # Execute Custom Filter
        st.markdown("#### 🚀 Execute Custom Filter")
        
        if st.button("🎯 Load Data with Custom Filters", type="primary", disabled=not custom_filters_valid):
            if not selected_provinces and not selected_regencies and not selected_districts:
                st.warning("⚠️ No geographic filters selected. This will load data from entire database. Continue?")
                
                if st.button("✅ Yes, Load All Geographic Areas", key="confirm_all_geo"):
                    execute_custom_filter = True
                else:
                    execute_custom_filter = False
            else:
                execute_custom_filter = True
            
            if execute_custom_filter:
                with st.spinner("🔍 Executing custom database query..."):
                    success, message = analyzer.load_custom_filtered_data(
                        provinces=selected_provinces if selected_provinces else None,
                        regencies=selected_regencies if selected_regencies else None,
                        districts=selected_districts if selected_districts else None,
                        hpm_min=hpm_min,
                        hpm_max=hpm_max,
                        limit=result_limit
                    )
                    
                    if success:
                        st.success(f"✅ {message}")
                        
                        # Clear the cached options to save memory
                        for key in ['province_options', 'regency_options', 'district_options']:
                            if key in st.session_state:
                                del st.session_state[key]
                        
                        st.session_state.processing_step = 'overview'
                        st.rerun()
                    else:
                        st.error(f"❌ {message}")
        
        # Quick Reset
        if st.button("🔄 Reset All Filters", key="reset_custom_filters"):
            for key in ['province_options', 'regency_options', 'district_options', 
                    'custom_provinces', 'custom_regencies', 'custom_districts']:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()

elif st.session_state.processing_step == 'overview':
    if fun_mode:
        st.markdown('## <img src="https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExaXZod3R3NnJ2cW93MjkycXJ3dTRxeHluYXlkemhwdnVyZTFmOWhibyZlcD12MV9naWZzX3RyZW5kaW5nJmN0PWc/0GtVKtagi2GvWuY3vm/giphy.gif" alt="data gif" style="height:96px; vertical-align:middle;"> Data Overview', unsafe_allow_html=True)
    else:
        st.markdown('## Data Overview')
    
    if analyzer.current_data is None:
        st.warning("⚠️ No data loaded. Please go back to Data Selection.")
        if st.button("← Back to Data Selection"):
            st.session_state.processing_step = 'selection'
            st.rerun()
        st.stop()

    else:
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

    if analyzer.current_data is None:
        st.warning("⚠️ No data loaded. Please go back to Data Selection.")
        if st.button("← Back to Data Selection"):
            st.session_state.processing_step = 'selection'
            st.rerun()
        st.stop()

    else:
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
                try:
                    non_null_series = analyzer.current_data[col].dropna()
                    if len(non_null_series) > 0:
                        sample_values.append(str(non_null_series.iloc[0]))
                    else:
                        sample_values.append('N/A')
                except Exception:
                    sample_values.append('Error')
            
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

    if analyzer.current_data is None:
        st.warning("⚠️ No data loaded. Please go back to Data Selection.")
        if st.button("← Back to Data Selection"):
            st.session_state.processing_step = 'selection'
            st.rerun()
        st.stop()

    else:
        # Initialize selected_filters dictionary
        selected_filters = {}
        
        # Geographic Filtering (Priority Section)
        st.markdown("### 🗺️ Smart Geographic Filtering")

        # Clear geographic filters button
        if st.button("🗑️ Clear Geographic Filters"):
            # Clear the multiselect keys to reset selections
            for key in ['filter_geo_wadmpr', 'filter_geo_wadmkk', 'filter_geo_wadmkc']:
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
                
                if wadmpr_col and wadmpr_col in analyzer.current_data.columns:
                    try:
                        wadmpr_options = sorted(analyzer.current_data[wadmpr_col].dropna().unique())
                        selected_wadmpr = st.multiselect(
                            f"Select Province ({wadmpr_col})", 
                            wadmpr_options,
                            key="filter_geo_wadmpr"  # Fixed key naming
                        )
                        if selected_wadmpr:
                            selected_filters[wadmpr_col] = {'type': 'categorical', 'value': selected_wadmpr}
                    except Exception as e:
                        st.error(f"Error loading province options: {e}")

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
                        key="filter_geo_wadmkk"
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
                        key="filter_geo_wadmkc"
                    )
                    if selected_wadmkc:
                        selected_filters[wadmkc_col] = {'type': 'categorical', 'value': selected_wadmkc}

        # Show current geographic selection summary
        geographic_filters = {k: v for k, v in selected_filters.items() 
                            if any(geo in k for geo in ['wadmpr', 'wadmkk', 'wadmkc', 'wadmkd'])}

        if geographic_filters:
            preview_geo_df = analyzer.current_data.copy()
            for col, config in geographic_filters.items():
                if col in preview_geo_df.columns:  # Safety check
                    preview_geo_df = preview_geo_df[preview_geo_df[col].isin(config['value'])]
            
            geo_reduction = len(analyzer.current_data) - len(preview_geo_df)
            
            if len(preview_geo_df) == 0:
                st.error("❌ Geographic filters eliminate all data. Please adjust your selection.")
            elif geo_reduction > 0:
                st.success(f"✅ Geographic filters will reduce data by {geo_reduction:,} records → {len(preview_geo_df):,} remaining")
            else:
                st.info("ℹ️ Geographic filters don't reduce the current dataset")

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

    if analyzer.current_data is None:
        st.warning("⚠️ No data loaded. Please go back to Data Selection.")
        if st.button("← Back to Data Selection"):
            st.session_state.processing_step = 'selection'
            st.rerun()
        st.stop()

    else:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Basic Cleaning:**")
            remove_duplicates = st.checkbox("Remove duplicate records (keep first)", value=False)
            handle_missing = st.checkbox("Handle missing values", value=False)
        
        with col2:
            st.markdown("**Outlier Removal:**")
            remove_outliers = st.checkbox("Remove outliers", value=False)
            outlier_column = None
            group_column = None

            if remove_outliers:
                numeric_cols = analyzer.current_data.select_dtypes(include=[np.number]).columns.tolist()
                outlier_column = st.selectbox("Column for outlier detection", numeric_cols)
                
                # Select group column (categorical or numeric with low cardinality)
                candidate_groups = analyzer.current_data.select_dtypes(include=['object', 'category']).columns.tolist()
                numeric_low_card = [col for col in analyzer.current_data.select_dtypes(include=[np.number]).columns
                                if analyzer.current_data[col].nunique() <= 20]
                candidate_groups.extend(numeric_low_card)
                
                group_column = st.selectbox("Group column (for group-wise outlier detection)", candidate_groups)

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

    if analyzer.current_data is None:
        st.warning("⚠️ No data loaded. Please go back to Data Selection.")
        if st.button("← Back to Data Selection"):
            st.session_state.processing_step = 'selection'
            st.rerun()
        st.stop()

    else:
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
            # Only add if they exist in the dataset
            for col_name in ['hpm', 'luas_tanah']:
                if col_name in numeric_columns:
                    distance_columns.append(col_name)
            
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
                    if st.button("⚡ Transform All Distance + (HPM & Luas Tanah) Columns", type="secondary"):
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
                
                # st.markdown("**Distance Columns Found:**")
                # for col in distance_columns:
                #     st.write(f"- {col}")
                
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

            st.markdown("---")
            
            # Categorical Encoding Section
            st.markdown("### 🏷️ Categorical Column Encoding")
            
            # Specific columns for encoding
            target_columns = ['bentuk_tapak', 'posisi_tapak', 'orientasi', 'kondisi_wilayah_sekitar', 
                            'jenis_jalan_utama', 'perkerasan_jalan', 'jenis_jalan']

            # Filter to only include columns that exist in the dataframe
            all_categorical = [col for col in target_columns if col in analyzer.current_data.columns]

            # Show which columns were found/missing
            missing_columns = [col for col in target_columns if col not in analyzer.current_data.columns]
            if missing_columns:
                st.warning(f"⚠️ These columns were not found in the dataset: {', '.join(missing_columns)}")
            
            if not all_categorical:
                st.info("No categorical columns detected in your dataset.")
            else:
                st.info(f"Found {len(all_categorical)} categorical columns: {', '.join(all_categorical)}")
                
                # Initialize session state for encoding configurations
                if 'encoding_configs' not in st.session_state:
                    st.session_state.encoding_configs = {}
                
                # For each categorical column, show encoding options
                encoding_configs = {}
                
                for i, cat_col in enumerate(all_categorical):
                    with st.expander(f"🏷️ Configure Encoding for: **{cat_col}**", expanded=False):
                        unique_values = sorted(analyzer.current_data[cat_col].dropna().unique())
                        unique_count = len(unique_values)
                        
                        st.write(f"**Column Info:**")
                        st.write(f"- Unique values: {unique_count}")
                        st.write(f"- Sample values: {unique_values[:5]}")
                        if unique_count > 5:
                            st.write(f"- ... and {unique_count - 5} more")
                        
                        # Encoding method selection
                        col1, col2 = st.columns([1, 2])
                        
                        with col1:
                            encoding_method = st.selectbox(
                                "Encoding Method:",
                                ["None", "Ordinal (ordered score)", "One-Hot Encoding"],
                                key=f"encoding_method_{cat_col}_{i}"
                            )
                        
                        with col2:
                            if encoding_method != "None":
                                # Preview of new columns that will be created
                                if encoding_method == "Ordinal (ordered score)":
                                    st.info(f"📊 Will create: **{cat_col}_ordinal** (numeric column)")
                                elif encoding_method == "One-Hot Encoding":
                                    # Show preview of one-hot column names
                                    preview_cols = [f"is_{str(val).lower().replace(' ', '_')}" for val in unique_values[:3]]
                                    if len(unique_values) > 3:
                                        preview_cols.append(f"... and {len(unique_values) - 3} more")
                                    st.info(f"📊 Will create: {', '.join(preview_cols)}")
                        
                        # Custom order for ordinal encoding
                        custom_order = None
                        custom_scores = None
                        if encoding_method == "Ordinal (ordered score)":
                            st.markdown("**🔢 Define Ordinal Mapping:**")
                            
                            col1, col2 = st.columns([1, 1])
                            
                            with col1:
                                use_custom_mapping = st.checkbox(
                                    "Use custom mapping", 
                                    key=f"use_custom_mapping_{cat_col}_{i}"
                                )
                            
                            with col2:
                                if use_custom_mapping:
                                    st.info("💡 Define custom scores for each category")
                            
                            if use_custom_mapping:
                                st.markdown("**📊 Assign Custom Scores:**")
                                custom_scores = {}
                                
                                # Create input fields for each unique value
                                for j, val in enumerate(unique_values):
                                    score = st.number_input(
                                        f"Score for '{val}':",
                                        value=float(j),  # Default to index
                                        step=0.1,
                                        key=f"score_{cat_col}_{i}_{j}"
                                    )
                                    custom_scores[val] = score
                                
                                # Show preview of mapping
                                st.success("✅ Custom mapping defined:")
                                for val, score in custom_scores.items():
                                    st.write(f"• **{val}** → {score}")
                                    
                            else:
                                # Default alphabetical order with sequential numbers
                                custom_order = unique_values
                                st.write("**Default mapping (alphabetical order):**")
                                st.write(" → ".join([f"**{val}**: {j}" for j, val in enumerate(unique_values)]))
                        
                        # Reference category selection for One-Hot Encoding
                        reference_category = None
                        if encoding_method == "One-Hot Encoding":
                            st.markdown("**🎯 Leave-One-Out Configuration:**")
                            st.info("💡 One category will be left out to avoid multicollinearity (reference category)")
                            
                            reference_category = st.selectbox(
                                "Select reference category to leave out:",
                                unique_values,
                                key=f"reference_cat_{cat_col}_{i}",
                                help="This category will NOT get its own column (acts as baseline)"
                            )
                            
                            # Show preview of columns that will be created
                            remaining_categories = [val for val in unique_values if val != reference_category]
                            preview_cols = [f"is_{str(val).lower().replace(' ', '_')}" for val in remaining_categories[:3]]
                            if len(remaining_categories) > 3:
                                preview_cols.append(f"... and {len(remaining_categories) - 3} more")
                            
                            st.success(f"✅ Will create {len(remaining_categories)} columns: {', '.join(preview_cols)}")
                            st.info(f"📋 Reference category: **{reference_category}** (left out)")
                        
                        # Store configuration
                        if encoding_method != "None":
                            encoding_configs[cat_col] = {
                                'method': encoding_method,
                                'custom_scores': custom_scores if encoding_method == "Ordinal (ordered score)" else None,
                                'custom_order': custom_order if encoding_method == "Ordinal (ordered score)" else None,
                                'reference_category': reference_category if encoding_method == "One-Hot Encoding" else None,
                                'unique_values': unique_values
                            }
                
                # Apply all encodings button
                if encoding_configs:
                    st.markdown("### 🚀 Apply Categorical Encodings")
                    
                    # Show summary of what will be done
                    st.markdown("**📋 Encoding Summary:**")
                    for col, config in encoding_configs.items():
                        if config['method'] == "Ordinal (ordered score)":
                            mapping_info = "custom scores" if config['custom_scores'] else "sequential scores (0,1,2...)"
                            st.write(f"• **{col}** → Ordinal encoding ({mapping_info}) → **{col}_ordinal**")
                        elif config['method'] == "One-Hot Encoding":
                            remaining_cats = len(config['unique_values']) - 1  # Minus reference category
                            ref_cat = config['reference_category']
                            st.write(f"• **{col}** → One-Hot encoding → {remaining_cats} new columns (ref: {ref_cat})")
                    
                    if st.button("🎯 Apply All Categorical Encodings", type="primary"):
                        try:
                            with st.spinner("Applying categorical encodings..."):
                                encoded_df = analyzer.current_data.copy()
                                encoding_results = []
                                
                                for col, config in encoding_configs.items():
                                    if config['method'] == "Ordinal (ordered score)":
                                        # Apply ordinal encoding
                                        if config['custom_scores']:
                                            # Use custom scores defined by user
                                            ordinal_mapping = config['custom_scores']
                                        else:
                                            # Use default sequential mapping
                                            order = config['custom_order'] if config['custom_order'] else config['unique_values']
                                            ordinal_mapping = {val: idx for idx, val in enumerate(order)}
                                        
                                        # Apply mapping
                                        new_col_name = f"{col}_ordinal"
                                        encoded_df[new_col_name] = encoded_df[col].map(ordinal_mapping)
                                        
                                        encoding_results.append(f"✅ {col} → {new_col_name} (ordinal)")
                                        
                                        # Track in transformed columns
                                        analyzer.transformed_columns[col] = new_col_name
                                    
                                    elif config['method'] == "One-Hot Encoding":
                                        # Apply one-hot encoding with leave-one-out
                                        reference_cat = config['reference_category']
                                        categories_to_encode = [val for val in config['unique_values'] if val != reference_cat]
                                        
                                        for val in categories_to_encode:
                                            # Create column name (clean the value name)
                                            clean_val = str(val).lower().replace(' ', '_').replace('-', '_')
                                            clean_val = ''.join(c for c in clean_val if c.isalnum() or c == '_')
                                            new_col_name = f"is_{clean_val}"
                                            
                                            # Create binary column
                                            encoded_df[new_col_name] = (encoded_df[col] == val).astype(int)
                                        
                                        encoding_results.append(f"✅ {col} → {len(categories_to_encode)} one-hot columns (ref: {reference_cat})")
                                
                                # Update analyzer data
                                analyzer.current_data = encoded_df
                                st.session_state.data_changed = True
                                
                                # Show results
                                st.success(f"🎉 Successfully applied {len(encoding_configs)} categorical encodings!")
                                
                                for result in encoding_results:
                                    st.success(result)
                                
                                # Show preview of encoded data
                                st.markdown("### 👀 Encoded Data Preview")
                                st.dataframe(analyzer.current_data.head(10), use_container_width=True)
                                
                                st.rerun()
                                
                        except Exception as e:
                            st.error(f"❌ Categorical encoding failed: {str(e)}")
                            st.code(traceback.format_exc())
                
                else:
                    st.info("💡 Configure encoding methods above, then apply all encodings at once.")

elif st.session_state.processing_step == 'model':
    if fun_mode:
        st.markdown('## <img src="https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExd3phdHl3Zm0zZTZoZ2RkdHc5Z3NncHIzaXpjdWI4bmw1YzluMm0ydiZlcD12MV9naWZzX3NlYXJjaCZjdD1n/9ADoZQgs0tyww/giphy.gif" alt="data gif" style="height:72px; vertical-align:middle;"> OLS Regression Analysis', unsafe_allow_html=True)
    else:
        st.markdown('## OLS Regression Analysis')

    if analyzer.current_data is None:
        st.warning("⚠️ No data loaded. Please go back to Data Selection.")
        if st.button("← Back to Data Selection"):
            st.session_state.processing_step = 'selection'
            st.rerun()
        st.stop()

    else:
        st.markdown("**Model Variables:**")
        
        # Get all numeric columns (including transformed ones)
        numeric_columns = analyzer.current_data.select_dtypes(include=[np.number]).columns.tolist()
        
        # Target variable (Y)
        y_column = st.selectbox("Dependent Variable (Y)", numeric_columns, 
                                index=numeric_columns.index('ln_hpm') if 'ln_hpm' in numeric_columns else 0)
        
        # Independent variables (X)
        available_x_cols = [col for col in numeric_columns if col != y_column]
        st.markdown("**Independent Variables (X):**")
        default_checked_cols = [
        'lebar_jalan_di_depan',
        'tahun_pengambilan_data',
        'ln_distance_to_airport',
        'ln_distance_to_bus_stop',
        'ln_distance_to_cafe',
        'ln_distance_to_cemetery',
        'ln_distance_to_convenience_store',
        'ln_distance_to_government',
        'ln_distance_to_hotel',
        'ln_distance_to_mall',
        'ln_distance_to_retail',
        'ln_distance_to_school',
        'ln_distance_to_main_road',
        'ln_distance_to_big_city',
        'ln_distance_to_coastline',
        'ln_distance_to_pharmacy',
        'ln_distance_to_railways',
        'ln_distance_to_sport_center',
        'ln_distance_to_clinic',
        'ln_distance_to_hospital',
        'ln_distance_to_park',
        'ln_hpm',
        'ln_luas_tanah'
        ]

        checkbox_cols_ols = st.columns(3)
        x_columns = []

        for i, col in enumerate(available_x_cols):
            with checkbox_cols_ols[i % 3]:
                default_value = col in default_checked_cols
                if st.checkbox(col, value=default_value, key=f"x_var_{col}"):
                    x_columns.append(col)
                
        # Run OLS Model button
        # Add this after the existing OLS model configuration
        if x_columns:
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
            
            if st.button("💾 Save Variables for ML", type="secondary", help="Save current variable selection for ML models"):
                success, message = analyzer.save_ols_variables(y_column, x_columns)
                if success:
                    st.success(message)
                    st.info(f"Saved: Y={y_column}, X={len(x_columns)} variables")
                    # Show saved variables info
                    st.session_state.show_saved_vars = True
                else:
                    st.error(message)

        # Show saved variables info if available
        if hasattr(st.session_state, 'show_saved_vars') and st.session_state.show_saved_vars:
            saved_vars = analyzer.get_saved_ols_variables()
            if saved_vars:
                with st.expander("💾 Saved Variables", expanded=True):
                    st.write(f"**Target (Y):** {saved_vars['y_column']}")
                    st.write(f"**Features (X):** {', '.join(saved_vars['x_columns'])}")
                    st.write(f"**Saved at:** {saved_vars['timestamp']}")

       
elif st.session_state.processing_step == 'advanced':
    # Initialize advanced_step if not exists FIRST
    if 'advanced_step' not in st.session_state:
        st.session_state.advanced_step = 'ml'
    
    # Show secondary menu at the TOP
    st.markdown("#### 🤖 Advanced Analytics")
    
    # Secondary navigation
    adv_col1, adv_col2 = st.columns(2)
    with adv_col1:
        if st.button("🤖 ML Models", key="adv_ml", 
                    type="primary" if st.session_state.advanced_step == 'ml' else "secondary"):
            st.session_state.advanced_step = 'ml'
            st.rerun()
    
    with adv_col2:
        if st.button("🔗 Hybrid Model", key="adv_hybrid",
                    type="primary" if st.session_state.advanced_step == 'hybrid' else "secondary"):
            st.session_state.advanced_step = 'hybrid'
            st.rerun()
    
    # Add a divider
    st.markdown("---")
    
    # NOW Route to correct advanced model based on advanced_step
    if st.session_state.advanced_step == 'ml':
        if fun_mode:
            st.markdown('## <img src="https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExaWZoeTByeWI1YmdsMHU3dnJ3ejNnem04MmM4Zjh5eThvbG10ZjFiaCZlcD12MV9naWZzX3NlYXJjaCZjdD1n/Gf1RA1jNSpbbuDE40m/giphy.gif" alt="data gif" style="height:96px; vertical-align:middle;"> Machine Learning Models', unsafe_allow_html=True)
        else:
            st.markdown('## Machine Learning Models')

        if analyzer.current_data is not None:

            # Model Selection
            st.markdown("### 🤖 Model Selection")
            model_type = st.selectbox("Choose Algorithm", 
                                    ["Random Forest", "Gradient Boosting (GBDT)"], 
                                    key="advanced_ml_model_type")

            model_name = "RandomForest" if model_type == "Random Forest" else "GradientBoosting"

            # Model Configuration
            st.markdown("### 🎯 Model Configuration")
            
            # Check for saved OLS variables
            saved_vars = analyzer.get_saved_ols_variables()
            if saved_vars:
                st.success(f"💾 **Saved OLS Variables Available**: Y={saved_vars['y_column']}, X={len(saved_vars['x_columns'])} variables")
                
                use_saved = st.checkbox("🔄 Use Saved OLS Variables", value=False, key="use_saved_vars")
            else:
                use_saved = False
                st.info("💡 No saved variables from OLS step. Configure manually or run OLS first.")     

            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Model Variables:**")
                
                # Get all numeric columns (including transformed ones)
                numeric_columns = analyzer.current_data.select_dtypes(include=[np.number]).columns.tolist()
                
                if use_saved and saved_vars:
                    ml_y_column = saved_vars['y_column']
                    ml_x_columns = saved_vars['x_columns']
                else:
                    ml_y_column = st.selectbox("Target Variable (Y)", numeric_columns, key="ml_y_select")
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
                st.markdown("**Manual Parameters**")
                
                if model_type == "Random Forest":
                    # Check if we have best params from Optuna
                    if 'best_params' in st.session_state:
                        st.info("💡 Optuna found these optimal parameters. You can use them or modify below:")
                        st.json(st.session_state.best_params)
                        
                        # Pre-fill with Optuna results
                        default_n_est = str(st.session_state.best_params.get('n_estimators', 100))
                        default_max_depth = str(st.session_state.best_params.get('max_depth', 10))
                        default_min_split = str(st.session_state.best_params.get('min_samples_split', 2))
                        default_min_leaf = str(st.session_state.best_params.get('min_samples_leaf', 1))
                        default_max_feat = st.session_state.best_params.get('max_features', 'sqrt')
                    else:
                        # Default values
                        default_n_est = "100"
                        default_max_depth = "10"
                        default_min_split = "2"
                        default_min_leaf = "1"
                        default_max_feat = 'sqrt'
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        manual_n_estimators = st.text_input("n_estimators", value=default_n_est)
                        manual_max_depth = st.text_input("max_depth", value=default_max_depth)
                    
                    with col2:
                        manual_min_samples_split = st.text_input("min_samples_split", value=default_min_split)
                        manual_min_samples_leaf = st.text_input("min_samples_leaf", value=default_min_leaf)
                    
                    with col3:
                        manual_max_features = st.selectbox("max_features", ['sqrt', 'log2', 'None'], 
                                                        index=['sqrt', 'log2', 'None'].index(default_max_feat) if default_max_feat in ['sqrt', 'log2', 'None'] else 0)
                        manual_random_state = st.text_input("random_state", value=str(random_state))

                elif model_type == "Gradient Boosting (GBDT)":
                    # Check if we have best params from Optuna
                    if 'best_params' in st.session_state:
                        st.info("💡 Optuna found these optimal parameters. You can use them or modify below:")
                        st.json(st.session_state.best_params)
                        
                        # Pre-fill with Optuna results
                        default_n_est = str(st.session_state.best_params.get('n_estimators', 100))
                        default_learning_rate = str(st.session_state.best_params.get('learning_rate', 0.1))
                        default_max_depth = str(st.session_state.best_params.get('max_depth', 6))
                        default_subsample = str(st.session_state.best_params.get('subsample', 1.0))
                        default_min_split = str(st.session_state.best_params.get('min_samples_split', 2))
                        default_min_leaf = str(st.session_state.best_params.get('min_samples_leaf', 1))
                    else:
                        # Default values
                        default_n_est = "100"
                        default_learning_rate = "0.1"
                        default_max_depth = "6"
                        default_subsample = "1.0"
                        default_min_split = "2"
                        default_min_leaf = "1"
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        manual_n_estimators = st.text_input("n_estimators", value=default_n_est)
                        manual_learning_rate = st.text_input("learning_rate", value=default_learning_rate)
                    
                    with col2:
                        manual_max_depth = st.text_input("max_depth", value=default_max_depth)
                        manual_subsample = st.text_input("subsample", value=default_subsample)
                    
                    with col3:
                        manual_random_state = st.text_input("random_state", value=str(random_state))
                        manual_min_samples_split = st.text_input("min_samples_split", value=default_min_split)
                        manual_min_samples_leaf = st.text_input("min_samples_leaf", value=default_min_leaf)

            # Model Training
            st.markdown("### 🚀 Model Training & Evaluation")

            if st.button(f"🤖 Train {model_name} Model", type="primary") and ml_x_columns:
                # Add this INSIDE the if block but BEFORE the try block
                manual_n_estimators = getattr(st.session_state, 'manual_n_estimators', '100')
                manual_max_depth = getattr(st.session_state, 'manual_max_depth', '10')
                manual_min_samples_split = getattr(st.session_state, 'manual_min_samples_split', '2')
                manual_min_samples_leaf = getattr(st.session_state, 'manual_min_samples_leaf', '1')
                manual_max_features = getattr(st.session_state, 'manual_max_features', 'sqrt')
                manual_random_state = getattr(st.session_state, 'manual_random_state', str(random_state))
                
                # GBDT specific variables if needed
                if model_type == "Gradient Boosting (GBDT)":
                    manual_learning_rate = getattr(st.session_state, 'manual_learning_rate', '0.1')
                    manual_subsample = getattr(st.session_state, 'manual_subsample', '1.0')
                
                try:
                    # Parse and validate parameters
                    n_est = int(manual_n_estimators)
                    if n_est < 1 or n_est > 1000:
                        raise ValueError("n_estimators must be between 1 and 1000")
                    
                    max_d = int(manual_max_depth) if manual_max_depth not in ['None', ''] else None
                    if max_d is not None and (max_d < 1 or max_d > 50):
                        raise ValueError("max_depth must be between 1 and 50")
                    
                    min_split = int(manual_min_samples_split)
                    if min_split < 2:
                        raise ValueError("min_samples_split must be >= 2")
                    
                    min_leaf = int(manual_min_samples_leaf)
                    if min_leaf < 1:
                        raise ValueError("min_samples_leaf must be >= 1")
                    
                    max_feat = manual_max_features if manual_max_features != 'None' else None
                    rand_state = int(manual_random_state)
                    
                    # Add GBDT parameters only when needed
                    if model_type == "Gradient Boosting (GBDT)":
                        learning_rate = float(manual_learning_rate)
                        subsample = float(manual_subsample)
                    
                    # Create model based on model_type
                    if model_type == "Random Forest":
                        model = RandomForestRegressor(
                            n_estimators=n_est,
                            max_depth=max_d,
                            min_samples_split=min_split,
                            min_samples_leaf=min_leaf,
                            max_features=max_feat,
                            random_state=rand_state
                        )
                    elif model_type == "Gradient Boosting (GBDT)":
                        model = GradientBoostingRegressor(
                            n_estimators=n_est,
                            learning_rate=learning_rate,
                            max_depth=max_d,
                            subsample=subsample,
                            min_samples_split=min_split,
                            min_samples_leaf=min_leaf,
                            random_state=rand_state
                        )
                    
                    with st.spinner(f"Training {model_name} model..."):
                        # Run model training and evaluation
                        final_model, evaluation_df, train_results_df, global_train_metrics, global_test_metrics, y_test_last, y_pred_last, is_log_transformed = analyzer.goval_machine_learning(
                            ml_x_columns, ml_y_column, model,
                            group_column if use_group else None,
                            n_splits, rand_state, min_sample
                        )
                        
                        # Store all necessary variables in session state
                        st.session_state.ml_model = final_model
                        st.session_state.ml_model_name = model_name
                        st.session_state.ml_feature_names = ml_x_columns
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
                        
                        # Model Export with ONNX
                        st.markdown("### 💾 Model Export")

                        # Check if model is available for export
                        if 'ml_model' in st.session_state and st.session_state.ml_model is not None:
    
                            # Show current model info
                            st.info(f"🤖 **{st.session_state.get('ml_model_name', 'Model')}** ready for export")
                            
                            # Prepare exports button
                            if st.button("🔄 Prepare Model for Download", type="secondary", use_container_width=True, key="prep_model_unique"):
                                try:
                                    with st.spinner("Converting model to ONNX..."):
                                        # Get model data from session state
                                        model = st.session_state.ml_model
                                        feature_names = st.session_state.get('ml_feature_names', [])
                                        model_name = st.session_state.get('ml_model_name', 'MLModel')
                                        
                                        if not feature_names:
                                            st.error("❌ Feature names not found. Please retrain the model.")
                                        else:
                                            # Convert to ONNX
                                            onnx_bytes, feature_json, success = analyzer.prepare_model_exports(
                                                model, feature_names, model_name
                                            )
                                            
                                            if success:
                                                # Store in session state with unique keys
                                                st.session_state['ml_onnx_ready'] = True
                                                st.session_state['ml_onnx_data'] = onnx_bytes
                                                st.session_state['ml_feature_json'] = feature_json
                                                st.session_state['ml_model_name_export'] = model_name
                                                st.session_state['ml_timestamp'] = datetime.now().strftime('%Y%m%d_%H%M')
                                                
                                                st.success("✅ Model converted to ONNX successfully!")
                                                st.balloons()  # Visual feedback
                                                # ❌ REMOVED: st.rerun() - this was causing the reload issue
                                            else:
                                                st.error(f"❌ ONNX conversion failed: {feature_json}")
                                                
                                except Exception as e:
                                    st.error(f"❌ Model export preparation failed: {str(e)}")
                                    st.code(traceback.format_exc())

                            # Download buttons (only show if data is prepared and ready)
                            if st.session_state.get('ml_onnx_ready', False) and st.session_state.get('ml_onnx_data') is not None:
                                
                                st.success("🎉 **Model is ready for download!**")
                                
                                # Model info display
                                with st.expander("📋 Export Information", expanded=True):
                                    st.write(f"**Model Type:** {st.session_state.get('ml_model_name_export', 'Unknown')}")
                                    st.write(f"**Features:** {len(st.session_state.get('ml_feature_names', []))}")
                                    st.write(f"**Export Time:** {st.session_state.get('ml_timestamp', 'Unknown')}")
                                    st.write(f"**Format:** ONNX + JSON")
                                
                                st.markdown("**📥 Download Files:**")
                                
                                col1, col2, col3 = st.columns(3)
                                
                                with col1:
                                    # Download ONNX model
                                    try:
                                        model_name_export = st.session_state.get('ml_model_name_export', 'model')
                                        timestamp = st.session_state.get('ml_timestamp', datetime.now().strftime('%Y%m%d_%H%M'))
                                        
                                        st.download_button(
                                            label=f"📦 {model_name_export}.onnx",
                                            data=st.session_state['ml_onnx_data'],
                                            file_name=f"{model_name_export.lower()}_model_{timestamp}.onnx",
                                            mime="application/octet-stream",
                                            use_container_width=True,
                                            key="download_onnx_unique"
                                        )
                                    except Exception as e:
                                        st.error(f"ONNX download error: {e}")
                                
                                with col2:
                                    # Download feature info
                                    try:
                                        st.download_button(
                                            label="📄 Features.json",
                                            data=st.session_state['ml_feature_json'],
                                            file_name=f"{model_name_export.lower()}_features_{timestamp}.json",
                                            mime="application/json",
                                            use_container_width=True,
                                            key="download_json_unique"
                                        )
                                    except Exception as e:
                                        st.error(f"JSON download error: {e}")
                                
                                with col3:
                                    # Clear prepared data (with confirmation)
                                    if st.button("🗑️ Clear Export Data", use_container_width=True, key="clear_export_unique"):
                                        # Clear all export-related session state
                                        export_keys = ['ml_onnx_ready', 'ml_onnx_data', 'ml_feature_json', 
                                                    'ml_model_name_export', 'ml_timestamp']
                                        for key in export_keys:
                                            if key in st.session_state:
                                                del st.session_state[key]
                                        st.success("🗑️ Export data cleared!")
                                        # ❌ REMOVED: st.rerun() - let user manually refresh if needed
                            
                            elif st.session_state.get('ml_onnx_ready', False):
                                st.warning("⚠️ Model preparation incomplete. Please try 'Prepare Model for Download' again.")
                                # Clear incomplete state
                                if st.button("🔄 Reset Preparation", key="reset_prep_unique"):
                                    st.session_state['ml_onnx_ready'] = False
                                    # ❌ REMOVED: st.rerun() - let user see the result
                            
                            else:
                                st.info("👆 Click 'Prepare Model for Download' to convert your model to ONNX format")

                        else:
                            st.warning("⚠️ No trained model available for export. Please train a model first.")

                        # Other download options (NO AUTO RELOAD)
                        st.markdown("### 📊 Additional Downloads")

                        col1, col2 = st.columns(2)

                        with col1:
                            # Download evaluation results
                            if 'ml_evaluation_df' in st.session_state:
                                eval_csv = st.session_state.ml_evaluation_df.to_csv(index=False)
                                st.download_button(
                                    label="📊 Download Results (.csv)",
                                    data=eval_csv,
                                    file_name=f"ml_evaluation_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                                    mime="text/csv",
                                    use_container_width=True,
                                    key="download_results_csv"
                                )
                            else:
                                st.info("No evaluation results available")

                        with col2:
                            # Download model info
                            if 'ml_model' in st.session_state:
                                model_info = {
                                    'timestamp': datetime.now().isoformat(),
                                    'algorithm': st.session_state.get('ml_model_name', 'Unknown'),
                                    'target_variable': st.session_state.get('ml_y_column', 'Unknown'),
                                    'features': st.session_state.get('ml_feature_names', []),
                                    'model_format': 'ONNX',
                                    'usage_example': {
                                        'python': f'''
                        import onnxruntime as ort
                        import numpy as np
                        import json

                        # Load model and features
                        session = ort.InferenceSession('model.onnx')
                        with open('features.json', 'r') as f:
                            feature_info = json.load(f)

                        # Prepare input (ensure correct feature order)
                        input_data = your_data[feature_info['feature_names']].values.astype(np.float32)

                        # Make predictions
                        predictions = session.run(None, {{'float_input': input_data}})[0]
                                        '''
                                    },
                                    'global_test_metrics': st.session_state.get('ml_global_test', {}),
                                    'global_train_metrics': st.session_state.get('ml_global_train', {})
                                }
                                
                                st.download_button(
                                    label="📋 Download Model Info (.json)",
                                    data=json.dumps(model_info, indent=2),
                                    file_name=f"ml_model_info_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
                                    mime="application/json",
                                    use_container_width=True,
                                    key="download_model_info"
                                )
                            else:
                                st.info("No model info available")

                        # Show current model status (without auto-reload)
                        if 'ml_model' in st.session_state:
                            st.markdown("---")
                            st.success(f"🤖 {st.session_state.get('ml_model_name', 'Model')} ready!")
                            
                            # Quick model info
                            with st.expander("📋 Current Model Info"):
                                st.write(f"**Algorithm:** {st.session_state.get('ml_model_name', 'Unknown')}")
                                if 'ml_global_test' in st.session_state:
                                    metrics = st.session_state.ml_global_test
                                    st.write(f"**Test R²:** {metrics.get('R2', 'N/A'):.4f}")
                                    st.write(f"**Test PE10:** {metrics.get('PE10', 'N/A'):.4f}")
                                    st.write(f"**Test RT20:** {metrics.get('RT20', 'N/A'):.4f}")
                                    st.write(f"**Test FSD:** {metrics.get('FSD', 'N/A'):.4f}")

                        # Manual refresh option (if user wants to refresh)
                        st.markdown("---")
                        if st.button("🔄 Refresh Page", help="Manual refresh if needed"):
                            st.rerun()

                except ValueError as e:
                    st.error(f"❌ Parameter validation error: {str(e)}")
                except Exception as e:
                    st.error(f"❌ Model training failed: {str(e)}")
                    st.code(traceback.format_exc())

    elif st.session_state.advanced_step == 'hybrid':
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

            # Check for saved OLS variables
            saved_vars = analyzer.get_saved_ols_variables()
            if saved_vars:
                st.success(f"💾 **Saved OLS Variables Available**: Y={saved_vars['y_column']}, X={len(saved_vars['x_columns'])} variables")
                use_saved = st.checkbox("🔄 Use Saved OLS Variables", value=False, key="use_saved_vars")
            else:
                use_saved = False
                st.info("💡 No saved variables from OLS step. Configure manually or run OLS first.")

            # Get all numeric columns (including transformed ones)
            numeric_columns = analyzer.current_data.select_dtypes(include=[np.number]).columns.tolist()

            # Initialize variables
            hybrid_x_columns = []
            ml_y_column = None
            ml_x_columns = []

            if use_saved and saved_vars:
                # Use saved variables directly
                ml_y_column = saved_vars['y_column']
                ml_x_columns = saved_vars['x_columns']

                st.markdown(f"**Using saved target variable (Y):** `{ml_y_column}`")
                st.markdown(f"**Using saved independent variables (X):** {ml_x_columns}")

            else:
                col1, col2 = st.columns(2)

                with col1:
                    hybrid_y_column = st.selectbox(
                        "Target Variable (Y)",
                        numeric_columns,
                        index=numeric_columns.index('ln_hpm') if 'ln_hpm' in numeric_columns else 0,
                        key="hybrid_y_select",
                        disabled=use_saved  # Disable if using saved variables
                    )

                with col2:
                    hybrid_n_splits = st.number_input(
                        "CV Folds",
                        min_value=3,
                        max_value=20,
                        value=5,
                        key="hybrid_cv_folds",
                        disabled=use_saved  # Disable if using saved variables
                    )

                st.markdown("**Independent Variables (X):**")
                available_x_cols = [col for col in numeric_columns if col != hybrid_y_column]

                checkbox_cols = st.columns(3)
                for i, col in enumerate(available_x_cols):
                    with checkbox_cols[i % 3]:
                        checked = (i < 5)
                        if use_saved:
                            st.checkbox(col, value=False, key=f"hybrid_x_{col}", disabled=True)
                        else:
                            if st.checkbox(col, value=checked, key=f"hybrid_x_{col}"):
                                hybrid_x_columns.append(col)

                ml_y_column = hybrid_y_column
                ml_x_columns = hybrid_x_columns
                
            # Random Forest Parameters section
            st.markdown("### ⚙️ Random Forest Parameters")
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

                        # Store final models for export
                        final_ols_model = None
                        final_rf_model = None
                        
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
                            
                            # Store models from last fold for export
                            if fold == hybrid_n_splits - 1:
                                final_ols_model = ols_model
                                final_rf_model = rf_model
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
                        st.session_state.final_ols_model = final_ols_model
                        st.session_state.final_rf_model = final_rf_model
                        st.session_state.hybrid_feature_names = hybrid_x_columns
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
                    
                    # Hybrid Model Export
                    st.markdown("### 💾 Save Trained Models")

                    # Prepare hybrid models button
                    if st.button("🔄 Prepare Hybrid Models for Download", type="secondary", use_container_width=True):
                        with st.spinner("Preparing hybrid models..."):
                            # Prepare RF model as ONNX
                            rf_onnx_bytes, rf_feature_json, rf_success = analyzer.prepare_model_exports(
                                final_rf_model, hybrid_x_columns, "HybridRF_Residuals"
                            )
                            
                            if rf_success:
                                # Prepare OLS model as pickle (since ONNX doesn't support OLS directly)
                                ols_model_bytes = pickle.dumps(final_ols_model)
                                
                                # Create hybrid feature info
                                hybrid_feature_info = {
                                    'model_type': 'Hybrid_OLS_RF',
                                    'feature_names': hybrid_x_columns,
                                    'target_name': hybrid_y_column,
                                    'instructions': {
                                        'step1': 'Load OLS model (pickle) and RF model (ONNX)',
                                        'step2': 'Get OLS prediction first',
                                        'step3': 'Get RF residual prediction',
                                        'step4': 'Final prediction = OLS + RF residual'
                                    },
                                    'timestamp': datetime.now().isoformat(),
                                    'usage_example': '''
                    import pickle
                    import onnxruntime as ort
                    import statsmodels.api as sm
                    import numpy as np

                    # Load models
                    with open('hybrid_ols_model.pkl', 'rb') as f:
                        ols_model = pickle.load(f)
                    rf_session = ort.InferenceSession('hybrid_rf_model.onnx')

                    # Make predictions
                    def predict_hybrid(X_new):
                        # Step 1: OLS prediction
                        X_ols = sm.add_constant(X_new)
                        ols_pred = ols_model.predict(X_ols)
                        
                        # Step 2: RF residual prediction  
                        rf_pred = rf_session.run(None, {'float_input': X_new.astype(np.float32)})[0]
                        
                        # Step 3: Combine
                        return ols_pred + rf_pred.flatten()
                                    '''
                                }
                                
                                # Store in session state
                                st.session_state.hybrid_rf_onnx = rf_onnx_bytes
                                st.session_state.hybrid_ols_pkl = ols_model_bytes
                                st.session_state.hybrid_feature_json = json.dumps(hybrid_feature_info, indent=2)
                                st.session_state.hybrid_timestamp = datetime.now().strftime('%Y%m%d_%H%M')
                                
                                st.success("✅ Hybrid models prepared! Download buttons are now available below.")
                            else:
                                st.error(f"❌ {rf_feature_json}")

                    # Download buttons for hybrid models
                    if 'hybrid_rf_onnx' in st.session_state:
                        st.markdown("**📥 Download Hybrid Model Files:**")
                        
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            # Download RF ONNX model
                            st.download_button(
                                label="📦 Hybrid_RF.onnx",
                                data=st.session_state.hybrid_rf_onnx,
                                file_name=f"hybrid_rf_model_{st.session_state.hybrid_timestamp}.onnx",
                                mime="application/octet-stream",
                                use_container_width=True
                            )
                        
                        with col2:
                            # Download OLS pickle model
                            st.download_button(
                                label="📦 Hybrid_OLS.pkl", 
                                data=st.session_state.hybrid_ols_pkl,
                                file_name=f"hybrid_ols_model_{st.session_state.hybrid_timestamp}.pkl",
                                mime="application/octet-stream",
                                use_container_width=True
                            )
                        
                        with col3:
                            # Download feature info
                            st.download_button(
                                label="📄 Hybrid_Features.json",
                                data=st.session_state.hybrid_feature_json,
                                file_name=f"hybrid_features_{st.session_state.hybrid_timestamp}.json",
                                mime="application/json",
                                use_container_width=True
                            )
                        
                        # Clear hybrid data button
                        if st.button("🗑️ Clear Prepared Hybrid Data", use_container_width=True):
                            for key in ['hybrid_rf_onnx', 'hybrid_ols_pkl', 'hybrid_feature_json', 'hybrid_timestamp']:
                                if key in st.session_state:
                                    del st.session_state[key]
                            st.rerun()

                    # Other hybrid downloads
                    st.markdown("**📊 Additional Hybrid Downloads:**")
                    
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
        
        # # Show transformations if any
        # if analyzer.transformed_columns:
        #     st.markdown("**🔄 Active Transformations:**")
        #     for original, transformed in analyzer.transformed_columns.items():
        #         st.write(f"- {original} → {transformed}")
        
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
