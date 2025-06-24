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
from sklearn.linear_model import LinearRegression
from datetime import timezone, timedelta
import pygwalker as pyg
from pygwalker.api.streamlit import StreamlitRenderer

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
        'show_saved_vars': False,
        'categorical_encodings': {}  
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
        if total_count > 100000:
            st.warning(f"⚠️ Large dataset detected ({total_count:,} records). Loading first 50,000 for performance.")
            limit = 100000
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
        
        # # Remove outliers (optimized)
        # if cleaning_options.get('remove_outliers', False) and cleaning_options.get('outlier_column'):
        #     group_col = cleaning_options['outlier_column']
        #     # flag outliers in the cleaned_df
        #     flagged = self.detect_outliers(
        #         cleaned_df,
        #         id_col='id',
        #         group_column=group_col,
        #         value_column='hpm',
        #         p10_quantile=cleaning_options.get('p10_quantile', 0.1),
        #         p90_quantile=cleaning_options.get('p90_quantile', 0.9),
        #         skew_threshold=cleaning_options.get('skew_threshold', 0.5)
        #     )
        #     # drop the outliers completely
        #     cleaned_df = (
        #         flagged
        #         .loc[flagged['is_outlier'] == 0]
        #         .drop(columns=['upper_border','lower_border','is_outlier'])
        #         .copy()
        #     )

            
            # if success:
            #     cleaned_df = analyzer.current_data
            #     print(message)
            # else:
            #     print(f"Outlier removal failed: {message}")
            
            # # Restore original data reference
            # analyzer.current_data = temp_data
        
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

def get_pyg_renderer(df: pd.DataFrame, spec_path: str) -> "StreamlitRenderer":
    return StreamlitRenderer(df, spec=spec_path, spec_io_mode="rw")

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
            # Reset categorical encodings for new data
            if 'st' in globals() and hasattr(st, 'session_state'):
                st.session_state.categorical_encodings = {}
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
            # Reset categorical encodings for new data
            if 'st' in globals() and hasattr(st, 'session_state'):
                st.session_state.categorical_encodings = {}
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
        """
        Applies basic cleaning (dedupe, missing‐value fill) and then, optionally,
        group‐wise outlier removal via detect_outliers.
        """
        # 1) Ensure there's data to work on
        if self.current_data is None:
            return False, "No data to clean"

        try:
            # 2) Delegate dedupe & missing‐value handling to the cached helper
            cleaned_df = cached_clean_data(self.current_data, cleaning_options)

            # 3) Optionally remove group‐wise outliers
            if cleaning_options.get('remove_outliers', False) and cleaning_options.get('outlier_column'):
                group_col = cleaning_options['outlier_column']

                # **FIX 1: Check for required columns**
                required_cols = ['hpm', 'id']
                missing_cols = [col for col in required_cols if col not in cleaned_df.columns]
                
                if missing_cols:
                    return False, f"Cannot remove outliers: Missing required columns: {missing_cols}"
                
                # **FIX 2: Ensure group column exists**
                if group_col not in cleaned_df.columns:
                    return False, f"Cannot remove outliers: Group column '{group_col}' not found"

                # **FIX 3: Check if we have enough data**
                if len(cleaned_df) < 10:
                    return False, "Cannot remove outliers: Not enough data (minimum 10 rows required)"

                try:
                    # Flag outliers
                    flagged = self.detect_outliers(
                        cleaned_df.copy(),
                        id_col='id',
                        group_column=group_col,
                        value_column='hpm',
                        p10_quantile=cleaning_options.get('p10_quantile', 0.1),
                        p90_quantile=cleaning_options.get('p90_quantile', 0.9),
                        skew_threshold=cleaning_options.get('skew_threshold', 0.5)
                    )

                    # **FIX 4: Check if flagged DataFrame has the expected columns**
                    if 'is_outlier' not in flagged.columns:
                        return False, "Outlier detection failed: 'is_outlier' column not created"

                    # Count outliers before removal
                    outliers_count = flagged['is_outlier'].sum()
                    total_count = len(flagged)
                    
                    # **FIX 5: Safety check - don't remove too many records**
                    if outliers_count / total_count > 0.5:  # If more than 50% would be removed
                        return False, f"Outlier removal would remove {outliers_count}/{total_count} records ({outliers_count/total_count*100:.1f}%). This seems excessive. Please check your parameters."

                    # Drop outliers and helper columns
                    outlier_cols_to_drop = ['upper_border', 'lower_border', 'is_outlier']
                    existing_cols_to_drop = [col for col in outlier_cols_to_drop if col in flagged.columns]
                    
                    cleaned_df = (
                        flagged
                        .loc[flagged['is_outlier'] == 0]  # Keep non-outliers
                        .drop(columns=existing_cols_to_drop)  # Only drop columns that exist
                        .copy()
                    )
                    
                    # **FIX 6: Reset index to avoid issues**
                    cleaned_df = cleaned_df.reset_index(drop=True)
                    
                    print(f"Outlier removal: Removed {outliers_count} outliers from {total_count} records")
                    
                except Exception as outlier_error:
                    return False, f"Outlier removal failed: {str(outlier_error)}"

            # 4) Commit the cleaned DataFrame back to the analyzer
            self.current_data = cleaned_df
            if 'st' in globals():
                st.session_state.data_changed = True

            return True, f"Cleaned data: {len(self.current_data):,} properties remaining"

        except Exception as e:
            # Return False so Streamlit's st.error is used
            return False, f"Data cleaning failed: {str(e)}"

    
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
                              hpm_min=50000, hpm_max=200000000,
                              luas_tanah_min=0, luas_tanah_max=100000,
                              lebar_jalan_min=0, lebar_jalan_max=50,
                              limit=50000):
        """Load data with custom database-level filtering"""
        try:
            query_parts = ["SELECT * FROM engineered_property_data WHERE 1=1"]
            
            # HPM filter
            query_parts.append(f"AND hpm BETWEEN {hpm_min} AND {hpm_max}")
            # Luas Tanah filter
            query_parts.append(f"AND luas_tanah BETWEEN {luas_tanah_min} AND {luas_tanah_max}")
            # Lebar Jalan filter
            query_parts.append(f"AND lebar_jalan_di_depan BETWEEN {lebar_jalan_min} AND {lebar_jalan_max}")
 
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
            # Reset categorical encodings for new data
            if 'st' in globals() and hasattr(st, 'session_state'):
                st.session_state.categorical_encodings = {}
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
    
    def detect_outliers(self,
                df: pd.DataFrame,
                id_col: str,
                group_column: str,
                value_column: str,
                p10_quantile: float = 0.1,
                p90_quantile: float = 0.9,
                skew_threshold: float = 0.5) -> pd.DataFrame:
        """
        Detects outliers in value data for each group in a given DataFrame.

        Parameters:
        -----------
        df : pd.DataFrame
            The DataFrame containing the data.
        id_col : str
            The name of the column representing the object ID.
        group_column : str
            The name of the column to group by (e.g., district or category).
        value_column : str
            The name of the column containing the values to analyze (e.g., price).
        p10_quantile : float, optional
            The lower quantile threshold for trimming (default is 0.1).
        p90_quantile : float, optional
            The upper quantile threshold for trimming (default is 0.9).
        skew_threshold : float, optional
            The threshold for skewness to choose between mean and median (default is 0.5).

        Returns:
        --------
        pd.DataFrame
            The modified DataFrame with 'upper_border', 'lower_border', and 'is_outlier' columns added.
        """
        import numpy as np
        import math
        from scipy.stats import skew
        
        df = df.copy()
        
        # **FIX 1: Handle missing values upfront**
        # Remove rows where key columns have NaN values
        required_cols = [id_col, group_column, value_column]
        original_count = len(df)
        
        # Check for missing values in required columns
        missing_mask = df[required_cols].isnull().any(axis=1)
        missing_count = missing_mask.sum()
        
        if missing_count > 0:
            print(f"Warning: Removing {missing_count} rows with missing values in required columns")
            df = df[~missing_mask].copy()
        
        if len(df) == 0:
            print("Error: No valid data remaining after removing missing values")
            # Return original DataFrame with outlier columns filled with 0
            original_df = df.copy() if len(df) > 0 else pd.DataFrame()
            if len(original_df) == 0:
                return pd.DataFrame()
            original_df['upper_border'] = np.nan
            original_df['lower_border'] = np.nan
            original_df['is_outlier'] = 0
            return original_df
        
        # **FIX 2: Ensure numeric types**
        try:
            df[value_column] = pd.to_numeric(df[value_column], errors='coerce')
            # Remove any rows that couldn't be converted to numeric
            numeric_mask = df[value_column].notna()
            df = df[numeric_mask].copy()
            
            if len(df) == 0:
                print("Error: No valid numeric data in value column")
                return pd.DataFrame()
                
        except Exception as e:
            print(f"Error converting {value_column} to numeric: {e}")
            return df
        
        top_border = {'group': [], 'upper_value': []}
        lower_border = {'group': [], 'lower_value': []}
        outlier_ids = []

        # Loop through each unique group
        unique_groups = df[group_column].dropna().unique()
        
        for group in unique_groups:
            try:
                temp_df = df[df[group_column] == group].copy()
                values = temp_df[value_column].dropna()

                if len(values) < 5:  # **FIX 3: Minimum sample size check**
                    print(f"Warning: Group '{group}' has only {len(values)} samples, skipping outlier detection")
                    # Mark all as non-outliers for this group
                    top_border['group'].append(group)
                    top_border['upper_value'].append(np.inf)
                    lower_border['group'].append(group)
                    lower_border['lower_value'].append(-np.inf)
                    continue

                # **FIX 4: Handle edge cases in quantile calculation**
                try:
                    p10 = np.quantile(values, p10_quantile)
                    p90 = np.quantile(values, p90_quantile)
                    
                    # Handle case where p10 == p90 (all values the same)
                    if p10 == p90:
                        print(f"Warning: Group '{group}' has no variance, skipping outlier detection")
                        top_border['group'].append(group)
                        top_border['upper_value'].append(np.inf)
                        lower_border['group'].append(group)
                        lower_border['lower_value'].append(-np.inf)
                        continue
                        
                except Exception as e:
                    print(f"Error calculating quantiles for group '{group}': {e}")
                    continue

                # Trim to between p10 and p90
                trimmed = values[(values > p10) & (values < p90)]
                if len(trimmed) == 0:
                    print(f"Warning: No data between quantiles for group '{group}'")
                    top_border['group'].append(group)
                    top_border['upper_value'].append(np.inf)
                    lower_border['group'].append(group)
                    lower_border['lower_value'].append(-np.inf)
                    continue

                std = trimmed.std()
                if std == 0 or np.isnan(std):
                    print(f"Warning: No variance in trimmed data for group '{group}'")
                    top_border['group'].append(group)
                    top_border['upper_value'].append(np.inf)
                    lower_border['group'].append(group)
                    lower_border['lower_value'].append(-np.inf)
                    continue

                mean = trimmed.mean()
                med = trimmed.median()
                
                # **FIX 5: Handle skewness calculation errors**
                try:
                    skewness = skew(trimmed, bias=True)
                    if np.isnan(skewness):
                        skewness = 0
                except Exception:
                    skewness = 0

                # choose center
                center = med if abs(skewness) > skew_threshold else mean

                # **FIX 6: Safe ceiling operation**
                try:
                    lower_n = max(1, math.ceil((center - p10) / std))  # Ensure at least 1
                    upper_n = max(1, math.ceil((p90 - center) / std))  # Ensure at least 1
                except (ValueError, OverflowError):
                    print(f"Warning: Error in ceiling calculation for group '{group}', using default values")
                    lower_n = 2
                    upper_n = 2

                lower_val = center - lower_n * std
                upper_val = center + upper_n * std

                top_border['group'].append(group)
                top_border['upper_value'].append(upper_val)
                lower_border['group'].append(group)
                lower_border['lower_value'].append(lower_val)

                # collect outlier ids
                group_outliers = temp_df[
                    (temp_df[value_column] < lower_val) |
                    (temp_df[value_column] > upper_val)
                ][id_col].tolist()
                outlier_ids.extend(group_outliers)

            except Exception as e:
                print(f"Error processing group '{group}': {e}")
                continue

        # **FIX 7: Handle empty results**
        if not top_border['group']:
            print("Warning: No groups could be processed for outlier detection")
            df['upper_border'] = np.inf
            df['lower_border'] = -np.inf
            df['is_outlier'] = 0
            return df

        # build lookup dicts
        up_map = dict(zip(top_border['group'], top_border['upper_value']))
        low_map = dict(zip(lower_border['group'], lower_border['lower_value']))

        # annotate dataframe
        df['upper_border'] = df[group_column].map(up_map)
        df['lower_border'] = df[group_column].map(low_map)
        
        # **FIX 8: Safe outlier marking**
        df['is_outlier'] = 0  # Initialize all as non-outliers
        if outlier_ids:
            outlier_mask = df[id_col].isin(outlier_ids)
            df.loc[outlier_mask, 'is_outlier'] = 1

        # **FIX 9: Fill NaN values in border columns**
        df['upper_border'] = df['upper_border'].fillna(np.inf)
        df['lower_border'] = df['lower_border'].fillna(-np.inf)

        # summary
        outlier_count = df['is_outlier'].sum()
        total_count = len(df)
        ratio = outlier_count / total_count if total_count > 0 else 0
        print(f"Outliers flagged: {outlier_count} out of {total_count} (ratio: {ratio:.2%})")

        return df


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
    
    def goval_machine_learning(self, X, y, algorithm, group=None, n_splits=10, random_state=101, min_sample=3, full_dataset_eval=False):
        """
        Enhanced version that stores raw predictions per fold for dynamic metric recalculation
        """
        data = self.current_data

        # Determine if target is log-transformed ONCE at the beginning
        is_log_transformed = ('ln_' in y) or ('log_' in y.lower())
        
        # Prepare data
        model_vars = [y] + X
        df_model = data[model_vars].dropna()
        X_full = df_model[X]
        y_full = df_model[y]

        # Initialize results containers
        evaluation_results = {'Fold': [], 'R2': [], 'FSD': [], 'PE10': [], 'RT20': []}
        train_results = {'R2': [], 'FSD': [], 'PE10': [], 'RT20': []}
        
        # NEW: Store raw predictions per fold for dynamic recalculation
        fold_raw_predictions = []

        global_train_metrics = {'R2': 0, 'FSD': 0, 'PE10': 0, 'RT20': 0}
        global_test_metrics = {'R2': 0, 'FSD': 0, 'PE10': 0, 'RT20': 0}

        # Store last fold test data for comparison graph
        last_test_idx = None
        last_group_test_data = None

        # === CROSS-VALIDATION EVALUATION ===
        if not full_dataset_eval:
            # --- Standard KFold Cross-Validation ---
            if group is None:
                from sklearn.model_selection import KFold
                kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
                
                for fold, (train_idx, test_idx) in enumerate(kf.split(X_full)):
                    X_train, X_test = X_full.iloc[train_idx], X_full.iloc[test_idx]
                    y_train, y_test = y_full.iloc[train_idx], y_full.iloc[test_idx]
                    
                    # Train model
                    fold_model = algorithm.__class__(**algorithm.get_params())
                    fold_model = fold_model.fit(X_train, y_train)
                    y_train_pred = fold_model.predict(X_train)
                    y_test_pred = fold_model.predict(X_test)
                    
                    # NEW: Store raw predictions for this fold
                    fold_raw_predictions.append({
                        'fold': fold + 1,
                        'y_actual': y_test.values.tolist(),
                        'y_predicted': y_test_pred.tolist(),
                        'fold_name': f"Fold-{fold + 1}"
                    })
                    
                    # Calculate metrics with current scale
                    train_metrics = evaluate(y_train, y_train_pred, squared=is_log_transformed)
                    test_metrics = evaluate(y_test, y_test_pred, squared=is_log_transformed)
                    
                    # Store metrics
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

                    # Store last fold test indices for comparison graph
                    if fold == n_splits - 1:
                        last_test_idx = test_idx

            # --- Group-Based Cross-Validation ---
            else:
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

                    fold_model = algorithm.__class__(**algorithm.get_params())
                    fold_model = fold_model.fit(X_train_all, y_train_all)
                    y_train_pred = fold_model.predict(X_train_all)
                    y_test_pred = fold_model.predict(X_test_all)
                    
                    # NEW: Store raw predictions for this fold
                    fold_raw_predictions.append({
                        'fold': fold + 1,
                        'y_actual': y_test_all.values.tolist(),
                        'y_predicted': y_test_pred.tolist(),
                        'fold_name': f"Fold-{fold + 1}"
                    })

                    # Evaluate train and test scores
                    train_metrics = evaluate(y_train_all, y_train_pred, squared=is_log_transformed)
                    test_metrics = evaluate(y_test_all, y_test_pred, squared=is_log_transformed)

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

                    # Store last fold test data for comparison graph
                    if fold == n_splits - 1:
                        last_group_test_data = {
                            'X_test': X_test_all,
                            'y_test': y_test_all
                        }

            # Average CV metrics across folds
            for metric in global_train_metrics.keys():
                global_train_metrics[metric] /= n_splits
                global_test_metrics[metric] /= n_splits

        # === FULL DATASET EVALUATION ===
        else:
            # Train on full dataset and evaluate on same dataset
            full_model = algorithm.fit(X_full, y_full)
            y_full_pred = full_model.predict(X_full)
            
            # Calculate full dataset metrics
            full_dataset_metrics = evaluate(y_full, y_full_pred, squared=is_log_transformed)
            
            # Replace global test metrics with full dataset metrics
            global_test_metrics = full_dataset_metrics.copy()
            global_train_metrics = full_dataset_metrics.copy()
            
            # Create single "fold" entry for full dataset
            evaluation_results['Fold'].append('Full-Dataset')
            evaluation_results['R2'].append(full_dataset_metrics['R2'])
            evaluation_results['FSD'].append(full_dataset_metrics['FSD'])
            evaluation_results['PE10'].append(full_dataset_metrics['PE10'])
            evaluation_results['RT20'].append(full_dataset_metrics['RT20'])
            
            train_results['R2'].append(full_dataset_metrics['R2'])
            train_results['FSD'].append(full_dataset_metrics['FSD'])
            train_results['PE10'].append(full_dataset_metrics['PE10'])
            train_results['RT20'].append(full_dataset_metrics['RT20'])
            
            # For full dataset mode, store single prediction set
            fold_raw_predictions.append({
                'fold': 'Full-Dataset',
                'y_actual': y_full.values.tolist(),
                'y_predicted': y_full_pred.tolist(),
                'fold_name': 'Full-Dataset'
            })
            
            # Use full dataset for comparison graph
            last_test_idx = None
            last_group_test_data = None

        # === FINAL MODEL TRAINING ===
        # Train final model on complete dataset
        final_model = algorithm.fit(X_full, y_full)

        # === PREPARE COMPARISON GRAPH DATA ===
        if full_dataset_eval:
            # Use full dataset predictions for comparison graph
            y_test_last = y_full
            y_pred_last = final_model.predict(X_full)
        else:
            # Use last fold test data for comparison graph
            if group is None and last_test_idx is not None:
                # Standard CV: use last fold test data with full-trained model
                y_test_last = y_full.iloc[last_test_idx]
                y_pred_last = final_model.predict(X_full.iloc[last_test_idx])
            elif group is not None and last_group_test_data is not None:
                # Group CV: use last fold test data with full-trained model
                y_test_last = last_group_test_data['y_test']
                y_pred_last = final_model.predict(last_group_test_data['X_test'])
            else:
                # Fallback: use a sample of full dataset for comparison
                sample_size = min(1000, len(y_full))
                sample_idx = np.random.choice(len(y_full), sample_size, replace=False)
                y_test_last = y_full.iloc[sample_idx]
                y_pred_last = final_model.predict(X_full.iloc[sample_idx])

        # Create result DataFrames
        evaluation_df = pd.DataFrame(evaluation_results)
        train_results_df = pd.DataFrame(train_results)

        # Prepare CV raw predictions for legacy compatibility
        if not full_dataset_eval and fold_raw_predictions:
            # Use the last fold data for representative CV predictions
            last_fold = fold_raw_predictions[-1]
            cv_raw_predictions = {
                'y_actual_all': last_fold['y_actual'],
                'y_pred_all': last_fold['y_predicted']
            }
        else:
            cv_raw_predictions = None

        # NEW: Return enhanced results with per-fold raw predictions
        return (
            final_model, evaluation_df, train_results_df,
            global_train_metrics, global_test_metrics,
            y_test_last, y_pred_last, is_log_transformed,
            cv_raw_predictions,  # Legacy compatibility
            fold_raw_predictions  # NEW: Per-fold raw predictions
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
    # ('eda', '🕵🏽‍♂️ EDA'),
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
        'bali': '🏝️ Bali Metropolitan Area (SarBaGiTa)',
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
        
        # INFO
        st.markdown("### NOTES ‼️: The min max value that shown below is not min max of the actual data, it's only a recommendation preset.")

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
        
        # Luas tanah Range Section
        st.markdown("#### 📐 Luas Tanah (Land Area)")

        col1, col2 = st.columns(2)
        with col1:
            luas_tanah_min = st.number_input(
                "Minimum Luas Tanah (m²)",
                min_value=0,
                max_value=100000,
                value=50,
                step=10,
                key="custom_luas_tanah_min"
            )
        with col2:
            luas_tanah_max = st.number_input(
                "Maximum Luas Tanah (m²)",
                min_value=0,
                max_value=100000,
                value=10000,
                step=10,
                key="custom_luas_tanah_max"
            )
        
        # Validation
        if luas_tanah_min >= luas_tanah_max:
            st.error("❌ Minimum Luas Tanah must be less than Maximum Luas Tanah")
            custom_filters_valid = False
        else:
            custom_filters_valid = True

        # Lebar Jalan Range Section
        st.markdown("#### 🛣️ Lebar Jalan di Depan (Road Width)")

        col1, col2 = st.columns(2)
        with col1:
            lebar_jalan_min = st.number_input(
                "Minimum Lebar Jalan (m)",
                min_value=0,
                max_value=100,
                value=2,
                step=1,
                key="custom_lebar_jalan_min"
            )
        with col2:
            lebar_jalan_max = st.number_input(
                "Maximum Lebar Jalan (m)",
                min_value=0,
                max_value=100,
                value=20,
                step=1,
                key="custom_lebar_jalan_max"
            )

        # Validation
        if lebar_jalan_min >= lebar_jalan_max:
            st.error("❌ Minimum Lebar Jalan must be less than Maximum Lebar Jalan")
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
        filter_summary.append(f"**Luas Tanah Range (m²):** {luas_tanah_min:,} - {luas_tanah_max:,}")
        filter_summary.append(f"**Lebar Jalan Range (m):** {lebar_jalan_min} - {lebar_jalan_max}")

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
                        luas_tanah_min=luas_tanah_min,
                        luas_tanah_max=luas_tanah_max,
                        lebar_jalan_min=lebar_jalan_min,
                        lebar_jalan_max=lebar_jalan_max,
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

        
        # # Map Visualization (Only after filtering)
        # if st.session_state.get('filter_applied', False) or len(analyzer.current_data) < len(analyzer.original_data):
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
                    # Prepare map data - include all geographic columns if available
                    map_columns = [lat_col, lon_col, 'hpm']
                    geo_cols = {}
                    
                    # Find all geographic columns
                    for geo_type in ['wadmpr', 'wadmkk', 'wadmkc', 'wadmkd']:
                        for col in analyzer.current_data.columns:
                            if geo_type in col:
                                geo_cols[geo_type] = col
                                map_columns.append(col)
                                break
                    
                    map_data = analyzer.current_data[map_columns].copy()
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
                                # Create tooltip text for THIS quantile's data only
                                tooltip_text = []
                                for idx, row in quantile_data.iterrows():
                                    tooltip = f"HPM: {row['hpm']:,.0f}<br>Quantile: {quantile}"
                                    
                                    # Add geographic info if available
                                    if 'wadmpr' in geo_cols:
                                        prov = row[geo_cols['wadmpr']] if geo_cols['wadmpr'] in row else 'N/A'
                                        tooltip += f"<br>Province: {prov}"
                                    
                                    if 'wadmkk' in geo_cols:
                                        regency = row[geo_cols['wadmkk']] if geo_cols['wadmkk'] in row else 'N/A'
                                        tooltip += f"<br>Regency: {regency}"
                                    
                                    if 'wadmkc' in geo_cols:
                                        district = row[geo_cols['wadmkc']] if geo_cols['wadmkc'] in row else 'N/A'
                                        tooltip += f"<br>District: {district}"
                                    
                                    if 'wadmkd' in geo_cols:
                                        village = row[geo_cols['wadmkd']] if geo_cols['wadmkd'] in row else 'N/A'
                                        tooltip += f"<br>Village: {village}"
                                    
                                    tooltip_text.append(tooltip)
                                
                                fig.add_trace(go.Scattermapbox(
                                    lat=quantile_data[lat_col],
                                    lon=quantile_data[lon_col],
                                    mode='markers',
                                    marker=dict(
                                        size=8,
                                        color=color_map[quantile],
                                        opacity=0.8
                                    ),
                                    text=tooltip_text,
                                    hovertemplate='<b>%{text}</b><extra></extra>',
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

# elif st.session_state.processing_step == 'eda':
#     if fun_mode:
#         st.markdown('## <img src="https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExN3VzcWJuc3EyeHB6d3N6ejBocmoxNzAwZnB1OWtqa2V3bGtrZTR0ZCZlcD12MV9naWZzX3NlYXJjaCZjdD1n/98C4E2HeR4NBm/giphy.gif" alt="data gif" style="height:72px; vertical-align:middle;"> Exploratory Data Analysis', unsafe_allow_html=True)
#     else:
#         st.markdown('## Exploratory Data Analysis')
    
#     if analyzer.current_data is None:
#         st.warning("⚠️ No data loaded. Please go back to Data Selection.")
#         if st.button("← Back to Data Selection"):
#             st.session_state.processing_step = 'selection'
#             st.rerun()
#         st.stop()
#     else:
#         st.info("🎯 **Interactive EDA Dashboard** - Create charts, explore patterns, and analyze your data visually")
        
#         try:
#             df = analyzer.current_data.copy()  # Use analyzer.current_data for consistency
            
#             # Create unique spec path for this dataset
#             data_hash = str(hash(str(df.shape) + str(df.columns.tolist())))[:8]
#             spec_path = f"./pyg_config_{data_hash}.json"
            
#             # Initialize PyGWalker
#             pyg_app = get_pyg_renderer(df, spec_path)
            
#             # Display the interactive explorer
#             pyg_app.explorer()
            
#         except Exception as e:
#             st.error(f"❌ EDA visualization failed: {str(e)}")
#             st.info("💡 Make sure pygwalker is installed: `pip install pygwalker`")
            
#             # Fallback: Show basic info
#             st.markdown("### 📋 Basic Data Information")
#             col1, col2, col3 = st.columns(3)
            
#             with col1:
#                 st.metric("Rows", f"{len(analyzer.current_data):,}")
#             with col2:
#                 st.metric("Columns", len(analyzer.current_data.columns))
#             with col3:
#                 numeric_cols = len(analyzer.current_data.select_dtypes(include=[np.number]).columns)
#                 st.metric("Numeric Columns", numeric_cols)

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
            group_column = None

            if remove_outliers:
                st.info("🎯 Outlier detection on 'hpm' column")
                
                # **ENHANCED VALIDATION**: Check for all required columns
                required_cols = ['hpm', 'id']
                missing_cols = [col for col in required_cols if col not in analyzer.current_data.columns]
                
                if missing_cols:
                    st.error(f"❌ Missing required columns for outlier detection: {missing_cols}")
                    st.info("💡 Required columns: 'hpm' (target values) and 'id' (unique identifier)")
                    remove_outliers = False
                else:
                    # Group column selection (only wadmkc and wadmkd)
                    available_group_cols = []
                    for col in ['wadmkc', 'wadmkd']:
                        if col in analyzer.current_data.columns:
                            available_group_cols.append(col)
                    
                    if not available_group_cols:
                        st.error("❌ No valid group columns found. Need 'wadmkc' or 'wadmkd'")
                        st.info("💡 Outlier detection works best with geographic grouping columns")
                        remove_outliers = False
                    else:
                        group_column = st.selectbox(
                            "Group column for outlier detection", 
                            available_group_cols,
                            help="Choose geographic level for group-wise outlier detection"
                        )
                        
                        # **NEW**: Show outlier detection parameters
                        show_advanced = st.checkbox("⚙️ Show Advanced Parameters", value=False)
                        
                        if show_advanced:
                            col1_param, col2_param, col3_param = st.columns(3)
                            
                            with col1_param:
                                p10_quantile = st.slider(
                                    "Lower quantile (P10)", 
                                    min_value=0.05, max_value=0.2, value=0.1, step=0.01,
                                    help="Lower boundary for outlier detection"
                                )
                            
                            with col2_param:
                                p90_quantile = st.slider(
                                    "Upper quantile (P90)", 
                                    min_value=0.8, max_value=0.95, value=0.9, step=0.01,
                                    help="Upper boundary for outlier detection"
                                )
                            
                            with col3_param:
                                skew_threshold = st.slider(
                                    "Skew threshold", 
                                    min_value=0.1, max_value=1.0, value=0.5, step=0.1,
                                    help="Threshold for choosing mean vs median"
                                )
                        else:
                            p10_quantile = 0.1
                            p90_quantile = 0.9
                            skew_threshold = 0.5
                        
                        # **NEW**: Preview outlier detection
                        if st.button("🔍 Preview Outlier Detection", help="See how many outliers would be detected"):
                            with st.spinner("Analyzing outliers..."):
                                try:
                                    # **ENHANCED**: Better data validation before preview
                                    preview_df = analyzer.current_data.copy()
                                    
                                    # Check data quality
                                    missing_hpm = preview_df['hpm'].isnull().sum()
                                    missing_id = preview_df['id'].isnull().sum()
                                    missing_group = preview_df[group_column].isnull().sum()
                                    
                                    if missing_hpm > 0:
                                        st.warning(f"⚠️ Found {missing_hpm:,} missing values in 'hpm' column - these will be excluded")
                                    if missing_id > 0:
                                        st.warning(f"⚠️ Found {missing_id:,} missing values in 'id' column - these will be excluded")
                                    if missing_group > 0:
                                        st.warning(f"⚠️ Found {missing_group:,} missing values in '{group_column}' column - these will be excluded")
                                    
                                    # Check if we have enough data after cleaning
                                    clean_data_count = len(preview_df.dropna(subset=['hpm', 'id', group_column]))
                                    if clean_data_count < 10:
                                        st.error(f"❌ Not enough clean data: only {clean_data_count} valid rows remaining")
                                        st.stop()
                                    
                                    # Run outlier detection
                                    flagged_preview = analyzer.detect_outliers(
                                        preview_df,
                                        id_col='id',
                                        group_column=group_column,
                                        value_column='hpm',
                                        p10_quantile=p10_quantile,
                                        p90_quantile=p90_quantile,
                                        skew_threshold=skew_threshold
                                    )
                                    
                                    if flagged_preview.empty:
                                        st.error("❌ Outlier detection returned empty results")
                                        st.stop()
                                    
                                    outliers_count = int(flagged_preview['is_outlier'].sum())
                                    total_count = len(flagged_preview)
                                    outlier_pct = (outliers_count / total_count) * 100 if total_count > 0 else 0
                                    
                                    # Display results
                                    col1, col2, col3 = st.columns(3)
                                    with col1:
                                        st.metric("Total Records", f"{total_count:,}")
                                    with col2:
                                        st.metric("Outliers Found", f"{outliers_count:,}")
                                    with col3:
                                        st.metric("Outlier Rate", f"{outlier_pct:.1f}%")
                                    
                                    if outlier_pct > 50:
                                        st.error(f"⚠️ **High outlier rate**: {outliers_count:,} outliers ({outlier_pct:.1f}%) detected")
                                        st.warning("🔧 **Suggestion**: Consider adjusting parameters - this seems excessive")
                                        st.info("💡 Try increasing P10 (e.g., 0.05) or decreasing P90 (e.g., 0.95)")
                                    elif outlier_pct > 20:
                                        st.warning(f"⚠️ **Moderate outlier rate**: {outliers_count:,} outliers ({outlier_pct:.1f}%) detected")
                                        st.info("This is acceptable but consider if it matches your expectations")
                                    else:
                                        st.success(f"✅ **Normal outlier rate**: {outliers_count:,} outliers ({outlier_pct:.1f}%) detected")
                                    
                                    # Show sample outliers
                                    if outliers_count > 0:
                                        st.markdown("**📋 Sample outliers (showing up to 10):**")
                                        outlier_cols = ['id', group_column, 'hpm', 'upper_border', 'lower_border']
                                        available_cols = [col for col in outlier_cols if col in flagged_preview.columns]
                                        
                                        sample_outliers = flagged_preview[flagged_preview['is_outlier'] == 1][available_cols].head(10)
                                        st.dataframe(sample_outliers, use_container_width=True)
                                        
                                        # Show distribution by group
                                        if len(sample_outliers) > 0:
                                            outlier_by_group = (
                                                flagged_preview[flagged_preview['is_outlier'] == 1]
                                                .groupby(group_column)
                                                .size()
                                                .sort_values(ascending=False)
                                                .head(10)
                                            )
                                            st.markdown("**🏘️ Top groups with most outliers:**")
                                            for group, count in outlier_by_group.items():
                                                st.write(f"• **{group}**: {count} outliers")
                                    else:
                                        st.info("✅ No outliers detected with current parameters")
                                    
                                except Exception as e:
                                    st.error(f"❌ **Preview failed**: {str(e)}")
                                    
                                    # **ENHANCED**: Provide specific troubleshooting
                                    if "NaN" in str(e):
                                        st.info("💡 **Issue**: Missing values detected. Try handling missing values first in Basic Cleaning.")
                                    elif "convert" in str(e):
                                        st.info("💡 **Issue**: Data type problem. Check if 'hpm' column contains numeric values.")
                                    elif "empty" in str(e).lower():
                                        st.info("💡 **Issue**: Not enough valid data. Check your data quality.")
                                    else:
                                        st.info("💡 **Suggestion**: Try running Basic Cleaning (handle missing values) first, then retry outlier detection.")
                                    
                                    # Show data quality summary
                                    st.markdown("**🔍 Data Quality Check:**")
                                    quality_info = []
                                    if 'hpm' in analyzer.current_data.columns:
                                        hpm_nulls = analyzer.current_data['hpm'].isnull().sum()
                                        hpm_numeric = pd.to_numeric(analyzer.current_data['hpm'], errors='coerce').notna().sum()
                                        quality_info.append(f"• HPM column: {hpm_nulls} nulls, {hpm_numeric} numeric values")
                                    
                                    if 'id' in analyzer.current_data.columns:
                                        id_nulls = analyzer.current_data['id'].isnull().sum()
                                        quality_info.append(f"• ID column: {id_nulls} nulls")
                                    
                                    if group_column in analyzer.current_data.columns:
                                        group_nulls = analyzer.current_data[group_column].isnull().sum()
                                        group_unique = analyzer.current_data[group_column].nunique()
                                        quality_info.append(f"• {group_column} column: {group_nulls} nulls, {group_unique} unique groups")
                                    
                                    for info in quality_info:
                                        st.write(info)

        # **ENHANCED**: Cleaning options dictionary with all parameters
        cleaning_options = {
            'remove_duplicates': remove_duplicates,
            'handle_missing': handle_missing,
            'remove_outliers': remove_outliers,
            'outlier_column': group_column,
            'p10_quantile': p10_quantile if remove_outliers else 0.1,
            'p90_quantile': p90_quantile if remove_outliers else 0.9,
            'skew_threshold': skew_threshold if remove_outliers else 0.5
        }
        
        # **ENHANCED**: Apply cleaning button with better validation
        cleaning_enabled = (
            remove_duplicates or handle_missing or 
            (remove_outliers and group_column is not None)
        )
        
        if not cleaning_enabled:
            st.info("💡 Select at least one cleaning option above")
        
        if st.button("🧹 Apply Data Cleaning", type="primary", disabled=not cleaning_enabled):
            with st.spinner("Cleaning data..."):
                success, message = analyzer.clean_data(cleaning_options)
                if success:
                    st.success(message)
                    
                    # **NEW**: Show cleaning summary
                    if remove_outliers and group_column:
                        st.info(f"✅ Outlier removal completed using {group_column} grouping")
                    if remove_duplicates:
                        st.info("✅ Duplicate removal completed")
                    if handle_missing:
                        st.info("✅ Missing value handling completed")
                    
                    st.rerun()
                else:
                    st.error(message)
                    
                    # **NEW**: Provide helpful suggestions
                    if "Missing required columns" in message:
                        st.info("💡 **Suggestion**: Make sure your data has 'id' and 'hpm' columns")
                    elif "excessive" in message.lower():
                        st.info("💡 **Suggestion**: Try adjusting the quantile parameters (P10/P90) to be less restrictive")
                    elif "Group column" in message:
                        st.info("💡 **Suggestion**: Check if your data has geographic columns like 'wadmkc' or 'wadmkd'")

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
                                
                                # Initialize encoding tracking
                                if 'categorical_encodings' not in st.session_state:
                                    st.session_state.categorical_encodings = {}
                                
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
                                        
                                        # Track encoding info
                                        st.session_state.categorical_encodings[col] = {
                                            'type': 'ordinal',
                                            'mapping': ordinal_mapping,
                                            'new_column': new_col_name
                                        }
                                        
                                        encoding_results.append(f"✅ {col} → {new_col_name} (ordinal)")
                                        
                                        # Track in transformed columns
                                        analyzer.transformed_columns[col] = new_col_name
                                    
                                    elif config['method'] == "One-Hot Encoding":
                                        # Apply one-hot encoding with leave-one-out
                                        reference_cat = config['reference_category']
                                        categories_to_encode = [val for val in config['unique_values'] if val != reference_cat]
                                        
                                        columns_created = []
                                        for val in categories_to_encode:
                                            # Create column name (clean the value name)
                                            clean_val = str(val).lower().replace(' ', '_').replace('-', '_')
                                            clean_val = ''.join(c for c in clean_val if c.isalnum() or c == '_')
                                            new_col_name = f"is_{clean_val}"
                                            
                                            # Create binary column
                                            encoded_df[new_col_name] = (encoded_df[col] == val).astype(int)
                                            columns_created.append(new_col_name)
                                        
                                        # Track encoding info
                                        st.session_state.categorical_encodings[col] = {
                                            'type': 'one_hot',
                                            'columns_created': columns_created,
                                            'reference_category': reference_cat,
                                            'original_categories': config['unique_values']
                                        }
                                        
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
    # Direct ML implementation with 3-tab structure
    if fun_mode:
        st.markdown('## <img src="https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExaWZoeTByeWI1YmdsMHU3dnJ3ejNnem04MmM4Zjh5eThvbG10ZjFiaCZlcD12MV9naWZzX3NlYXJjaCZjdD1n/Gf1RA1jNSpbbuDE40m/giphy.gif" alt="data gif" style="height:96px; vertical-align:middle;"> Machine Learning Models', unsafe_allow_html=True)
    else:
        st.markdown('## Machine Learning Models')

    if analyzer.current_data is not None:

        # Helper functions for RERF
        def evaluate_rerf_optuna(X_columns, y_column, data, n_estimators, max_depth, min_samples_split, min_samples_leaf, max_features, group_column, n_splits, random_state, min_sample):
            """Evaluate RERF model for Optuna optimization"""
            try:
                from sklearn.model_selection import KFold
                
                # Prepare data
                model_vars = [y_column] + X_columns
                df_model = data[model_vars].dropna()
                X = df_model[X_columns]
                y = df_model[y_column]
                
                # Cross-validation setup
                if group_column is None:
                    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
                    splits = list(kf.split(X))
                else:
                    # Group-based splits (simplified for Optuna)
                    splits = []
                    for fold in range(n_splits):
                        X_train_, X_test_ = [], []
                        y_train_, y_test_ = [], []
                        
                        for group_value in data[group_column].unique():
                            data_group = data[data[group_column] == group_value]
                            if len(data_group) > min_sample:
                                X_group = data_group[X_columns]
                                y_group = data_group[y_column]
                                X_train, X_test, y_train, y_test = train_test_split(
                                    X_group, y_group, test_size=0.33, random_state=random_state + fold
                                )
                                X_train_.append(X_train)
                                X_test_.append(X_test)
                                y_train_.append(y_train)
                                y_test_.append(y_test)
                            else:
                                X_train_.append(data_group[X_columns])
                                y_train_.append(data_group[y_column])
                        
                        if X_train_ and X_test_:
                            X_train_all = pd.concat(X_train_)
                            X_test_all = pd.concat(X_test_)
                            y_train_all = pd.concat(y_train_)
                            y_test_all = pd.concat(y_test_)
                            
                            train_idx = X_train_all.index
                            test_idx = X_test_all.index
                            splits.append((train_idx, test_idx))
                
                # Evaluate across folds
                fold_metrics = []
                for train_idx, test_idx in splits:
                    if group_column is None:
                        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
                    else:
                        X_train, X_test = df_model.loc[train_idx, X_columns], df_model.loc[test_idx, X_columns]
                        y_train, y_test = df_model.loc[train_idx, y_column], df_model.loc[test_idx, y_column]
                    
                    # RERF: Linear Regression + RF on residuals
                    lr_model = LinearRegression()
                    lr_model.fit(X_train, y_train)
                    lr_pred_train = lr_model.predict(X_train)
                    lr_pred_test = lr_model.predict(X_test)
                    
                    # Calculate residuals
                    residuals_train = y_train - lr_pred_train
                    
                    # Train RF on residuals
                    rf_model = RandomForestRegressor(
                        n_estimators=n_estimators,
                        max_depth=max_depth,
                        min_samples_split=min_samples_split,
                        min_samples_leaf=min_samples_leaf,
                        max_features=max_features,
                        random_state=random_state
                    )
                    rf_model.fit(X_train, residuals_train)
                    rf_pred_residuals_test = rf_model.predict(X_test)
                    
                    # Final RERF prediction
                    rerf_pred_test = lr_pred_test + rf_pred_residuals_test
                    
                    # Evaluate with same function as other models - FIXED
                    test_metrics = evaluate(y_test, rerf_pred_test, squared=True)
                    fold_metrics.append(test_metrics)
                
                # Average metrics across folds
                avg_r2 = np.mean([m['R2'] for m in fold_metrics])
                return avg_r2
                
            except Exception as e:
                return -999

        def train_rerf_model(data, X_columns, y_column, linear_model, rf_model, group_column, n_splits, random_state, min_sample, full_dataset_eval=False):
            """
            Enhanced RERF training that stores raw predictions per fold for dynamic metric recalculation
            """
            is_log_transformed = ('ln_' in y_column) or ('log_' in y_column.lower())

            try:
                # Prepare data
                model_vars = [y_column] + X_columns
                df_model = data[model_vars].dropna()
                X = df_model[X_columns]
                y = df_model[y_column]
                
                # Initialize results containers
                evaluation_results = {'Fold': [], 'R2': [], 'FSD': [], 'PE10': [], 'RT20': []}
                train_results = {'R2': [], 'FSD': [], 'PE10': [], 'RT20': []}
                
                # NEW: Store raw predictions per fold for dynamic recalculation
                fold_raw_predictions = []
                
                global_train_metrics = {'R2': 0, 'FSD': 0, 'PE10': 0, 'RT20': 0}
                global_test_metrics = {'R2': 0, 'FSD': 0, 'PE10': 0, 'RT20': 0}
                
                # Store test indices for final comparison graph
                last_test_idx = None
                last_group_test_data = None
                
                # === CROSS-VALIDATION EVALUATION ===
                if not full_dataset_eval:
                    # --- Standard KFold Cross-Validation ---
                    if group_column is None:
                        from sklearn.model_selection import KFold
                        kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
                        
                        for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
                            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
                            
                            # RERF Step 1: Train Linear Regression
                            lr_model = LinearRegression()
                            lr_model.fit(X_train, y_train)
                            lr_pred_train = lr_model.predict(X_train)
                            lr_pred_test = lr_model.predict(X_test)
                            
                            # RERF Step 2: Calculate residuals
                            residuals_train = y_train - lr_pred_train
                            
                            # RERF Step 3: Train RF on residuals
                            rf_model_fold = RandomForestRegressor(
                                n_estimators=rf_model.n_estimators,
                                max_depth=rf_model.max_depth,
                                min_samples_split=rf_model.min_samples_split,
                                min_samples_leaf=rf_model.min_samples_leaf,
                                max_features=rf_model.max_features,
                                random_state=rf_model.random_state
                            )
                            rf_model_fold.fit(X_train, residuals_train)
                            rf_pred_residuals_train = rf_model_fold.predict(X_train)
                            rf_pred_residuals_test = rf_model_fold.predict(X_test)
                            
                            # RERF Step 4: Final predictions
                            rerf_pred_train = lr_pred_train + rf_pred_residuals_train
                            rerf_pred_test = lr_pred_test + rf_pred_residuals_test
                            
                            # NEW: Store raw predictions for this fold
                            fold_raw_predictions.append({
                                'fold': fold + 1,
                                'y_actual': y_test.values.tolist(),
                                'y_predicted': rerf_pred_test.tolist(),
                                'fold_name': f"Fold-{fold + 1}"
                            })
                            
                            # Evaluate with same function as other models
                            train_metrics = evaluate(y_train, rerf_pred_train, squared=is_log_transformed)
                            test_metrics = evaluate(y_test, rerf_pred_test, squared=is_log_transformed)
                            
                            # Store metrics
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
                            
                            # Store last fold test indices for comparison graph
                            if fold == n_splits - 1:
                                last_test_idx = test_idx
                    
                    # --- Group-Based Cross-Validation ---
                    else:
                        for fold in range(n_splits):
                            X_train_, X_test_ = [], []
                            y_train_, y_test_ = [], []
                            
                            for group_value in data[group_column].unique():
                                data_group = data[data[group_column] == group_value]
                                if len(data_group) > min_sample:
                                    X_group = data_group[X_columns]
                                    y_group = data_group[y_column]
                                    X_train, X_test, y_train, y_test = train_test_split(
                                        X_group, y_group, test_size=0.33, random_state=random_state + fold
                                    )
                                    X_train_.append(X_train)
                                    X_test_.append(X_test)
                                    y_train_.append(y_train)
                                    y_test_.append(y_test)
                                else:
                                    X_train_.append(data_group[X_columns])
                                    y_train_.append(data_group[y_column])
                            
                            X_train_all = pd.concat(X_train_)
                            X_test_all = pd.concat(X_test_)
                            y_train_all = pd.concat(y_train_)
                            y_test_all = pd.concat(y_test_)
                            
                            # RERF Steps for group-based CV
                            lr_model = LinearRegression()
                            lr_model.fit(X_train_all, y_train_all)
                            lr_pred_train = lr_model.predict(X_train_all)
                            lr_pred_test = lr_model.predict(X_test_all)
                            
                            residuals_train = y_train_all - lr_pred_train
                            
                            rf_model_fold = RandomForestRegressor(
                                n_estimators=rf_model.n_estimators,
                                max_depth=rf_model.max_depth,
                                min_samples_split=rf_model.min_samples_split,
                                min_samples_leaf=rf_model.min_samples_leaf,
                                max_features=rf_model.max_features,
                                random_state=rf_model.random_state
                            )
                            rf_model_fold.fit(X_train_all, residuals_train)
                            rf_pred_residuals_train = rf_model_fold.predict(X_train_all)
                            rf_pred_residuals_test = rf_model_fold.predict(X_test_all)
                            
                            rerf_pred_train = lr_pred_train + rf_pred_residuals_train
                            rerf_pred_test = lr_pred_test + rf_pred_residuals_test
                            
                            # NEW: Store raw predictions for this fold
                            fold_raw_predictions.append({
                                'fold': fold + 1,
                                'y_actual': y_test_all.values.tolist(),
                                'y_predicted': rerf_pred_test.tolist(),
                                'fold_name': f"Fold-{fold + 1}"
                            })
                            
                            # Evaluate with same function as other models
                            train_metrics = evaluate(y_train_all, rerf_pred_train, squared=is_log_transformed)
                            test_metrics = evaluate(y_test_all, rerf_pred_test, squared=is_log_transformed)
                            
                            # Store metrics
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
                            
                            # Store last fold test data for comparison graph
                            if fold == n_splits - 1:
                                last_group_test_data = {
                                    'X_test': X_test_all,
                                    'y_test': y_test_all
                                }
                    
                    # Average CV metrics across folds
                    for metric in global_train_metrics.keys():
                        global_train_metrics[metric] /= n_splits
                        global_test_metrics[metric] /= n_splits

                # === FULL DATASET EVALUATION ===
                else:
                    # RERF on full dataset
                    lr_model_full = LinearRegression()
                    lr_model_full.fit(X, y)
                    lr_pred_full = lr_model_full.predict(X)
                    
                    residuals_full = y - lr_pred_full
                    
                    rf_model_full = RandomForestRegressor(
                        n_estimators=rf_model.n_estimators,
                        max_depth=rf_model.max_depth,
                        min_samples_split=rf_model.min_samples_split,
                        min_samples_leaf=rf_model.min_samples_leaf,
                        max_features=rf_model.max_features,
                        random_state=rf_model.random_state
                    )
                    rf_model_full.fit(X, residuals_full)
                    rf_pred_residuals_full = rf_model_full.predict(X)
                    
                    rerf_pred_full = lr_pred_full + rf_pred_residuals_full
                    
                    # Evaluate on full dataset
                    full_dataset_metrics = evaluate(y, rerf_pred_full, squared=is_log_transformed)
                    
                    # Replace global metrics with full dataset metrics
                    global_test_metrics = full_dataset_metrics.copy()
                    global_train_metrics = full_dataset_metrics.copy()
                    
                    # Create single "fold" entry for full dataset
                    evaluation_results['Fold'].append('Full-Dataset')
                    evaluation_results['R2'].append(full_dataset_metrics['R2'])
                    evaluation_results['FSD'].append(full_dataset_metrics['FSD'])
                    evaluation_results['PE10'].append(full_dataset_metrics['PE10'])
                    evaluation_results['RT20'].append(full_dataset_metrics['RT20'])
                    
                    train_results['R2'].append(full_dataset_metrics['R2'])
                    train_results['FSD'].append(full_dataset_metrics['FSD'])
                    train_results['PE10'].append(full_dataset_metrics['PE10'])
                    train_results['RT20'].append(full_dataset_metrics['RT20'])
                    
                    # For full dataset mode, store single prediction set
                    fold_raw_predictions.append({
                        'fold': 'Full-Dataset',
                        'y_actual': y.values.tolist(),
                        'y_predicted': rerf_pred_full.tolist(),
                        'fold_name': 'Full-Dataset'
                    })
                    
                    # Clear test data indicators for full dataset mode
                    last_test_idx = None
                    last_group_test_data = None

                # === FINAL MODEL TRAINING ===
                # Train final models on complete dataset for deployment
                final_linear_model = LinearRegression()
                final_linear_model.fit(X, y)
                final_lr_pred = final_linear_model.predict(X)
                
                final_residuals = y - final_lr_pred
                
                final_rf_model = RandomForestRegressor(
                    n_estimators=rf_model.n_estimators,
                    max_depth=rf_model.max_depth,
                    min_samples_split=rf_model.min_samples_split,
                    min_samples_leaf=rf_model.min_samples_leaf,
                    max_features=rf_model.max_features,
                    random_state=rf_model.random_state
                )
                final_rf_model.fit(X, final_residuals)
                
                # Create combined final model
                final_model = {
                    'linear': final_linear_model,
                    'rf': final_rf_model,
                    'type': 'rerf'
                }
                
                # === PREPARE COMPARISON GRAPH DATA ===
                if full_dataset_eval:
                    y_test_last = y
                    final_lr_pred_graph = final_linear_model.predict(X)
                    final_rf_pred_graph = final_rf_model.predict(X)
                    rerf_pred_last = final_lr_pred_graph + final_rf_pred_graph
                else:
                    # Use last fold test data for comparison graph
                    if group_column is None and last_test_idx is not None:
                        y_test_last = y.iloc[last_test_idx]
                        X_test_last = X.iloc[last_test_idx]
                        lr_pred_last = final_linear_model.predict(X_test_last)
                        rf_pred_residuals_last = final_rf_model.predict(X_test_last)
                        rerf_pred_last = lr_pred_last + rf_pred_residuals_last
                        
                    elif group_column is not None and last_group_test_data is not None:
                        y_test_last = last_group_test_data['y_test']
                        X_test_last = last_group_test_data['X_test']
                        lr_pred_last = final_linear_model.predict(X_test_last)
                        rf_pred_residuals_last = final_rf_model.predict(X_test_last)
                        rerf_pred_last = lr_pred_last + rf_pred_residuals_last
                        
                    else:
                        sample_size = min(1000, len(y))
                        sample_idx = np.random.choice(len(y), sample_size, replace=False)
                        y_test_last = y.iloc[sample_idx]
                        X_test_sample = X.iloc[sample_idx]
                        lr_pred_sample = final_linear_model.predict(X_test_sample)
                        rf_pred_residuals_sample = final_rf_model.predict(X_test_sample)
                        rerf_pred_last = lr_pred_sample + rf_pred_residuals_sample
                
                # Create result DataFrames
                evaluation_df = pd.DataFrame(evaluation_results)
                train_results_df = pd.DataFrame(train_results)
                
                # Check if log transformed
                is_log_transformed = ('ln_' in y_column) or ('log_' in y_column.lower())
                
                # Prepare CV raw predictions for legacy compatibility
                if not full_dataset_eval and fold_raw_predictions:
                    last_fold = fold_raw_predictions[-1]
                    cv_raw_predictions = {
                        'y_actual_all': last_fold['y_actual'],
                        'y_pred_all': last_fold['y_predicted']
                    }
                else:
                    cv_raw_predictions = None

                # NEW: Return enhanced results with per-fold raw predictions
                return (
                    final_model, evaluation_df, train_results_df,
                    global_train_metrics, global_test_metrics,
                    y_test_last, rerf_pred_last, is_log_transformed,
                    cv_raw_predictions,  # Legacy compatibility
                    fold_raw_predictions  # NEW: Per-fold raw predictions
                )
                
            except Exception as e:
                st.error(f"RERF training failed: {str(e)}")
                return None, None, None, None, None, None, None, False, None, None

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
                st.write(f"**Target (Y):** {ml_y_column}")
                st.write(f"**Features ({len(ml_x_columns)}):** {', '.join(ml_x_columns[:3])}{'...' if len(ml_x_columns) > 3 else ''}")
            else:
                ml_y_column = st.selectbox("Target Variable (Y)", numeric_columns, key="ml_y_select")
                available_x_cols = [col for col in numeric_columns if col != ml_y_column]
                ml_x_columns = st.multiselect("Feature Variables (X)", available_x_cols,
                                            default=available_x_cols[:5] if len(available_x_cols) >= 5 else available_x_cols,
                                            key="ml_x_select")
        
        with col2:
            st.markdown("**Cross-Validation Configuration:**")
            
            # Group column selection
            use_group = st.checkbox("Use Group-Based Cross-Validation", value=False)
            group_column = None
            
            if use_group:
                # Find geographic columns
                geo_candidates = []
                for col in analyzer.current_data.columns:
                    if any(geo in col for geo in ['wadmpr', 'wadmkk', 'wadmkc', 'wadmkd']):
                        geo_candidates.append(col)
                
                # Also include encoded columns
                encoded_candidates = [col for col in analyzer.current_data.columns if '_encoded' in col.lower()]
                
                all_group_candidates = geo_candidates + encoded_candidates
                
                if all_group_candidates:
                    group_column = st.selectbox("Select Group Column", all_group_candidates)
                else:
                    st.warning("No geographic or encoded columns found for grouping")
                    use_group = False
            
            min_sample = st.number_input("Minimum Sample per Group", min_value=1, max_value=10, value=3)
            n_splits = st.number_input("Cross-Validation Folds", min_value=3, max_value=20, value=10)
            random_state = st.number_input("Random State", min_value=1, max_value=1000, value=101)

        # Tab structure
        tab1, tab2, tab3 = st.tabs(["🎯 Hyperparameter Tuning", "🚀 Model Training & Evaluation", "📊 Model Comparison"])
        
        with tab1:
            st.markdown("### 🎯 Optuna Hyperparameter Optimization")
            st.info("💡 **Optional**: Optimize hyperparameters for better model performance")
            
            if not ml_x_columns:
                st.warning("⚠️ Please configure model variables first")
            else:
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown("**Models to Optimize:**")
                    optimize_rf = st.checkbox("Random Forest", value=True, key="opt_rf")
                    optimize_gbdt = st.checkbox("Gradient Boosting", value=True, key="opt_gbdt")
                    optimize_rerf = st.checkbox("RERF (Linear+RF)", value=True, key="opt_rerf")
                
                with col2:
                    st.markdown("**Optimization Settings:**")
                    objective_metric = st.selectbox("Optimization Objective", 
                                                ['R2', 'PE10', 'RT20', 'FSD'], 
                                                index=0)
                    n_trials = st.number_input("Number of Trials", min_value=10, max_value=200, value=50)
                
                with col3:
                    st.markdown("**Current Best Parameters:**")
                    if 'optuna_results' in st.session_state:
                        results = st.session_state.optuna_results
                        for model_name, params in results.items():
                            st.write(f"**{model_name}**: {params.get('best_value', 'N/A'):.4f}")
                    else:
                        st.info("No optimization results yet")
                
                if st.button("🔍 Run Hyperparameter Optimization", type="primary"):
                    
                    optuna_results = {}
                    models_to_optimize = []
                    
                    if optimize_rf:
                        models_to_optimize.append(('Random Forest', RandomForestRegressor))
                    if optimize_gbdt:
                        models_to_optimize.append(('Gradient Boosting', GradientBoostingRegressor))
                    if optimize_rerf:
                        models_to_optimize.append(('RERF', 'rerf_special'))
                    
                    if not models_to_optimize:
                        st.error("Please select at least one model to optimize")
                    else:
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        for i, (model_name, model_class) in enumerate(models_to_optimize):
                            status_text.text(f"Optimizing {model_name}...")
                            progress_bar.progress((i) / len(models_to_optimize))
                            
                            def objective(trial):
                                try:
                                    if model_name == 'Random Forest':
                                        n_estimators = trial.suggest_int('n_estimators', 50, 500)
                                        max_depth = trial.suggest_int('max_depth', 3, 20)
                                        min_samples_split = trial.suggest_int('min_samples_split', 2, 10)
                                        min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 5)
                                        max_features = trial.suggest_categorical('max_features', ['sqrt', 'log2', None])
                                        
                                        model = RandomForestRegressor(
                                            n_estimators=n_estimators,
                                            max_depth=max_depth,
                                            min_samples_split=min_samples_split,
                                            min_samples_leaf=min_samples_leaf,
                                            max_features=max_features,
                                            random_state=random_state
                                        )
                                        
                                    elif model_name == 'Gradient Boosting':
                                        n_estimators = trial.suggest_int('n_estimators', 50, 500)
                                        max_depth = trial.suggest_int('max_depth', 3, 15)
                                        min_samples_split = trial.suggest_int('min_samples_split', 2, 10)
                                        min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 5)
                                        learning_rate = trial.suggest_float('learning_rate', 0.01, 0.3)
                                        subsample = trial.suggest_float('subsample', 0.8, 1.0)
                                        
                                        model = GradientBoostingRegressor(
                                            n_estimators=n_estimators,
                                            max_depth=max_depth,
                                            min_samples_split=min_samples_split,
                                            min_samples_leaf=min_samples_leaf,
                                            learning_rate=learning_rate,
                                            subsample=subsample,
                                            random_state=random_state
                                        )
                                        
                                    elif model_name == 'RERF':
                                        # For RERF, optimize only RF part
                                        n_estimators = trial.suggest_int('n_estimators', 50, 500)
                                        max_depth = trial.suggest_int('max_depth', 3, 20)
                                        min_samples_split = trial.suggest_int('min_samples_split', 2, 10)
                                        min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 5)
                                        max_features = trial.suggest_categorical('max_features', ['sqrt', 'log2', None])
                                        
                                        # Use RERF evaluator
                                        return evaluate_rerf_optuna(
                                            ml_x_columns, ml_y_column, analyzer.current_data,
                                            n_estimators, max_depth, min_samples_split, min_samples_leaf, max_features,
                                            group_column if use_group else None, n_splits, random_state, min_sample
                                        )
                                    
                                    # Run evaluation for RF and GBDT
                                    _, _, _, _, global_test_metrics, _, _, _ = analyzer.goval_machine_learning(
                                        ml_x_columns, ml_y_column, model, 
                                        group_column if use_group else None,
                                        n_splits, random_state, min_sample
                                    )
                                    
                                    # Return objective based on selected metric
                                    if objective_metric == 'R2':
                                        return global_test_metrics['R2']
                                    elif objective_metric == 'PE10':
                                        return global_test_metrics['PE10']
                                    elif objective_metric == 'RT20':
                                        return -global_test_metrics['RT20']  # Minimize
                                    elif objective_metric == 'FSD':
                                        return -global_test_metrics['FSD']  # Minimize
                                        
                                except Exception as e:
                                    return -999
                            
                            # Run optimization
                            study = optuna.create_study(direction='maximize')
                            study.optimize(objective, n_trials=n_trials)
                            
                            # Store results
                            optuna_results[model_name] = {
                                'best_params': study.best_params,
                                'best_value': study.best_value
                            }
                            
                            progress_bar.progress((i + 1) / len(models_to_optimize))
                        
                        # Store in session state
                        st.session_state.optuna_results = optuna_results
                        
                        status_text.text("Optimization completed!")
                        progress_bar.progress(1.0)
                        
                        st.success("✅ Hyperparameter optimization completed!")
                        
                        # Show results
                        for model_name, result in optuna_results.items():
                            st.markdown(f"**{model_name} Best Parameters:**")
                            st.json(result['best_params'])
                            st.write(f"**Best {objective_metric}:** {result['best_value']:.4f}")
                            st.markdown("---")

        with tab2:
            st.markdown("### 🚀 Model Training & Evaluation")
            
            if not ml_x_columns:
                st.warning("⚠️ Please configure model variables first")
            else:
                # Manual Parameters for each model
                st.markdown("#### ⚙️ Model Parameters")
                
                # Random Forest Parameters
                with st.expander("🌳 Random Forest Parameters", expanded=True):
                    col1, col2, col3 = st.columns(3)
                    
                    # Check if we have Optuna results
                    rf_defaults = {}
                    if 'optuna_results' in st.session_state and 'Random Forest' in st.session_state.optuna_results:
                        rf_defaults = st.session_state.optuna_results['Random Forest']['best_params']
                        st.info("💡 Using Optuna optimized parameters as defaults")
                    
                    with col1:
                        rf_n_estimators = st.number_input(
                            "n_estimators", 
                            min_value=10, max_value=1000, 
                            value=rf_defaults.get('n_estimators', 100),
                            key="rf_n_est"
                        )
                        rf_max_depth = st.number_input(
                            "max_depth", 
                            min_value=1, max_value=50, 
                            value=rf_defaults.get('max_depth', 30),
                            key="rf_max_depth"
                        )
                    
                    with col2:
                        rf_min_samples_split = st.number_input(
                            "min_samples_split", 
                            min_value=2, max_value=20, 
                            value=rf_defaults.get('min_samples_split', 2),
                            key="rf_min_split"
                        )
                        rf_min_samples_leaf = st.number_input(
                            "min_samples_leaf", 
                            min_value=1, max_value=10, 
                            value=rf_defaults.get('min_samples_leaf', 1),
                            key="rf_min_leaf"
                        )
                    
                    with col3:
                        rf_max_features_default = rf_defaults.get('max_features', 'sqrt')
                        rf_max_features_options = ['sqrt', 'log2', None]
                        rf_max_features_index = rf_max_features_options.index(rf_max_features_default) if rf_max_features_default in rf_max_features_options else 0
                        
                        rf_max_features = st.selectbox(
                            "max_features", 
                            rf_max_features_options,
                            index=rf_max_features_index,
                            key="rf_max_feat"
                        )
                
                # Gradient Boosting Parameters
                with st.expander("🚀 Gradient Boosting Parameters", expanded=True):
                    col1, col2, col3 = st.columns(3)
                    
                    # Check if we have Optuna results
                    gbdt_defaults = {}
                    if 'optuna_results' in st.session_state and 'Gradient Boosting' in st.session_state.optuna_results:
                        gbdt_defaults = st.session_state.optuna_results['Gradient Boosting']['best_params']
                        st.info("💡 Using Optuna optimized parameters as defaults")
                    
                    with col1:
                        gbdt_n_estimators = st.number_input(
                            "n_estimators", 
                            min_value=10, max_value=1000, 
                            value=gbdt_defaults.get('n_estimators', 100),
                            key="gbdt_n_est"
                        )
                        gbdt_learning_rate = st.number_input(
                            "learning_rate", 
                            min_value=0.01, max_value=0.5, 
                            value=gbdt_defaults.get('learning_rate', 0.1),
                            step=0.01,
                            key="gbdt_lr"
                        )
                    
                    with col2:
                        gbdt_max_depth = st.number_input(
                            "max_depth", 
                            min_value=1, max_value=20, 
                            value=gbdt_defaults.get('max_depth', 6),
                            key="gbdt_max_depth"
                        )
                        gbdt_subsample = st.number_input(
                            "subsample", 
                            min_value=0.5, max_value=1.0, 
                            value=gbdt_defaults.get('subsample', 1.0),
                            step=0.1,
                            key="gbdt_subsample"
                        )
                    
                    with col3:
                        gbdt_min_samples_split = st.number_input(
                            "min_samples_split", 
                            min_value=2, max_value=20, 
                            value=gbdt_defaults.get('min_samples_split', 2),
                            key="gbdt_min_split"
                        )
                        gbdt_min_samples_leaf = st.number_input(
                            "min_samples_leaf", 
                            min_value=1, max_value=10, 
                            value=gbdt_defaults.get('min_samples_leaf', 1),
                            key="gbdt_min_leaf"
                        )
                
                # RERF Parameters
                with st.expander("🔗 RERF (Linear + Random Forest) Parameters", expanded=True):
                    st.info("🔗 RERF combines Linear Regression + Random Forest on residuals")
                    col1, col2, col3 = st.columns(3)
                    
                    # Check if we have Optuna results
                    rerf_defaults = {}
                    if 'optuna_results' in st.session_state and 'RERF' in st.session_state.optuna_results:
                        rerf_defaults = st.session_state.optuna_results['RERF']['best_params']
                        st.info("💡 Using Optuna optimized parameters as defaults")
                    
                    with col1:
                        rerf_n_estimators = st.number_input(
                            "RF n_estimators", 
                            min_value=10, max_value=1000, 
                            value=rerf_defaults.get('n_estimators', 100),
                            key="rerf_n_est"
                        )
                        rerf_max_depth = st.number_input(
                            "RF max_depth", 
                            min_value=1, max_value=50, 
                            value=rerf_defaults.get('max_depth', 30),
                            key="rerf_max_depth"
                        )
                    
                    with col2:
                        rerf_min_samples_split = st.number_input(
                            "RF min_samples_split", 
                            min_value=2, max_value=20, 
                            value=rerf_defaults.get('min_samples_split', 2),
                            key="rerf_min_split"
                        )
                        rerf_min_samples_leaf = st.number_input(
                            "RF min_samples_leaf", 
                            min_value=1, max_value=10, 
                            value=rerf_defaults.get('min_samples_leaf', 1),
                            key="rerf_min_leaf"
                        )
                    
                    with col3:
                        rerf_max_features_default = rerf_defaults.get('max_features', 'sqrt')
                        rerf_max_features_options = ['sqrt', 'log2', None]
                        rerf_max_features_index = rerf_max_features_options.index(rerf_max_features_default) if rerf_max_features_default in rerf_max_features_options else 0
                        
                        rerf_max_features = st.selectbox(
                            "RF max_features", 
                            rerf_max_features_options,
                            index=rerf_max_features_index,
                            key="rerf_max_feat"
                        )
                        
                        st.info("Linear Regression has no hyperparameters")
                
                # NEW SECTION: Simplified ZIP name configuration
                st.markdown("---")
                st.markdown("#### 📦 Export Configuration")
                st.info("💡 Configure your model package name before training")

                col1, col2 = st.columns([2, 1])

                with col1:
                    default_timestamp = datetime.now().strftime('%Y%m%d_%H%M')
                    zip_package_name = st.text_input(
                        "Model Package Name", 
                        value=f"ml_models_package_{default_timestamp}",
                        placeholder="Enter package name (without .zip)",
                        key="zip_package_name_input",
                        help="This will be the name of your downloaded ZIP file"
                    )

                with col2:
                    st.markdown("**📋 Contents Preview:**")
                    st.write("• random_forest_model.pkl")
                    st.write("• gradient_boosting_model.pkl") 
                    st.write("• rerf_model.pkl")
                    st.write("• features_info.json")

                st.markdown("---")
                
                # st.markdown("---")
                # st.markdown("#### ⚙️ Evaluation Configuration")

                # col1, col2 = st.columns(2)

                # with col1:
                #     st.markdown("**Evaluation Mode:**")
                #     evaluation_mode = st.radio(
                #         "Choose evaluation approach:",
                #         ["Cross-Validation (Recommended)", "Full Dataset Training Performance"],
                #         index=0,
                #         help="""
                #         • Cross-Validation: More realistic, tests on unseen data
                #         • Full Dataset: Shows maximum training performance, may overestimate
                #         """,
                #         key="eval_mode_radio"
                #     )

                #     full_dataset_eval = (evaluation_mode == "Full Dataset Training Performance")

                # with col2:
                #     if full_dataset_eval:
                #         st.warning("⚠️ **Full Dataset Mode**")
                #         st.write("• Models trained and evaluated on same data")
                #         st.write("• Results show training performance")
                #         st.write("• May be overly optimistic")
                #     else:
                #         st.info("✅ **Cross-Validation Mode**")
                #         st.write("• Models tested on unseen data")
                #         st.write("• More reliable for real-world performance")
                #         st.write("• Conservative estimates")
                
                # Train All Models Button
                st.markdown("#### 🚀 Model Training")
                
                if st.button("🤖 Train All Models", type="primary", use_container_width=True):
                    with st.spinner("Training all models sequentially..."):
                        
                        all_results = {}
                        overall_progress = st.progress(0)
                        status_text = st.empty()
                        
                        # Model configurations (EXISTING - NO CHANGE)
                        models_config = [
                            {
                                'name': 'Random Forest',
                                'model': RandomForestRegressor(
                                    n_estimators=rf_n_estimators,
                                    max_depth=rf_max_depth,
                                    min_samples_split=rf_min_samples_split,
                                    min_samples_leaf=rf_min_samples_leaf,
                                    max_features=rf_max_features,
                                    random_state=random_state
                                ),
                                'type': 'standard'
                            },
                            {
                                'name': 'Gradient Boosting',
                                'model': GradientBoostingRegressor(
                                    n_estimators=gbdt_n_estimators,
                                    learning_rate=gbdt_learning_rate,
                                    max_depth=gbdt_max_depth,
                                    subsample=gbdt_subsample,
                                    min_samples_split=gbdt_min_samples_split,
                                    min_samples_leaf=gbdt_min_samples_leaf,
                                    random_state=random_state
                                ),
                                'type': 'standard'
                            },
                            {
                                'name': 'RERF',
                                'model': {
                                    'linear': LinearRegression(),
                                    'rf': RandomForestRegressor(
                                        n_estimators=rerf_n_estimators,
                                        max_depth=rerf_max_depth,
                                        min_samples_split=rerf_min_samples_split,
                                        min_samples_leaf=rerf_min_samples_leaf,
                                        max_features=rerf_max_features,
                                        random_state=random_state
                                    )
                                },
                                'type': 'rerf'
                            },
                            {
                                'name': 'OLS',
                                'model': LinearRegression(),
                                'type': 'ols'
                            }
                        ]
                        
                        # Train each model (CORRECTED SECTION)
                        for i, model_config in enumerate(models_config):
                            model_name = model_config['name']
                            status_text.text(f"Training {model_name} (CV + Full Dataset)...")
                            overall_progress.progress(i / len(models_config))
                            
                            try:
                                if model_config['type'] == 'standard':
                                    # Run CV evaluation first - ENHANCED with per-fold storage
                                    try:
                                        cv_results = analyzer.goval_machine_learning(
                                            ml_x_columns, ml_y_column, model_config['model'],
                                            group_column if use_group else None,
                                            n_splits, random_state, min_sample,
                                            full_dataset_eval=False  # CV mode
                                        )
                                        
                                        # Handle both old and new return formats
                                        if len(cv_results) == 10:  # New enhanced format
                                            (cv_model, cv_evaluation_df, cv_train_results_df, cv_global_train_metrics, 
                                            cv_global_test_metrics, cv_y_test_last, cv_y_pred_last, cv_is_log_transformed, 
                                            cv_raw_predictions, cv_fold_raw_predictions) = cv_results
                                        else:  # Old format (fallback)
                                            (cv_model, cv_evaluation_df, cv_train_results_df, cv_global_train_metrics, 
                                            cv_global_test_metrics, cv_y_test_last, cv_y_pred_last, cv_is_log_transformed, 
                                            cv_raw_predictions) = cv_results
                                            cv_fold_raw_predictions = None  # Not available in old format
                                            
                                    except Exception as e:
                                        st.error(f"CV training failed for {model_name}: {str(e)}")
                                        continue
                                    
                                    # Run Full Dataset training
                                    try:
                                        full_results = analyzer.goval_machine_learning(
                                            ml_x_columns, ml_y_column, model_config['model'],
                                            group_column if use_group else None,
                                            n_splits, random_state, min_sample,
                                            full_dataset_eval=True  # Full dataset mode
                                        )
                                        
                                        # Handle both old and new return formats
                                        if len(full_results) == 10:  # New enhanced format
                                            (full_model, full_evaluation_df, full_train_results_df, full_global_train_metrics, 
                                            full_global_test_metrics, full_y_test_last, full_y_pred_last, full_is_log_transformed, 
                                            _, full_fold_raw_predictions) = full_results
                                        else:  # Old format (fallback)
                                            (full_model, full_evaluation_df, full_train_results_df, full_global_train_metrics, 
                                            full_global_test_metrics, full_y_test_last, full_y_pred_last, full_is_log_transformed, 
                                            _) = full_results
                                            full_fold_raw_predictions = None  # Not available in old format
                                            
                                    except Exception as e:
                                        st.error(f"Full dataset training failed for {model_name}: {str(e)}")
                                        continue
                                        
                                elif model_config['type'] == 'rerf':
                                    # Run CV evaluation first - ENHANCED RERF with per-fold storage
                                    try:
                                        cv_results = train_rerf_model(
                                            analyzer.current_data, ml_x_columns, ml_y_column,
                                            model_config['model']['linear'], model_config['model']['rf'],
                                            group_column if use_group else None,
                                            n_splits, random_state, min_sample,
                                            full_dataset_eval=False  # CV mode
                                        )
                                        
                                        # Handle both old and new return formats for RERF
                                        if cv_results and len(cv_results) == 10:  # New enhanced format
                                            (cv_model, cv_evaluation_df, cv_train_results_df, cv_global_train_metrics, 
                                            cv_global_test_metrics, cv_y_test_last, cv_y_pred_last, cv_is_log_transformed, 
                                            cv_raw_predictions, cv_fold_raw_predictions) = cv_results
                                        elif cv_results and len(cv_results) == 9:  # Old format (fallback)
                                            (cv_model, cv_evaluation_df, cv_train_results_df, cv_global_train_metrics, 
                                            cv_global_test_metrics, cv_y_test_last, cv_y_pred_last, cv_is_log_transformed, 
                                            cv_raw_predictions) = cv_results
                                            cv_fold_raw_predictions = None  # Not available in old format
                                        else:
                                            st.error(f"RERF CV training failed for {model_name}")
                                            continue
                                            
                                    except Exception as e:
                                        st.error(f"RERF CV training failed for {model_name}: {str(e)}")
                                        continue
                                    
                                    # Run Full Dataset training for RERF
                                    try:
                                        full_results = train_rerf_model(
                                            analyzer.current_data, ml_x_columns, ml_y_column,
                                            model_config['model']['linear'], model_config['model']['rf'],
                                            group_column if use_group else None,
                                            n_splits, random_state, min_sample,
                                            full_dataset_eval=True  # Full dataset mode
                                        )
                                        
                                        # Handle both old and new return formats for RERF
                                        if full_results and len(full_results) == 10:  # New enhanced format
                                            (full_model, full_evaluation_df, full_train_results_df, full_global_train_metrics, 
                                            full_global_test_metrics, full_y_test_last, full_y_pred_last, full_is_log_transformed, 
                                            _, full_fold_raw_predictions) = full_results
                                        elif full_results and len(full_results) == 9:  # Old format (fallback)
                                            (full_model, full_evaluation_df, full_train_results_df, full_global_train_metrics, 
                                            full_global_test_metrics, full_y_test_last, full_y_pred_last, full_is_log_transformed, 
                                            _) = full_results
                                            full_fold_raw_predictions = None  # Not available in old format
                                        else:
                                            st.error(f"RERF Full dataset training failed for {model_name}")
                                            continue
                                            
                                    except Exception as e:
                                        st.error(f"RERF Full dataset training failed for {model_name}: {str(e)}")
                                        continue
                                
                                elif model_config['type'] == 'ols':
                                    # Handle OLS (Linear Regression) - same as standard but simpler
                                    ols_lr_model = LinearRegression()
                                    
                                    try:
                                        cv_results = analyzer.goval_machine_learning(
                                            ml_x_columns, ml_y_column, ols_lr_model,
                                            group_column if use_group else None,
                                            n_splits, random_state, min_sample,
                                            full_dataset_eval=False  # CV mode
                                        )
                                        
                                        # Handle both old and new return formats
                                        if len(cv_results) == 10:  # New enhanced format
                                            (cv_model, cv_evaluation_df, cv_train_results_df, cv_global_train_metrics, 
                                            cv_global_test_metrics, cv_y_test_last, cv_y_pred_last, cv_is_log_transformed, 
                                            cv_raw_predictions, cv_fold_raw_predictions) = cv_results
                                        else:  # Old format (fallback)
                                            (cv_model, cv_evaluation_df, cv_train_results_df, cv_global_train_metrics, 
                                            cv_global_test_metrics, cv_y_test_last, cv_y_pred_last, cv_is_log_transformed, 
                                            cv_raw_predictions) = cv_results
                                            cv_fold_raw_predictions = None  # Not available in old format
                                            
                                    except Exception as e:
                                        st.error(f"OLS CV training failed: {str(e)}")
                                        continue
                                    
                                    try:
                                        full_results = analyzer.goval_machine_learning(
                                            ml_x_columns, ml_y_column, ols_lr_model,
                                            group_column if use_group else None,
                                            n_splits, random_state, min_sample,
                                            full_dataset_eval=True  # Full dataset mode
                                        )
                                        
                                        # Handle both old and new return formats
                                        if len(full_results) == 10:  # New enhanced format
                                            (full_model, full_evaluation_df, full_train_results_df, full_global_train_metrics, 
                                            full_global_test_metrics, full_y_test_last, full_y_pred_last, full_is_log_transformed, 
                                            _, full_fold_raw_predictions) = full_results
                                        else:  # Old format (fallback)
                                            (full_model, full_evaluation_df, full_train_results_df, full_global_train_metrics, 
                                            full_global_test_metrics, full_y_test_last, full_y_pred_last, full_is_log_transformed, 
                                            _) = full_results
                                            full_fold_raw_predictions = None  # Not available in old format
                                            
                                    except Exception as e:
                                        st.error(f"OLS Full dataset training failed: {str(e)}")
                                        continue
                                
                                # Store BOTH CV and Full Dataset results with enhanced fold data
                                all_results[model_name] = {
                                    'model': full_model,
                                    'cv_evaluation_df': cv_evaluation_df,
                                    'full_evaluation_df': full_evaluation_df,
                                    'cv_global_test_metrics': cv_global_test_metrics,
                                    'full_global_test_metrics': full_global_test_metrics,
                                    'cv_raw_predictions': cv_raw_predictions,  # Legacy compatibility
                                    'cv_fold_raw_predictions': cv_fold_raw_predictions,  # NEW: Per-fold raw predictions
                                    'y_test_last': full_y_test_last,
                                    'y_pred_last': full_y_pred_last,
                                    'is_log_transformed': full_is_log_transformed,
                                    'feature_names': ml_x_columns,
                                    'target_name': ml_y_column
                                }
                                
                            except Exception as e:
                                st.error(f"❌ {model_name} training failed: {str(e)}")
                                continue
                        
                        overall_progress.progress(1.0)
                        status_text.text("All models trained successfully!")
                        
                        # Store in session state for comparison
                        if 'all_model_results' not in st.session_state:
                            st.session_state.all_model_results = {}
                        
                        # Add timestamp to results
                        wib_timezone = timezone(timedelta(hours=7))  # WIB is UTC+7
                        timestamp = datetime.now(wib_timezone).strftime('%Y%m%d_%H%M%S_WIB')
                        st.session_state.all_model_results[timestamp] = all_results
                        
                        st.success(f"✅ All models trained successfully! Results saved with timestamp: {timestamp}")
                        
                        # SIMPLIFIED DOWNLOAD SECTION - Only ZIP download button
                        st.markdown("### 💾 Download Trained Models")
                        st.info("📋 Note: OLS model is for comparison only and not included in downloads")

                        # Filter out OLS from downloads
                        downloadable_results = {k: v for k, v in all_results.items() if k != 'OLS'}

                        # Prepare ZIP file with all models using standard filenames
                        try:
                            import zipfile
                            import io
                            
                            # Create in-memory ZIP file
                            zip_buffer = io.BytesIO()
                            
                            with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                                # Add each model to ZIP with standard filenames
                                for model_name, result in downloadable_results.items():
                                    # Prepare model data
                                    model_data = {
                                        'model': result['model'],
                                        'feature_names': result['feature_names'],
                                        'target_name': result['target_name'],
                                        'model_type': model_name,
                                        'metrics': {
                                            'cv_metrics': result['cv_global_test_metrics'],
                                            'full_metrics': result['full_global_test_metrics']
                                        },
                                        'timestamp': timestamp,
                                    }
                                    
                                    # Standard filenames (no user customization)
                                    if model_name == 'Random Forest':
                                        model_filename = "random_forest_model.pkl"
                                    elif model_name == 'Gradient Boosting':
                                        model_filename = "gradient_boosting_model.pkl"
                                    elif model_name == 'RERF':
                                        model_filename = "rerf_model.pkl"
                                    
                                    model_bytes = pickle.dumps(model_data)
                                    zip_file.writestr(model_filename, model_bytes)
                                
                                # Create and add features JSON with standard filename
                                features_info = {
                                    'training_session': timestamp,
                                    'target_variable': ml_y_column,
                                    'feature_names': ml_x_columns,
                                    'feature_count': len(ml_x_columns),
                                    'models_trained': list(downloadable_results.keys()),
                                    'cross_validation': {
                                        'n_splits': n_splits,
                                        'random_state': random_state,
                                        'group_column': group_column if use_group else None
                                    },
                                    'categorical_encodings': st.session_state.get('categorical_encodings', None),
                                    'model_filenames': {
                                        'Random Forest': "random_forest_model.pkl",
                                        'Gradient Boosting': "gradient_boosting_model.pkl",
                                        'RERF': "rerf_model.pkl"
                                    },
                                    'performance_summary': {
                                        model_name: {
                                            'cv_r2': result['cv_global_test_metrics']['R2'],
                                            'cv_pe10': result['cv_global_test_metrics']['PE10'], 
                                            'cv_rt20': result['cv_global_test_metrics']['RT20'],
                                            'cv_fsd': result['cv_global_test_metrics']['FSD'],
                                            'full_r2': result['full_global_test_metrics']['R2'],
                                            'full_pe10': result['full_global_test_metrics']['PE10'],
                                            'full_rt20': result['full_global_test_metrics']['RT20'],
                                            'full_fsd': result['full_global_test_metrics']['FSD']
                                        }
                                        for model_name, result in downloadable_results.items()
                                    }
                                }
                                
                                # Add features JSON to ZIP with standard filename
                                features_json = json.dumps(features_info, indent=2)
                                zip_file.writestr("features_info.json", features_json)
                            
                            zip_buffer.seek(0)
                            
                            # Single download button with user-defined ZIP name
                            st.download_button(
                                label="📦 Download Model Package (ZIP)",
                                data=zip_buffer.getvalue(),
                                file_name=f"{zip_package_name}.zip",
                                mime="application/zip",
                                key=f"download_package_{timestamp}",
                                type="primary",
                                use_container_width=True
                            )
                            
                            st.success("✅ Click above to download your model package!")
                            st.info("📋 **Package Contents**: random_forest_model.pkl, gradient_boosting_model.pkl, rerf_model.pkl, features_info.json")
                                
                        except Exception as e:
                            st.error(f"❌ Failed to prepare model package: {str(e)}")

        with tab3:
            st.markdown("### 📊 Model Comparison Dashboard")
            
            if 'all_model_results' not in st.session_state or not st.session_state.all_model_results:
                st.info("🎯 Train models first to see comparison results here")
            else:
                # Show all training sessions
                st.markdown("#### 📈 Training History")
                
                session_options = list(st.session_state.all_model_results.keys())
                selected_session = st.selectbox(
                    "Select Training Session",
                    session_options,
                    index=len(session_options)-1,
                    key="comparison_session"
                )
                
                if selected_session:
                    results = st.session_state.all_model_results[selected_session]
                    
                    # MOVED: Evaluation Settings to the top
                    st.markdown("#### ⚙️ Evaluation Settings")
                    eval_mode_comparison = st.toggle(
                        "🔄 Evaluate in Original Scale (exp transform)", 
                        value=True, 
                        key=f"eval_mode_comparison_{selected_session}",
                        help="Toggle between ln scale (False) and original HPM scale (True)"
                    )

                    if eval_mode_comparison:
                        st.success("📈 Evaluating **all metrics below** in **Original Scale** (HPM values)")
                    else:
                        st.info("📊 Evaluating **all metrics below** in **Log Scale** (ln values)")

                    st.info("💡 **All CV fold results will now recalculate when you toggle the scale!**")
                    
                    # Cross-Validation Analytics Section - NOW FULLY DYNAMIC
                    st.markdown("#### 📊 Cross-Validation Analytics")
                    st.info("📈 Individual fold performance - **dynamically calculated** based on evaluation mode")
                    
                    for model_name, result in results.items():
                        with st.expander(f"📋 {model_name} - CV Fold Results", expanded=False):
                            
                            # Check if we have per-fold raw predictions
                            cv_fold_predictions = result.get('cv_fold_raw_predictions')
                            
                            if cv_fold_predictions is not None and len(cv_fold_predictions) > 0:
                                # DYNAMIC RECALCULATION - This will change with the toggle!
                                st.markdown(f"**📊 Individual Fold Performance ({('Original Scale' if eval_mode_comparison else 'Log Scale')}):**")
                                
                                # Recalculate metrics for each fold based on current toggle
                                dynamic_fold_results = []
                                
                                for fold_data in cv_fold_predictions:
                                    fold_name = fold_data['fold_name']
                                    y_actual = np.array(fold_data['y_actual'])
                                    y_predicted = np.array(fold_data['y_predicted'])
                                    
                                    # Recalculate metrics with current evaluation mode
                                    fold_metrics = evaluate(y_actual, y_predicted, squared=eval_mode_comparison)
                                    
                                    dynamic_fold_results.append({
                                        'Fold': fold_name,
                                        'R2': fold_metrics['R2'],
                                        'FSD': fold_metrics['FSD'], 
                                        'PE10': fold_metrics['PE10'],
                                        'RT20': fold_metrics['RT20']
                                    })
                                
                                # Create dynamic DataFrame that responds to toggle
                                dynamic_cv_df = pd.DataFrame(dynamic_fold_results)
                                
                                # Display the dynamically calculated results
                                st.dataframe(dynamic_cv_df.style.format({
                                    'R2': '{:.4f}',
                                    'FSD': '{:.4f}',
                                    'PE10': '{:.4f}',
                                    'RT20': '{:.4f}'
                                }), use_container_width=True)
                                
                                # Show dynamic summary stats
                                st.markdown(f"**📈 Summary Statistics ({('Original Scale' if eval_mode_comparison else 'Log Scale')}):**")
                                
                                col1, col2, col3, col4 = st.columns(4)
                                with col1:
                                    st.metric("CV R² Mean", f"{dynamic_cv_df['R2'].mean():.4f}")
                                    st.caption("✅ Dynamic")
                                with col2:
                                    st.metric("CV PE10 Mean", f"{dynamic_cv_df['PE10'].mean():.4f}")
                                    st.caption("✅ Dynamic")
                                with col3:
                                    st.metric("CV RT20 Mean", f"{dynamic_cv_df['RT20'].mean():.4f}")
                                    st.caption("✅ Dynamic")
                                with col4:
                                    st.metric("CV FSD Mean", f"{dynamic_cv_df['FSD'].mean():.4f}")
                                    st.caption("✅ Dynamic")
                                
                                # Show variability stats
                                st.markdown("**📊 Fold Variability:**")
                                col1, col2, col3, col4 = st.columns(4)
                                with col1:
                                    st.write(f"R² Std: {dynamic_cv_df['R2'].std():.4f}")
                                with col2:
                                    st.write(f"PE10 Std: {dynamic_cv_df['PE10'].std():.4f}")
                                with col3:
                                    st.write(f"RT20 Std: {dynamic_cv_df['RT20'].std():.4f}")
                                with col4:
                                    st.write(f"FSD Std: {dynamic_cv_df['FSD'].std():.4f}")
                                    
                            else:
                                # Fallback to original static data with clear warning
                                st.warning("⚠️ **Static Data**: Per-fold raw predictions not available. Showing original training scale only.")
                                st.markdown("**📊 Original Fold Performance (Training Scale):**")
                                
                                cv_df = result['cv_evaluation_df']
                                st.dataframe(cv_df.style.format({
                                    'R2': '{:.4f}',
                                    'FSD': '{:.4f}',
                                    'PE10': '{:.4f}',
                                    'RT20': '{:.4f}'
                                }), use_container_width=True)
                                
                                # Fallback summary stats
                                col1, col2, col3, col4 = st.columns(4)
                                with col1:
                                    st.metric("CV R² Mean", f"{cv_df['R2'].mean():.4f}")
                                    st.caption("⚠️ Static")
                                with col2:
                                    st.metric("CV PE10 Mean", f"{cv_df['PE10'].mean():.4f}")
                                    st.caption("⚠️ Static")
                                with col3:
                                    st.metric("CV RT20 Mean", f"{cv_df['RT20'].mean():.4f}")
                                    st.caption("⚠️ Static")
                                with col4:
                                    st.metric("CV FSD Mean", f"{cv_df['FSD'].mean():.4f}")
                                    st.caption("⚠️ Static")

                    # Performance Metrics Comparison - Show both CV and Full Dataset
                    st.markdown("#### 📊 Performance Metrics Comparison")

                    # Create comprehensive metrics table with DYNAMIC CV calculations
                    metrics_data = []
                    for model_name, result in results.items():
                        # Get fresh predictions for full dataset recalculation
                        y_test = result['y_test_last']
                        y_pred = result['y_pred_last']
                        
                        # Recalculate full dataset metrics with chosen evaluation mode
                        fresh_full_metrics = evaluate(y_test, y_pred, squared=eval_mode_comparison)
                        
                        # ENHANCED: Recalculate CV metrics using per-fold data when available
                        cv_fold_predictions = result.get('cv_fold_raw_predictions')
                        
                        if cv_fold_predictions is not None and len(cv_fold_predictions) > 0:
                            # Calculate CV metrics by averaging across all folds
                            fold_metrics_list = []
                            
                            for fold_data in cv_fold_predictions:
                                y_actual = np.array(fold_data['y_actual'])
                                y_predicted = np.array(fold_data['y_predicted'])
                                fold_metrics = evaluate(y_actual, y_predicted, squared=eval_mode_comparison)
                                fold_metrics_list.append(fold_metrics)
                            
                            # Average metrics across folds
                            fresh_cv_metrics = {
                                'R2': np.mean([m['R2'] for m in fold_metrics_list]),
                                'PE10': np.mean([m['PE10'] for m in fold_metrics_list]),
                                'RT20': np.mean([m['RT20'] for m in fold_metrics_list]),
                                'FSD': np.mean([m['FSD'] for m in fold_metrics_list])
                            }
                            
                        elif result.get('cv_raw_predictions') is not None and result['cv_raw_predictions']['y_actual_all']:
                            # Fallback to legacy CV predictions
                            cv_y_actual = np.array(result['cv_raw_predictions']['y_actual_all'])
                            cv_y_pred = np.array(result['cv_raw_predictions']['y_pred_all'])
                            fresh_cv_metrics = evaluate(cv_y_actual, cv_y_pred, squared=eval_mode_comparison)
                        else:
                            # Final fallback to stored CV metrics (with warning)
                            fresh_cv_metrics = result['cv_global_test_metrics']
                            if eval_mode_comparison != result.get('is_log_transformed', True):
                                st.warning(f"⚠️ {model_name}: CV metrics cannot be recalculated (using stored values)")
                        
                        metrics_data.append({
                            'Model': model_name,
                            'CV R²': fresh_cv_metrics['R2'],
                            'Full R²': fresh_full_metrics['R2'],
                            'CV PE10': fresh_cv_metrics['PE10'],
                            'Full PE10': fresh_full_metrics['PE10'],
                            'CV RT20': fresh_cv_metrics['RT20'],
                            'Full RT20': fresh_full_metrics['RT20'],
                            'CV FSD': fresh_cv_metrics['FSD'],
                            'Full FSD': fresh_full_metrics['FSD']
                        })

                    metrics_df = pd.DataFrame(metrics_data)

                    # Display comprehensive metrics table
                    st.dataframe(metrics_df.style.format({
                        'CV R²': '{:.4f}',
                        'Full R²': '{:.4f}',
                        'CV PE10': '{:.4f}',
                        'Full PE10': '{:.4f}',
                        'CV RT20': '{:.4f}',
                        'Full RT20': '{:.4f}',
                        'CV FSD': '{:.4f}',
                        'Full FSD': '{:.4f}'
                    }), use_container_width=True)
                    
                    # Side-by-side bar charts for metrics comparison (using Full Dataset metrics)
                    st.markdown("#### 📊 Metrics Comparison Charts (Full Dataset)")
                    
                    # Create side-by-side bar charts
                    fig_metrics = make_subplots(
                        rows=2, cols=2,
                        subplot_titles=('R² (Higher is Better)', 'PE10 (Higher is Better)', 
                                    'RT20 (Lower is Better)', 'FSD (Lower is Better)'),
                        specs=[[{"secondary_y": False}, {"secondary_y": False}],
                            [{"secondary_y": False}, {"secondary_y": False}]]
                    )
                    
                    models = metrics_df['Model'].tolist()
                    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
                    
                    # R² chart
                    fig_metrics.add_trace(
                        go.Bar(x=models, y=metrics_df['Full R²'], name='R²', 
                            marker_color=colors[0], showlegend=False),
                        row=1, col=1
                    )
                    
                    # PE10 chart
                    fig_metrics.add_trace(
                        go.Bar(x=models, y=metrics_df['Full PE10'], name='PE10', 
                            marker_color=colors[1], showlegend=False),
                        row=1, col=2
                    )
                    
                    # RT20 chart
                    fig_metrics.add_trace(
                        go.Bar(x=models, y=metrics_df['Full RT20'], name='RT20', 
                            marker_color=colors[2], showlegend=False),
                        row=2, col=1
                    )
                    
                    # FSD chart
                    fig_metrics.add_trace(
                        go.Bar(x=models, y=metrics_df['Full FSD'], name='FSD', 
                            marker_color=colors[3], showlegend=False),
                        row=2, col=2
                    )
                    
                    fig_metrics.update_layout(
                        height=600,
                        title_text="Model Performance Comparison (Full Dataset)",
                        showlegend=False
                    )
                    
                    st.plotly_chart(fig_metrics, use_container_width=True)
                    
                    # Combined Actual vs Predicted Plot - FIXED VERSION
                    st.markdown("#### 🎯 Combined Actual vs Predicted Comparison")

                    # REMOVED: Independent toggle that was causing the issue
                    # FIXED: Use the main evaluation mode toggle instead
                    st.info(f"📊 Plot scale follows evaluation mode: {'Original Scale' if eval_mode_comparison else 'Log Scale'}")
                    st.caption("💡 Toggle the evaluation mode above to change both metrics and plot scale")

                    fig_pred = go.Figure()

                    # Add scatter plot for each model
                    for i, (model_name, result) in enumerate(results.items()):
                        y_actual = result['y_test_last']
                        y_pred = result['y_pred_last']
                        
                        # FIXED: Use eval_mode_comparison instead of show_ln_scale
                        # Apply transformation based on main evaluation toggle
                        if eval_mode_comparison:
                            # Original scale mode - convert from log if needed
                            if result['is_log_transformed'] or ('ln_' in result['target_name']):
                                y_actual_plot = np.exp(y_actual)
                                y_pred_plot = np.exp(y_pred)
                            else:
                                y_actual_plot = y_actual
                                y_pred_plot = y_pred
                            axis_title = "HPM Values (Original Scale)"
                        else:
                            # Log scale mode - keep in log if available
                            if result['is_log_transformed'] or ('ln_' in result['target_name']):
                                y_actual_plot = y_actual
                                y_pred_plot = y_pred
                            else:
                                # Convert to log scale if data is in original scale
                                y_actual_plot = np.log(y_actual)
                                y_pred_plot = np.log(y_pred)
                            axis_title = "HPM Values (Log Scale)"
                        
                        # FIXED: Use dynamically calculated metrics that match the evaluation mode
                        # Recalculate R² for this specific plot with current evaluation mode
                        plot_r2 = evaluate(y_actual, y_pred, squared=eval_mode_comparison)['R2']
                        
                        fig_pred.add_trace(go.Scatter(
                            x=y_actual_plot,
                            y=y_pred_plot,
                            mode='markers',
                            name=f'{model_name} (R²={plot_r2:.3f})',
                            marker=dict(
                                color=colors[i % len(colors)],
                                size=6,
                                opacity=0.7
                            ),
                            hovertemplate=f'<b>{model_name}</b><br>' +
                                        'Actual: %{x:.0f}<br>' +
                                        'Predicted: %{y:.0f}<br>' +
                                        '<extra></extra>'
                        ))

                    # Add perfect prediction line
                    if results:
                        # FIXED: Calculate min/max based on current evaluation mode
                        all_actual = []
                        all_pred = []
                        
                        for result in results.values():
                            y_actual = result['y_test_last']
                            y_pred = result['y_pred_last']
                            
                            if eval_mode_comparison:
                                # Original scale
                                if result['is_log_transformed'] or ('ln_' in result['target_name']):
                                    all_actual.extend(np.exp(y_actual))
                                    all_pred.extend(np.exp(y_pred))
                                else:
                                    all_actual.extend(y_actual)
                                    all_pred.extend(y_pred)
                            else:
                                # Log scale
                                if result['is_log_transformed'] or ('ln_' in result['target_name']):
                                    all_actual.extend(y_actual)
                                    all_pred.extend(y_pred)
                                else:
                                    all_actual.extend(np.log(y_actual))
                                    all_pred.extend(np.log(y_pred))
                        
                        min_val = min(min(all_actual), min(all_pred))
                        max_val = max(max(all_actual), max(all_pred))
                        
                        fig_pred.add_trace(go.Scatter(
                            x=[min_val, max_val],
                            y=[min_val, max_val],
                            mode='lines',
                            name='Perfect Prediction',
                            line=dict(color='red', dash='dash', width=2),
                            hovertemplate='Perfect Prediction Line<extra></extra>'
                        ))

                    fig_pred.update_layout(
                        title=f'Actual vs Predicted Values - All Models ({("Original Scale" if eval_mode_comparison else "Log Scale")})',
                        xaxis_title=f'Actual {axis_title}',
                        yaxis_title=f'Predicted {axis_title}',
                        height=600,
                        hovermode='closest'
                    )

                    # FIXED: Auto-adjust axis ranges based on current evaluation mode
                    if results:
                        all_x_vals = []
                        all_y_vals = []
                        
                        for result in results.values():
                            y_actual = result['y_test_last']
                            y_pred = result['y_pred_last']
                            
                            if eval_mode_comparison:
                                # Original scale
                                if result['is_log_transformed'] or ('ln_' in result['target_name']):
                                    all_x_vals.extend(np.exp(y_actual))
                                    all_y_vals.extend(np.exp(y_pred))
                                else:
                                    all_x_vals.extend(y_actual)
                                    all_y_vals.extend(y_pred)
                            else:
                                # Log scale
                                if result['is_log_transformed'] or ('ln_' in result['target_name']):
                                    all_x_vals.extend(y_actual)
                                    all_y_vals.extend(y_pred)
                                else:
                                    all_x_vals.extend(np.log(y_actual))
                                    all_y_vals.extend(np.log(y_pred))
                        
                        if all_x_vals and all_y_vals:
                            # Calculate ranges with padding
                            x_min, x_max = min(all_x_vals), max(all_x_vals)
                            y_min, y_max = min(all_y_vals), max(all_y_vals)
                            
                            # Add 5% padding on each side
                            x_range = x_max - x_min
                            y_range = y_max - y_min
                            x_padding = x_range * 0.05
                            y_padding = y_range * 0.05
                            
                            fig_pred.update_xaxes(range=[x_min - x_padding, x_max + x_padding])
                            fig_pred.update_yaxes(range=[y_min - y_padding, y_max + y_padding])

                    st.plotly_chart(fig_pred, use_container_width=True)

                    # FIXED: Add status indicator to confirm the plot is updating
                    st.success(f"Plot updated with {('Original Scale' if eval_mode_comparison else 'Log Scale')} evaluation mode")
                    
                    # Model Rankings (based on Full Dataset metrics)
                    st.markdown("#### 🏆 Model Rankings (Full Dataset)")
                    
                    # Create simplified metrics for ranking
                    ranking_metrics_df = pd.DataFrame({
                        'Model': metrics_df['Model'],
                        'R²': metrics_df['Full R²'],
                        'PE10': metrics_df['Full PE10'],
                        'RT20': metrics_df['Full RT20'],
                        'FSD': metrics_df['Full FSD']
                    })
                    
                    # Calculate rankings for each metric
                    rankings_data = []
                    models = ranking_metrics_df['Model'].tolist()
                    for model_name in models:
                        model_data = ranking_metrics_df[ranking_metrics_df['Model'] == model_name].iloc[0]
                        rankings_data.append({
                            'Model': model_name,
                            'R² Rank': ranking_metrics_df['R²'].rank(ascending=False)[ranking_metrics_df['Model'] == model_name].iloc[0],
                            'PE10 Rank': ranking_metrics_df['PE10'].rank(ascending=False)[ranking_metrics_df['Model'] == model_name].iloc[0],
                            'RT20 Rank': ranking_metrics_df['RT20'].rank(ascending=True)[ranking_metrics_df['Model'] == model_name].iloc[0],  # Lower is better
                            'FSD Rank': ranking_metrics_df['FSD'].rank(ascending=True)[ranking_metrics_df['Model'] == model_name].iloc[0],   # Lower is better
                        })
                    
                    rankings_df = pd.DataFrame(rankings_data)
                    
                    # Calculate average rank
                    rankings_df['Average Rank'] = rankings_df[['R² Rank', 'PE10 Rank', 'RT20 Rank', 'FSD Rank']].mean(axis=1)
                    rankings_df = rankings_df.sort_values('Average Rank')
                    
                    # Add ranking indicators
                    rankings_df['Overall Rank'] = range(1, len(rankings_df) + 1)
                    
                    # Display rankings
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**📊 Detailed Rankings by Metric:**")
                        st.dataframe(rankings_df[['Model', 'R² Rank', 'PE10 Rank', 'RT20 Rank', 'FSD Rank', 'Average Rank']].style.format({
                            'R² Rank': '{:.0f}',
                            'PE10 Rank': '{:.0f}',
                            'RT20 Rank': '{:.0f}',
                            'FSD Rank': '{:.0f}',
                            'Average Rank': '{:.1f}'
                        }), use_container_width=True)
                    
                    with col2:
                        st.markdown("**🏆 Overall Model Ranking:**")
                        for i, row in rankings_df.iterrows():
                            rank = row['Overall Rank']
                            model = row['Model']
                            avg_rank = row['Average Rank']
                            
                            if rank == 1:
                                st.success(f"🥇 **{rank}. {model}** (Avg Rank: {avg_rank:.1f})")
                            elif rank == 2:
                                st.info(f"🥈 **{rank}. {model}** (Avg Rank: {avg_rank:.1f})")
                            elif rank == 3:
                                st.warning(f"🥉 **{rank}. {model}** (Avg Rank: {avg_rank:.1f})")
                            else:
                                st.write(f"**{rank}. {model}** (Avg Rank: {avg_rank:.1f})")
                    
                    # Download comparison results
                    st.markdown("#### 💾 Download Comparison Results")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        # Download metrics comparison
                        st.download_button(
                            label="📊 Download Metrics (.csv)",
                            data=metrics_df.to_csv(index=False),
                            file_name=f"model_metrics_comparison_{selected_session}.csv",
                            mime="text/csv",
                            key=f"download_metrics_{selected_session}",
                            use_container_width=True
                        )
                    
                    with col2:
                        # Download rankings
                        st.download_button(
                            label="🏆 Download Rankings (.csv)",
                            data=rankings_df.to_csv(index=False),
                            file_name=f"model_rankings_{selected_session}.csv",
                            mime="text/csv",
                            key=f"download_rankings_{selected_session}",
                            use_container_width=True
                        )
                    
                    with col3:
                        # Download complete comparison report
                        comparison_report = {
                            'timestamp': selected_session,
                            'session_info': {
                                'target_variable': results[list(results.keys())[0]]['target_name'],
                                'feature_count': len(results[list(results.keys())[0]]['feature_names']),
                                'models_trained': list(results.keys())
                            },
                            'cv_metrics': {model: result['cv_global_test_metrics'] for model, result in results.items()},
                            'full_metrics': {model: result['full_global_test_metrics'] for model, result in results.items()},
                            'rankings': rankings_df.to_dict('records'),
                            'best_model': {
                                'name': rankings_df.iloc[0]['Model'],
                                'average_rank': rankings_df.iloc[0]['Average Rank']
                            }
                        }
                        
                        st.download_button(
                            label="📋 Download Report (.json)",
                            data=json.dumps(comparison_report, indent=2),
                            file_name=f"model_comparison_report_{selected_session}.json",
                            mime="application/json",
                            key=f"download_report_{selected_session}",
                            use_container_width=True
                        )
                    
                    # Clear session option
                    st.markdown("---")
                    if st.button("🗑️ Clear This Training Session", help="Remove this training session from history"):
                        if selected_session in st.session_state.all_model_results:
                            del st.session_state.all_model_results[selected_session]
                            st.success("Training session cleared!")
                            st.rerun()

    else:
        st.warning("⚠️ No data loaded. Please go back to Data Selection.")
        if st.button("← Back to Data Selection"):
            st.session_state.processing_step = 'selection'
            st.rerun()

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
