import numpy as np
import pandas as pd
import re
import time
 
def get_dataframe_patterns(df):
    def get_shape(value):
        if value is pd.NA or pd.isna(value) or value is None:  # fixed: pd.NA must be checked first, before pd.isna()
            return "NULL"
        s = str(value)
        s = re.sub(r'[a-zA-Z]+', 'W', s)
        s = re.sub(r'[0-9]', 'N', s)
        return s
 
    all_column_patterns = {}
    for col in df.columns:
        patterns = df[col].map(get_shape).value_counts().to_dict()
        all_column_patterns[col] = patterns
 
    return all_column_patterns
 
def stream_text(text):
    """Generator to create a typewriter effect for Streamlit."""
    for char in text:
        yield char
        time.sleep(.02)
 
def process_csv(file_path):
    if file_path.name.endswith('.csv'):
        try:
            df = pd.read_csv(file_path)
            return [f"SUCCESS: I've loaded your CSV. Rows: {len(df)}, Columns: {list(df.columns)}", df]
        except Exception as e:
            return [f"FAILURE: Could not load file. Error: {str(e)}", None]
        
    if file_path.name.endswith('.xlsx'):
        try:
            df = pd.read_excel(file_path)
            return [f"SUCCESS: I've loaded your Excel file. Rows: {len(df)}, Columns: {list(df.columns)}", df]
        except Exception as e:
            return [f"FAILURE: Could not load file. Error: {str(e)}", None]
    
    return [f"FAILURE: Unsupported file type. Please upload a .csv or .xlsx file.", None]
 