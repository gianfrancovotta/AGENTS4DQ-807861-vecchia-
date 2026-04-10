import numpy as np
import pandas as pd
from outputs import Outputs
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
 
load_dotenv()
 
class SchemaValidator:
    def __init__(self):
        self.model = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite-preview", temperature=0)
 
    def run_validation_check(self, manager_prompt, df, file_to_check, pattern_report):
        prompt = (
        f"Context: {file_to_check.name}, {manager_prompt}\n"
        f"Current Schema (Column Name: Type): {df.dtypes.to_dict()}\n"
        f"Regex Pattern Report (Dominant shapes in data): {pattern_report}\n\n"
        "TASK:\n"
        "You are an Expert Data Analyst. Compare the Current Schema to the Regex Pattern Report "
        "Base the analysis only on the dominant regex patterns for each column, ignoring the column names.\n"
        "to identify necessary type conversions. Apply these rules strictly in order:\n\n"
        
        "1. **Object to Numeric**: If a column's type is 'object' (dtype 'O') but the regex patterns "
        "show it consists mostly of numbers (e.g., 'N', 'NN.N', 'N.NNN'), flag it to be cast as 'float' or 'int'.\n"
        
        "2. **String to Numeric**: If a column is already typed as 'string' but contains mostly numeric "
        "patterns (e.g., 'N', 'N.N'), flag it to be cast as 'float' or 'int'.\n\n"
        
        "IMPORTANT: Flag all columns that meet these criteria, regardless of current formatting errors. "
        "Python will handle the cleaning; you only need to identify the TARGET type.\n\n"
        
        "OUTPUT RULES:\n"
        "1. If mismatches are found: Begin with exactly: 'Here's a list of columns with type mismatches I found:'\n"
        "   - List each column with a bullet point.\n"
        "   - Provide a brief explanation. Example: '- spesa: Currently an object, but regex shows numeric patterns; should be a float.'\n"
        "2. If all columns correctly match their regex patterns: Return ONLY 'All columns have the expected types they should have.'"
        )
        message = self.model.invoke(prompt)
        return message.content
    
    def run_validation_correction(self, validation_check, df):
        prompt = (
            f"Context: You are a data corection agent. You are given the following validation report that highlights potential issues with the schema of a dataset: {validation_check}\n\n"
            "Task: For each column that is highlighted in the report, populate a json style list with the type that the columns should be cast into." 
            'Only populate the list for the columns that are highlighted in the report, and leave the other columns out of the list. Example of expected output if "price" and "id" were highlighted in the report: { "price": "float", "id": "string" }\n\n'
        )
        message = self.model.invoke(prompt)
        corrections = Outputs(message.content).get_json_obj()  # fixed: was json.loads(Outputs(...)) without .get_text()
        for col, new_type in corrections.items():
            if new_type in ["int", "Int64", "int64","integer"]:
                cleaned_series = df[col].astype(str).str.extract('(\d+)')[0]
                df[col] = pd.to_numeric(cleaned_series, errors="coerce")
                df[col] = df[col].fillna(pd.NA)
                df[col] = df[col].astype("Int64")
            if new_type in ["float", "Float64", "float64"]:
                cleaned_series = df[col].astype(str).str.extract(r'(-?\d+[\d.,]*)')[0]
                df[col] = pd.to_numeric(cleaned_series, errors='coerce')
                df[col] = df[col].fillna(pd.NA)
                df[col] = df[col].astype("float64")
        return df
 
    def run_naming_check(self, manager_prompt, df, file_to_check):
        prompt = (
            f"Context: {file_to_check.name}, {manager_prompt}\n"
            f"Current column names: {df.columns.to_list()}\n\n"
            "Task: Identify columns where the name of the column doesn't follow a known standard naming standard or the general structure of the names of the other columns\n"
            "(e.g., a column using special characters, wrong casing, or reserved words). \n\n"
            "Output Rules:\n"
            "1. If issues found: Begin the report with the phrase:\n"
            "'Here's a list of columns with naming violations I found:'\n" 
            "and then list each column using a bullet point, followed by a brief,\n"
            "natural explanation of the naming violation. Example: '- price: Currently uses uppercase, but should be lowercase.'\n"
            "2. If no issues: Return ONLY 'No columns with naming violations found.'"
        )
        message = self.model.invoke(prompt)
        return message.content
    
    def run_naming_correction(self, naming_check, df):
        prompt = (
            f"Context: You are a data corection agent. You are given the following naming validation report that highlights potential issues with the column names of a dataset: {naming_check}\n\n"
            'Task: For each column that is highlighted in the report, populate a json style dicionary with the corrected name that the column should be renamed to. Only populate the dictionary with the columns that are highlighted in the report, and leave the other columns out of the list. Example of expected output if "Price" and "id" were highlighted in the report: { "Price": "price", "id": "ID" }\n\n'
        )
        message = self.model.invoke(prompt)
        corrections = Outputs(message.content).get_json_obj()
        safe_corrections = {}
        taken_names = df.columns.to_list()
        for old_name, new_name in corrections.items():
            final_new_name = new_name
            while final_new_name in taken_names:
                final_new_name += f"_dup"
            safe_corrections[old_name] = final_new_name
            taken_names.append(final_new_name)
 
        df = df.rename(columns=safe_corrections)
        return df