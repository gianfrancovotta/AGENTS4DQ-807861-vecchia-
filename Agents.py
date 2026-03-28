import numpy as np
import pandas as pd
import streamlit as st
import json
import time
import re
from dotenv import load_dotenv

from langchain_openrouter import ChatOpenRouter
from langchain_google_genai import ChatGoogleGenerativeAI


load_dotenv()

msg_to_validator = "The user gave us a .csv that I have already loaded in a dataframe, perform a validation check."
msg_to_completeness_evaluator = "The user gave us a .csv that I have already loaded in a dataframe, perform a data completeness check."
msg_to_consistency_evaluator = "The user gave us a .csv that I have already loaded in a dataframe, perform a data consistency check."

def get_dataframe_patterns(df):
    def get_shape(value):
        if pd.isna(value):
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
    try:
        df = pd.read_csv(file_path)
        return [f"SUCCESS: I've loaded your CSV. Rows: {len(df)}, Columns: {list(df.columns)}", df]
    except Exception as e:
        return f"FAILURE: Could not load file. Error: {str(e)}"
    
class Outputs:
    def __init__(self, response):
        self.response = response
        
    def __str__(self):
        content = self.response
        if isinstance(content, list):
            return content[0].get("text", "")
        return str(content)
        
    def get_list_out(self):
        content = self.response
        return json.loads(content)
        
    def get_text(self):
        content = self.response
        if isinstance(content, list):
            return content[0].get("text", "")
        return str(content)

class DataOrchestrator:
    def __init__(self):
        self.model = ChatOpenRouter(model="nvidia/nemotron-3-super-120b-a12b:free", temperature=0)

    def run_loading(self, user_input, file_path):
        prompt = (
            f"The user says: '{user_input}'. "
            f"They provided a file path: '{file_path}'. "
            "Task: Does the user want me to load and check this file? Answer only 'YES' or 'NO'."
        )
        intent_check = Outputs(self.model.invoke(prompt))
        intent_check = str(intent_check).strip().upper()
        
        if "YES" in intent_check and file_path:
            result_of_function = process_csv(file_path)
            final_response = result_of_function[0]
            return [final_response, result_of_function[1]]
        else:
            return "I'm just a chatbot to check for data quality! I can't really help with anything else, sorry :("

class SchemaValidator:
    def __init__(self):
        self.model = ChatOpenRouter(model="nvidia/nemotron-3-super-120b-a12b:free", temperature=0)
    
    def run_validation_check(self, manager_prompt):
        internal_schema_info = {
            col: str(dtype) 
            for col, dtype in zip(df.columns, df.dtypes)
        }
        prompt =(
            f"Context: {manager_prompt}\n"
            f"Current Schema (Column: Type): {internal_schema_info}\n\n"
            f"This is the head of the dataframe, so you can check your assumptions if you are unsure about the type a column should be cast as: {df.head()}\n"
            "Task: Identify columns where the data type does not match logical expectations based on the column name and the head of the dataframe.\n"
            "(e.g., a 'Price' column being a string, or 'ID' being a float).\n\n"
            "Output Rules:\n"
            "1. If mismatches are found: Begin the report with the phrase:\n"
            "'Here's a list of columns with type mismatches I found:'\n"
            "and then list each column using a bullet point, followed by a brief,\n"
            "natural explanation of why the type is suspicious. Example: '- price: Currently a string, but should be a float.'\n"
            "2. If all match: Return ONLY:\n"
            " 'All columns have the expected types they should have.'"
        )
        message = self.model.invoke(prompt)
        return message.content
    
    def run_naming_check(self, manager_prompt):
        prompt = (
            f"Context: {manager_prompt}\n"
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

class CompletenessAnalyst:
    def __init__(self):
        self.model = ChatOpenRouter(model="nvidia/nemotron-3-super-120b-a12b:free", temperature=0)
    
    def run_completeness_analysis(self, manager_prompt):
        prompt = (
            f"Context: {manager_prompt}\n"
            f"This is the dataframe: {df}\n"
            "Identify common placeholder strings in the dataframe (like 'null', 'n/a', '-', 'none')\n"
            "that should be treated as missing values (NaN).\n"
            'Output Rules: Return ONLY a python list in JSON style of strings. Example: ["-", "null", "N/A"]'
        )
        message = self.model.invoke(prompt)
        placeholders = json.loads(str(Outputs(message.content))) 
        df.replace(placeholders, np.nan, inplace=True)
        return None
    
    def summarize(self, schema, overall, to_drop):
        prompt = (
            "The user gave a .csv that has been analysed for completeness\n"
            f"This is the schema of the missing value percentages of all the columns: {schema}\n"
            f"This is the percentage of overall missing values in the entire dataser: {overall}\n"
            f"This is a list of all the columns where the percentage of missing values is above 0.5: {to_drop}\n"
            "Task: Summarise the findings of the analysis in a human readable way. Divide it in three sections:\n"
            "1: Overall Completeness Summary: Briefly explain the overall completeness of the dataset based on the overall missing percentage. Use bullet points.\n"
            "2: Column Completeness Highlights: Create a table showing the missing percentage of each column. List critical insights as a bullet point after the table.\n"
            "3: Flag columns for removal: Based on the list of columns with more than 50% missing values, create a bullet point list of the columns that should be removed, along with a brief explanation of why (e.g. 'Column XYZ has 78% missing values, which is a strong indicator that it may not be useful for analysis and could potentially introduce bias if retained.'). If no columns are above the threshold, state that no columns need to be removed."
        )
        message = self.model.invoke(prompt)
        return message.content

    @staticmethod
    def NA_percentages():
        completeness_schema = {
            col: value 
            for col, value in zip(list(df.columns), [x/len(df) for x in list(df.isnull().sum())])
        }
        return completeness_schema

class ConsistencyValidator:

    def __init__(self):
        self.model = ChatGoogleGenerativeAI(model="gemini-3-flash-preview", temperature=0)

    def run_duplicate_detection(self):
        exact_dupes = df.duplicated().sum()
        prompt = (
            f"Here are the columns of a dataset: {list(df.columns)}\n"
            "Task: Identify any columns that likely represent a unique identifier for a record\n"
            '(e.g., "ID", "Tax Code", "Email", "SSN", "SKU").\n'
            'Output Rules: Return ONLY a python list of strings containing the exact column names. Example: ["user_id", "email"]\n'
            "If none are found, return an empty list []."
        )
        message = self.model.invoke(prompt)
        
        try:
            key_columns = json.loads(str(Outputs(message.content)))
        except json.JSONDecodeError:
            key_columns = []

        key_dupes_report = []
        if key_columns:
            for col in key_columns:
                if col in df.columns:
                    dupes = df.duplicated(subset=[col]).sum()
                    if dupes > 0:
                        key_dupes_report.append(f"Column '{col}' has {dupes} duplicate entries.")
        
        return {
            "exact_duplicates": exact_dupes,
            "key_column_duplicates": key_dupes_report if key_dupes_report else "No key duplicates found."
        }

    def run_format_consistency_check(self, pattern):        
        prompt = (
            f"You are given the following regex report on the columns of a dataframe: {pattern}\n\n"
            "TASK: Flag a column for inconsistent patterns and list how many pattern violations are there in that column compared to the dominant pattern.\n"
            "Anomalous patterns are defined as patterns that appear in less than 0.05 percent of the rows compared to a dominant pattern.\n" 
            "This is just a general guideline and should be ignored in favor of following the rules if a contradiction is found.\n\n"
            "RULES for flagging:\n"
            "- Do NOT flag a column if it has a high variety of patterns (e.g., a 'Description' column).\n"
            "- ONLY flag a column if there is a clear 'Standard Format' being broken by a few outliers.\n"
            "- Example: If 'Date' has 990 'N-N-N' and 10 'N/N/N', flag 'N/N/N' as the inconsistency.\n\n"
            "Output Rules:\n"
            "1. If format inconsistencies are found: List the column name and briefly explain the mixed formats.\n"
            "2. If formatting looks consistent across all columns: Return ONLY 'Formats are consistent'."
        )
        message = self.model.invoke(prompt)
        return message.content

    def run_cross_column_logic(self, df_modified):
        prompt = (
            f"Context: This is the a sample of the dataframe you are working on: {df_modified.sample(frac=0.1)}\n"
            "Task: Based on the column names and sample data, identify any logical cross-column relationships\n"
            "Ignore: Ignore rows in the sample\n"
            "(e.g., 'End_Date' should be after 'Start_Date', 'Age' should match 'Birth_Year', 'Total' should equal 'Price' * 'Quantity').\n"
            "Make sure to also list columns that seem to always show the same data recorded in both of them (e.g., two columns called 'ABC' and 'XYZ' seem to always have the same data in both of them in your sample).\n"
            "Then, check if any of the provided sample rows violate these inferred rules.\n\n"
            "Output Rules:\n"
            f"1. First, tell the user that you sampled 10% of the data,\n" 
            "then list the logical rules you inferred based on the column names and sample data. For example:\n"
            "   - 'End_Date' should be after 'Start_Date'\n"
            "   - 'Age' should match 'Birth_Year'\n"
            "   - 'Total' should equal 'Price' * 'Quantity'\n"
            "   - 'ABC' and 'XYZ' seem to always have the same data recorded in both of them\n"
            "2. Then, for each rule, check if any of the sample rows violate the rule.\n"
            "If violations are found, list the rule and the percentage of rows with that violation in the sample. For example:\n"
            f"   - Rule: 'End_Date' should be after 'Start_Date' - Violations found in 2% of sample rows.\n"
            "If no obvious logical rules apply, or no violations are found in the sample: Return ONLY 'No logical violations detected in sample'."
        )
        message = self.model.invoke(prompt)
        return message.content
    
class AnomalyDetector:
    def __init__(self):
        self.model = ChatOpenRouter(model="nvidia/nemotron-3-super-120b-a12b:free", temperature = 0)

    def univariate_outlier_detection(self, pattern_report):
        prompt = (
            f"This is a sample of the dataframe you are working on: {df.head()}"
            f"This is the regex pattern report of all the columns of the dataframe: {pattern_report}"
            "Task: Identify candidate columns for univariate outlier detection.\n"
            "A column is a candidate for univariate outlier detection if both of these conditions are true."
            "1. The pattern report of a column shows that most of the entires are made up of numbers (e.g. the vast majority are in the pattern NN.NNN, N.N, NN.N)"
            "2. It makes sense to check for numerical outliers in that column using standard deviations (e.g. it would make sense to check for outliers in a column called 'expsenses', but not in a column called 'id')"
            'Output: Return ONLY a python list in JSON style of strings. Example: ["rate", "spesa"]'
            )
        message = self.model.invoke(prompt)
        univariate_columns = json.loads(str(Outputs(message.content)))
        
        report = ""
        if len(univariate_columns) == 0:
            report += "No columns were found as candidates for univariate outliers detection.\n"
        else:
            for col in univariate_columns:
                series = pd.to_numeric(df[col], errors='coerce').dropna()
                if series.empty:
                    continue
                mean_val = series.mean()
                std_val = series.std()
                upper_limit = mean_val + (3 * std_val)
                lower_limit = mean_val - (3 * std_val)
                outlier_mask = (series > upper_limit) | (series < lower_limit)
                outlier_data = series[outlier_mask]
                if len(outlier_data) == 0:
                    report += f"Column '{col}': No outliers detected.\n\n"
                else:
                    report += f"Column '{col}': Found {len(outlier_data)} outliers.\n"
                    report += f"Details: {outlier_data.to_dict()}\n\n"
            
            report += "Remember that this analysis is done only on values that could be converted into a number. Refer to the report for the list of rows and values that cannot be cast into a number.\n"
            report += "All the values that couldn't be cast have been ignored."
            
        return report
    
    def categorical_outlier_detection(self, pattern_report):
        prompt = (
            f"This is a sample of the dataframe you are working on: {df.head()}"
            f"This is the regex pattern report of all the columns of the dataframe: {pattern_report}"
            "Task: Identify candidate columns for categorical outlier detection.\n"
            "A column is a candidate for categorical outlier detection if both of these conditions are true."
            "1. The pattern report of a column shows that most of the entires are made up of words (e.g. the vast majority are in the pattern W W, W, W W W)"
            "2. It makes sense to check for categorical outliers in that column (e.g. it would make sense to check for outliers in a column called 'job title' or 'education level', but not in a column called 'notes' or 'client names')"
            'Output: Return ONLY a python list in JSON style of strings. Example: ["job title", "level"]'
            )
        message = self.model.invoke(prompt)
        categorical_columns = json.loads(str(Outputs(message.content)))
        
        report = ""
        if len(categorical_columns) == 0:
            report += "No columns were found as candidates for categorical outliers detection.\n"
        else:
            for col in categorical_columns:
                series = df[col].astype(str)
                if series.empty:
                    continue
                series_counts = series.value_counts()
                outlier_mask = (series_counts < len(series)*0.01)
                outlier_data = series_counts[outlier_mask]
                if len(outlier_data) == 0:
                    report += f"Column '{col}': No outliers detected.\n\n"
                else:
                    report += f"Column '{col}': Found {len(outlier_data)} outliers.\n"
                    report += f"Details: {outlier_data.to_dict()}\n\n"
        return report

orchestrator = DataOrchestrator()
validator = SchemaValidator()
completeness_analyst = CompletenessAnalyst()
consistency_validator = ConsistencyValidator()
anomaly_detector = AnomalyDetector()

df = None
file_to_check = st.file_uploader("Upload a CSV to begin")

with st.form("my_form"):
    text = st.text_area(
        "Enter text:",
        "Can you help me clean up this dataset?",
    )
    submitted = st.form_submit_button("Submit")
    
    if submitted and file_to_check:
        response_to_user = orchestrator.run_loading(text, file_to_check)
        if isinstance(response_to_user, list):
            response = Outputs(response_to_user[0])
            df = response_to_user[1]
            st.write_stream(stream_text(response.get_text()))
            st.write("---")
        else:
            response = Outputs(response_to_user)
            st.write(stream_text(response.get_text()))
            
        if df is not None:
            st.write(stream_text("## Executing: Step 1, Schema Validation"))
            first_message_back_validator = validator.run_validation_check(msg_to_validator)
            first_message_back_validator = Outputs(first_message_back_validator).get_text()

            second_message_back_validator = validator.run_naming_check(msg_to_validator)
            second_message_back_validator = Outputs(second_message_back_validator).get_text()

            validator_output_message = (f"{first_message_back_validator}"+
            "\n\n"+
            f"{second_message_back_validator}")

            st.write_stream(stream_text(validator_output_message))

            st.write("---")
            
            st.write_stream(stream_text("## Executing: Step 2, Completeness Analysis"))
            completeness_analyst.run_completeness_analysis(msg_to_completeness_evaluator)
            completeness_output = completeness_analyst.NA_percentages()
            overall_missing_percentage = sum([x/len(list(df.columns)) for x in list(completeness_output.values())])
            list_of_droppable_columns = [key for key, v in completeness_output.items() if v > 0.5]
            
            summary_completeness_analyst = completeness_analyst.summarize(
                completeness_output, overall_missing_percentage, list_of_droppable_columns
            )
            completeness_analyst_output_message = Outputs(summary_completeness_analyst).get_text()

            st.write_stream(stream_text(completeness_analyst_output_message))

            st.write("---")

            st.write_stream(stream_text("## Executing: Step 3, Consistency Validation"))
            pattern_report = get_dataframe_patterns(df)
            pattern_report_str = json.dumps(pattern_report, indent=4)
            df_modified = df.drop(columns=list_of_droppable_columns).dropna(axis=0)

            duplicate_results = consistency_validator.run_duplicate_detection()
            format_results = Outputs(consistency_validator.run_format_consistency_check(pattern_report_str)).get_text()
            logic_results = Outputs(consistency_validator.run_cross_column_logic(df_modified)).get_text()

            dupe_text = (
                f"**Exact row duplicates found:** {duplicate_results['exact_duplicates']}\n\n"
                f"**Key column duplicates:** {duplicate_results['key_column_duplicates']}"
            )

            consistency_validator_output = ("**Duplicate Analysis Results:**"+
            "\n\n"+
            f"{dupe_text}"+
            "\n\n"+
            "**Format Consistency Analysis Results**"+
            "\n\n"+
            f"{format_results}"+
            "\n\n"+
            "*Legend: 'N' represents a number (or group of numbers), 'W' represents a word (or cluster of words)*"+
            "\n\n"+
            "**Cross Column Consistency Analysis Results**"+
            "\n\n"+
            f"{logic_results}"
            )

            st.write_stream(stream_text(consistency_validator_output))
            st.write("---")

            st.write_stream(stream_text("### Executing: Step 4, Anomaly detection"))
            outlier_report = anomaly_detector.univariate_outlier_detection(pattern_report_str)
            cat_outlier_report = anomaly_detector.categorical_outlier_detection(pattern_report_str)

            anomaly_detector_output = ("**Univariate Outlier Detection Results**:"
            +"\n\n"+
            f"{outlier_report}"+
            "\n\n"+
            "**Categorical Outlier Detection Results**:"+
            "\n\n"+
            f"{cat_outlier_report}"
            )

            st.write_stream(stream_text(anomaly_detector_output))
            st.write("---")