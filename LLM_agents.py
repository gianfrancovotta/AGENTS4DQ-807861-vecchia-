import numpy as np
import pandas as pd
import json
import time
import re
from dotenv import load_dotenv

from langchain_openrouter import ChatOpenRouter

user_msg = "Hey, can you please convert this file into a dataframe for me?"
msg_to_validator = "The user gave us a .csv that I have already loaded in a dataframe, perform a validation check."
msg_to_completeness_evaluator = "The user gave us a .csv that I have already loaded in a dataframe, perform a data completeness check."
msg_to_consistency_evaluator = "The user gave us a .csv that I have already loaded in a dataframe, perform a data consistency check."

load_dotenv()

def write(text):
    for char in text:
        print(char, end="", flush=True)
        time.sleep(.02)

def process_csv(file_path):
    try:
        df = pd.read_csv(file_path)
        return [f"SUCCESS: Loaded {file_path}. Rows: {len(df)}, Columns: {list(df.columns)}", df]
    except Exception as e:
        return f"FAILURE: Could not load file. Error: {str(e)}"
    
class Outputs:
    def __init__(self, response):
        self.response =  response
    def __str__(self):
        content = self.response
        if isinstance(content, list):
            return content[0].get("text", "")
        return str(content)
    def get_list_out(self):
        content = self.response
        return json.loads(content)
    def print_text(self):
        content = self.response
        if isinstance(content, list):
            content = content[0].get("text", "")
        else:
            content = str(content)
        for char in content:
            print(char, end="", flush=True)
            time.sleep(.02)

class DataOrchestrator:
    def __init__(self):
        self.model = ChatOpenRouter(model="stepfun/step-3.5-flash:free", temperature=0)

    def run_loading(self, user_input, file_path):
        prompt = (
            f"The user says: '{user_input}'. "
            f"They provided a file path: '{file_path}'. "
            "Task: Is this file extension .csv? Answer only 'YES' or 'NO'."
        )
        intent_check = Outputs(self.model.invoke(prompt))
        intent_check = str(intent_check)
        intent_check = intent_check.strip().upper()
        if "YES" in intent_check:
            result_of_function = process_csv(file_path)
            final_prompt = (
                f"The user asked: '{user_input}'. "
                f"The Python system ran a process and returned this result: '{result_of_function[0]}'. "
                "Give a concise summary of this result to the user."
            )
            final_response = self.model.invoke(final_prompt)
            return [final_response.content, result_of_function[1]]
        else:
            return "The file you've given me isn't a .csv. Check if you have uploaded the correct file."

orchestrator = DataOrchestrator()

# Execution
df=None
file_to_check = "spesa.csv"

response_to_user = orchestrator.run_loading(user_msg, file_to_check)
if isinstance(response_to_user, list):
    response = Outputs(response_to_user[0])
    df = response_to_user[1]
    response.print_text()
    print("\n\n")
else:
    response = Outputs(response_to_user)
    response.print_text()

class SchemaValidator:
    def __init__(self):
        self.model=ChatOpenRouter(model="stepfun/step-3.5-flash:free", temperature=0)
    
    def run_validation_check(self, manager_prompt):
        internal_schema_info = {
            col: str(dtype) 
            for col, dtype in zip(df.columns, df.dtypes)
        }
        prompt = (
            f"Context: {manager_prompt}\n"
            f"Current Schema (Column: Type): {internal_schema_info}\n\n"
            f"This is the head of the dataframe, so you can check your assumptions if you are unsure about the type a column should be cast as: {df.head()}"
            "Task: Identify columns where the data type does not match logical expectations "
            "(e.g., a 'Price' column being a string, or 'ID' being a float).\n\n"
            "Output Rules:\n"
            "1. If mismatches are found: List each column using a bullet point, followed by a brief, "
            "natural explanation of why the type is suspicious. Example: '- price: Currently a string, but should be a float.'\n"
            "2. If all match: Return ONLY 'All match'."
        )
        message = self.model.invoke(prompt)
        return message.content
    
    def run_naming_check(self, manager_prompt):
        prompt = (
            f"Context: {manager_prompt}\n"
            f"Current column names: {df.columns.to_list()}\n\n"
            "Task: Identify columns where the name of the column doesn't follow a known standard naming standard or the general structure of the names of the other columns"
            "(e.g., a column using special characters, wrong casing, or reserved words). \n\n"
            "Output Rules:\n"
            "1. If issues found: List each column with a bullet point and a human-friendly explanation of the naming violation.\n"
            "2. If no issues: Return ONLY 'No convention breaks'."
        )
        message = self.model.invoke(prompt)
        return message.content

validator = SchemaValidator()

#Execution
first_message_back_validator = validator.run_validation_check(msg_to_validator)
first_message_back_validator  = Outputs(first_message_back_validator)
second_message_back_validator  = validator.run_naming_check(msg_to_validator)
second_message_back_validator  = Outputs(second_message_back_validator)
first_message_back_validator.print_text()
print("\n")
second_message_back_validator.print_text()
print("\n\n")

class CompletenessAnalyst:
    def __init__(self):
        self.model = ChatOpenRouter(model="stepfun/step-3.5-flash:free", temperature=0)
    
    def run_completeness_analysis(self, manager_prompt):
        prompt = (
            f"Context: {manager_prompt}"
            f"This is the dataframe: {df}"
            "Identify common placeholder strings in the dataframe (like 'null', 'n/a', '-', 'none') "
            "that should be treated as missing values (NaN).\n"
            'Output Rules: Return ONLY a python list in JSON style of strings. Example: ["-", "null", "N/A"]'
        )
        message = self.model.invoke(prompt)
        placeholders = json.loads(str(Outputs(message.content))) #It's messy, basically it gets the content from the output of the LLM, converts it into outputs, then into a string, before converting into a list
        df.replace(placeholders, np.nan, inplace=True)
        return None
    
    def summarize(self, schema, overall, to_drop):
        prompt = (
            "The user gave a .csv that has been analysed for completeness"
            f"This is the schema of the missing value percentages of all the columns: {schema}"
            f"This is the percentage of overall missing values in the entire dataser: {overall}"
            f"This is a list of all the columns where the percentage of missing values is above 0.5: {to_drop}"
            "Task: Summarise the findings of the analysis in a human readable way. Divide it in three sections"
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
    
completeness_analyst = CompletenessAnalyst()

#Execution
completeness_analyst.run_completeness_analysis(msg_to_completeness_evaluator)
completeness_output = completeness_analyst.NA_percentages()
overall_missing_percentage = sum([x/len(list(df.columns)) for x in list(completeness_output.values())])
list_of_droppable_columns = [key for key, v in completeness_output.items() if v > 0.5]
summary_completeness_analyst = completeness_analyst.summarize(completeness_output, overall_missing_percentage, list_of_droppable_columns)
summary_completeness_analyst = Outputs(summary_completeness_analyst)
summary_completeness_analyst.print_text()
print("\n\n")

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

class ConsistencyValidator:
    def __init__(self):
        self.model = ChatOpenRouter(model="stepfun/step-3.5-flash:free", temperature=0)

    def run_duplicate_detection(self):
        exact_dupes = df.duplicated().sum()
        prompt = (
            f"Here are the columns of a dataset: {list(df.columns)}\n"
            "Task: Identify any columns that likely represent a unique identifier for a record "
            '(e.g., "ID", "Tax Code", "Email", "SSN", "SKU").\n'
            'Output Rules: Return ONLY a python list of strings containing the exact column names. Example: ["user_id", "email"]'
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
            f"You are given the following regex report on the columns of a dataframe: {pattern_report}\n\n"
            "TASK: Flag a column for inconsistent patterns and list how many pattern violations are there in that column compared to the dominant pattern."
            "Anomalous patterns are defined as patterns that appear in less than 0.05 percent of the rows compared to a dominant pattern." 
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

    def run_cross_column_logic(self):
        prompt = (
            f"Context: This is the a sample of the dataframe you are working on: {df_modified.sample(frac=0.1)}"
            "Task: Based on the column names and sample data, identify any logical cross-column relationships"
            "Ignore: Ignore occurences where the data is shown in a temporary filled in value (e.g. /, //, --)"
            "(e.g., 'End_Date' should be after 'Start_Date', 'Age' should match 'Birth_Year', 'Total' should equal 'Price' * 'Quantity')."
            "Make sure to also list columns that seem to always show the same data recorded in both of them (e.g., two columns called 'ABC' and 'XYZ' seem to always have the same data in both of them in your sample)."
            "Then, check if any of the provided sample rows violate these inferred rules.\n\n"
            "Output Rules:\n"
            "1. If you find a logical violation in the sample: Explain the rule and point out the presence of logical conflicts\n"
            "2. If no obvious logical rules apply, or no violations are found in the sample: Return ONLY 'No logical violations detected in sample'."
        )
        message = self.model.invoke(prompt)
        return message.content

consistency_validator = ConsistencyValidator()
pattern_report = get_dataframe_patterns(df)
pattern_report = json.dumps(pattern_report, indent=4)

df_modified = df.drop(columns=list_of_droppable_columns)
df_modified = df_modified.dropna(axis=0)

#Execution
write("Duplicate analysis results:\n\n")
duplicate_results = consistency_validator.run_duplicate_detection()
write(f"Exact row duplicates found: {duplicate_results['exact_duplicates']}\n")
write(f"Key column duplicates: {duplicate_results['key_column_duplicates']}\n")
print("\n")
write("Fromat consistency analysis results:\n\n")
format_results = consistency_validator.run_format_consistency_check(pattern_report)
format_results = Outputs(format_results)
format_results.print_text()
print("\n")
write("Legend: 'N' represents a number (or group of numbers), 'W' represents a word (or cluster of words)\n")
print("\n")
write("Cross column consistency analysis results:\n\n")
logic_results = consistency_validator.run_cross_column_logic()
logic_results = Outputs(logic_results)
logic_results.print_text()
print("\n\n")

class AnomalyDetector:
    def __init__(self):
        self.model = ChatOpenRouter(model="stepfun/step-3.5-flash:free", temperature = 0)

    def univariate_outlier_detection(self):
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
        if len(univariate_columns)==0:
            write("No columns were found as candidates for univariate outliers detection")
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
                    write(f"Column '{col}': No outliers detected.\n\n")
                else:
                    write(f"Column '{col}': Found {len(outlier_data)} outliers.\n")
                    write(f"Details: {outlier_data.to_dict()}\n\n")
            write("Remember that this analysis is done only on values that could be converted into a number. Refer to the report for the list of rows and values that cannot be cast into a number.\n")
            write("All the values that couldn't be cast have been ignored.")
    
    def categorical_outlier_detection(self):
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
        if len(categorical_columns) == 0:
            write("No columns were found as candidates for categorical outliers detection\n")
        else:
            for col in categorical_columns:
                series = df[col].astype(str)
                if series.empty:
                    continue
                series_counts = series.value_counts()
                outlier_mask = (series_counts < len(series)*0.01)
                outlier_data = series_counts[outlier_mask]
                if len(outlier_data) == 0:
                    write(f"Column '{col}': No outliers detected.\n\n")
                else:
                    write(f"Column '{col}': Found {len(outlier_data)} outliers.\n")
                    write(f"Details: {outlier_data.to_dict()}\n\n")
                
anomaly_detector = AnomalyDetector()

#Execution
write("Results of the univariate outlier detection:\n\n")
anomaly_detector.univariate_outlier_detection()
print("\n\n")
write("Results of the categorical outlier detection:\n\n")
anomaly_detector.categorical_outlier_detection()