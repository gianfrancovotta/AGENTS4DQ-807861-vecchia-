from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from outputs import Outputs
import numpy as np
import pandas as pd

load_dotenv()

class CompletenessAnalyst:
    def __init__(self):
        self.model = ChatGoogleGenerativeAI(model="gemma-4-31b-it", temperature=0)

    def run_completeness_analysis(self, manager_prompt, df, file_to_check):
        prompt = (
            "You are a data completeness analyst. Your task is to analyze the completeness of a dataset and provide insights on the missing values."
            "You are highly competent in your secotr and are an expert in identifying common placeholder strings that are often used to represent missing values in datasets."
            f"Context: {file_to_check.name}, {manager_prompt}\n"
            f"This is the list of the columns in the dataframe: {list(df.columns)}\n"
            f"This is the head of the dataframe: {df.head().to_dict()}\n"
            "Identify common placeholder strings in the dataframe (like 'null', 'n/a', '-', 'none'), both in english and the inferred language of the dataset\n"
            "that should be treated as missing values (NaN).\n"
            'Output Rules: Return ONLY a python list of strings. Example: ["-", "null", "N/A", "missing", "none", "n/a", "unknown", "//", "/"]'
        )
        message = self.model.invoke(prompt)
        placeholders = Outputs(message.content).get_json_obj()
        for col in df.columns:
            if df[col].dtype == "string":
                df[col] = df[col].replace(placeholders, pd.NA)
        return df
    
    def summarize_columns(self, schema, overall, to_drop, file_to_check):
        prompt = (
            f"Context: {file_to_check.name}"
            "The user gave a .csv that has been analysed for completeness\n"
            f"This is the schema of the missing value percentages of all the columns: {schema}\n"
            f"This is the percentage of overall missing values in the entire dataser: {overall}\n"
            f"This is a list of all the columns where the percentage of missing values is above 50%: {to_drop}\n"
            "Task: Summarise the findings of the analysis in a human readable way. Divide it in three sections:\n"
            "1: Overall Completeness Summary: Briefly explain the overall completeness of the dataset based on the overall missing percentage. Use bullet points.\n"
            "2: Column Completeness Highlights: Create a table showing the missing percentage of each column. List critical insights as a bullet point after the table.\n"
            "3: Flag columns for removal: Based on the list of columns with more than 50% missing values, create a bullet point list of the columns that should be removed, along with a brief explanation of why (e.g. 'Column XYZ has 78% missing values, which is a strong indicator that it may not be useful for analysis and could potentially introduce bias if retained.'). If no columns are above the threshold, state that no columns need to be removed."
        )
        message = self.model.invoke(prompt)
        return message.content
    
    def summarize_rows(self, schema, to_drop, file_to_check):
        prompt = (
            f"Context: {file_to_check.name}"
            "The user gave a .csv that has been analysed for completeness\n"
            f"This is the schema of the missing value percentages of all the rows: {schema}\n"
            f"This is a list of all the rows where the percentage of missing values is above 50%, after dropping the columns with more than 50% of missing values: {to_drop}\n"
            f"This is the percentage of rows that have more than 50% of missing values: {len(to_drop)/len(schema):.2%}\n"
            "Task: Summarise the findings of the analysis in a human readable way. Keep it in a singular section.\n"
            "This is the format to follow for the output, make sure to number the section as number 4:\n"
            "4: Row completeness summary: Give a brief summary of the completeness analysis of the rows in the dataset, including the percentage of rows that have more than 50% missing values and list the rows that have more than 50% missing values.\n"
        )
        message = self.model.invoke(prompt)
        return message.content

    @staticmethod
    def NA_percentages_columns(df):
        completeness_schema = {
            col: value 
            for col, value in zip(list(df.columns), [x/len(df) for x in list(df.isnull().sum())])
        }
        return completeness_schema
    
    @staticmethod
    def NA_percentages_rows(df, list_of_droppable_columns):
        completeness_schema = {
            idx: value 
            for idx, value in zip(list(df.index), [x/len(df.drop(columns=list_of_droppable_columns).columns) for x in list(df.drop(columns=list_of_droppable_columns).isnull().sum(axis=1))])
        }
        return completeness_schema
