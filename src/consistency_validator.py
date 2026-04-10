from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from outputs import Outputs
import numpy as np
import pandas as pd

load_dotenv()

class ConsistencyValidator:

    def __init__(self):
        self.model = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite-preview", temperature=0)

    def run_duplicate_detection(self, df, file_to_check):
        exact_dupes = df.duplicated().sum()
        prompt = (
            f"This is the dataset name: {file_to_check.name}\n"
            f"Here are the columns of the dataset: {list(df.columns)}\n"
            "Task: Identify any columns that likely represent a unique identifier for a record\n"
            '(e.g., "ID", "Tax Code", "Email", "SSN", "SKU").\n'
            'Output Rules: Return ONLY a python list of strings containing the exact column names. Example: ["user_id", "email"]\n'
            "If none are found, return an empty list []."
        )
        message = self.model.invoke(prompt)
        key_columns = Outputs(message.content).get_json_obj()

        key_dupes_report = []
        if key_columns:
            for col in key_columns:
                if col in df.columns:
                    dupes = df.duplicated(subset=[col]).sum()
                    if dupes > 0:
                        key_dupes_report.append(f"Column '{col}' has {dupes} duplicate entries.")
        
        df.drop_duplicates(inplace=True)
        return {
            "exact_duplicates": exact_dupes,
            "key_column_duplicates": key_dupes_report if key_dupes_report else "No key duplicates found."
        }

    def run_format_consistency_check(self, pattern, df, file_to_check):        
        prompt = (
            f"This is the dataset name: {file_to_check.name}\n"
            f"Here are the columns of the dataset: {list(df.columns)}\n"
            f"You are given the following regex report on the columns of the dataframe: {pattern}\n\n"
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
        sample_size = min(30, len(df_modified))
        sample_data = df_modified.sample(n=sample_size, random_state=42).to_markdown()

        prompt = (
            f"You are a Senior Data Quality Analyst. I am providing a sample of {sample_size} rows "
            f"from a larger dataset. \n\n"
            f"### DATA SAMPLE:\n{sample_data}\n\n"
            "### TASK:\n"
            "1. **Analyze Relationships**: Based on the column names and the sample, identify logical rules "
            "that *should* exist (e.g., Date order, Mathematical sums, or Redundant columns).\n"
            "2. **Audit the Sample**: Check the provided rows against these rules.\n"
            "3. **Identify Redundancy**: Specifically look for columns that appear to be 'mirrors' of "
            "each other (identical data under different names).\n\n"
            "### OUTPUT FORMAT (Strict Markdown):\n"
            "If issues are found, return a report with these sections:\n"
            "**Inferred Logical Rules:**\n"
            "- [Rule description]\n\n"
            "**Data Redundancy Alerts:**\n"
            "- [Column A] and [Column B] appear to contain identical information.\n\n"
            "If NO logical violations or rules are found, return ONLY: 'No logical violations detected in sample.'"
        )
        message = self.model.invoke(prompt)
        return message.content