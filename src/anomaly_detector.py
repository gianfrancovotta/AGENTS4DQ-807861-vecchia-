from langchain_openrouter import ChatOpenRouter
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from outputs import Outputs
import numpy as np
import pandas as pd
import json

load_dotenv

class AnomalyDetector:
    def __init__(self):
        self.model = ChatOpenRouter(model="stepfun/step-3.5-flash:free", temperature=0)

    def univariate_outlier_detection(self, pattern_report, df, file_to_check):
        prompt = (
            f"Context: {file_to_check.name}\n"
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
    
    def categorical_outlier_detection(self, pattern_report, df, file_to_check):
        prompt = (
            f"Context: {file_to_check.name}\n"
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