import numpy as np
import pandas as pd
from langchain_openrouter import ChatOpenRouter
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

class SchemaValidator:
    def __init__(self):
        self.model = ChatOpenRouter(model="stepfun/step-3.5-flash:free", temperature=0)

    def run_validation_check(self, manager_prompt, df, file_to_check):
        internal_schema_info = {
            col: str(dtype) 
            for col, dtype in zip(df.columns, df.dtypes)
        }
        prompt =(
            f"Context: {file_to_check.name}, {manager_prompt}\n"
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