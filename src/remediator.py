from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from outputs import Outputs
import numpy as np
import pandas as pd

class Remediator:
    def __init__(self):
        self.model = ChatGoogleGenerativeAI(model="gemma-4-31b-it", temperature=0)
    
    def create_report(self, df, file_to_check):
        prompt = (
            "You are a data remediator. Your task is to create a report that summarizes the remediation actions taken on a dataset.\n"
        )
