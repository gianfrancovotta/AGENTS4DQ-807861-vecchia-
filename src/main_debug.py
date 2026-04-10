from email.mime import message

import streamlit as st
import json
import pandas as pd

from data_orchestrator import DataOrchestrator
from schema_validator import SchemaValidator
from completeness_analyst import CompletenessAnalyst
from consistency_validator import ConsistencyValidator
from anomaly_detector import AnomalyDetector

from outputs import Outputs
from functions import get_dataframe_patterns, stream_text

from langchain_core.globals import set_llm_cache
from langchain_community.cache import SQLiteCache

CACHE_DB_PATH = "presentation_cache.db"
set_llm_cache(SQLiteCache(database_path=CACHE_DB_PATH))

msg_to_validator = "The user gave us a .csv that I have already loaded in a dataframe, perform a validation check."
msg_to_completeness_evaluator = "The user gave us a .csv that I have already loaded in a dataframe, perform a data completeness check."
msg_to_consistency_evaluator = "The user gave us a .csv that I have already loaded in a dataframe, perform a data consistency check."

orchestrator = DataOrchestrator()
validator = SchemaValidator()
completeness_analyst = CompletenessAnalyst()
consistency_validator = ConsistencyValidator()
anomaly_detector = AnomalyDetector()

def main():
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
                pattern_report_1= get_dataframe_patterns(df)
                pattern_report_str = json.dumps(pattern_report_1, indent=4)
                st.write(stream_text("## Executing: Step 1, Schema Validation"))
                first_message_back_validator = validator.run_validation_check(msg_to_validator, df, file_to_check, pattern_report_str)
                first_message_back_validator = Outputs(first_message_back_validator).get_text()
                st.write_stream(stream_text(first_message_back_validator))
                df = validator.run_validation_correction(first_message_back_validator, df)
                st.write(df.dtypes)
if __name__ == "__main__":
    main()