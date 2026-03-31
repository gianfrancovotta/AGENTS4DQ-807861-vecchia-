import streamlit as st
import json

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
                st.write(stream_text("## Executing: Step 1, Schema Validation"))
                first_message_back_validator = validator.run_validation_check(msg_to_validator, df, file_to_check)
                first_message_back_validator = Outputs(first_message_back_validator).get_text()

                second_message_back_validator = validator.run_naming_check(msg_to_validator, df, file_to_check)
                second_message_back_validator = Outputs(second_message_back_validator).get_text()

                parts = [
                    f"{first_message_back_validator}",
                    f"{second_message_back_validator}"
                ]

                validator_output_message = "\n\n".join(parts)
                st.write_stream(stream_text(validator_output_message))

                st.write("---")
                
                st.write_stream(stream_text("## Executing: Step 2, Completeness Analysis"))
                completeness_analyst.run_completeness_analysis(msg_to_completeness_evaluator, df, file_to_check)
                completeness_output = completeness_analyst.NA_percentages(df)
                overall_missing_percentage = sum([x/len(list(df.columns)) for x in list(completeness_output.values())])
                list_of_droppable_columns = [key for key, v in completeness_output.items() if v > 0.5]
                
                summary_completeness_analyst = completeness_analyst.summarize(
                    completeness_output, overall_missing_percentage, list_of_droppable_columns, file_to_check
                )
                completeness_analyst_output_message = Outputs(summary_completeness_analyst).get_text()

                st.write_stream(stream_text(completeness_analyst_output_message))

                st.write("---")

                st.write_stream(stream_text("## Executing: Step 3, Consistency Validation"))
                pattern_report = get_dataframe_patterns(df)
                pattern_report_str = json.dumps(pattern_report, indent=4)
                df_modified = df.drop(columns=list_of_droppable_columns).dropna(axis=0)

                duplicate_results = consistency_validator.run_duplicate_detection(df, file_to_check)
                format_results = Outputs(consistency_validator.run_format_consistency_check(pattern_report_str, df, file_to_check)).get_text()
                logic_results = Outputs(consistency_validator.run_cross_column_logic(df_modified)).get_text()

                dupe_text = (
                    f"**Exact row duplicates found:** {duplicate_results['exact_duplicates']}\n\n"
                    f"**Key column duplicates:** {duplicate_results['key_column_duplicates']}"
                )

                parts = [
                "### Duplicate Analysis Results:",
                dupe_text.strip(),
                "### Format Consistency Analysis Results:",
                format_results.strip(),
                "*Legend: 'N' represents a number, 'W' represents a word*",
                "### Cross Column Consistency Analysis Results:",
                logic_results.strip()
                ]

                consistency_validator_output = "\n\n".join(parts)

                st.write_stream(stream_text(consistency_validator_output))
                st.write("---")

                st.write_stream(stream_text("## Executing: Step 4, Anomaly detection"))
                outlier_report = anomaly_detector.univariate_outlier_detection(pattern_report_str, df, file_to_check)
                cat_outlier_report = anomaly_detector.categorical_outlier_detection(pattern_report_str, df, file_to_check)

                parts = ["### Univariate Outlier Detection Results:",
                outlier_report.strip(),
                "### Categorical Outlier Detection Results:",
                cat_outlier_report.strip()
                ]

                anomaly_detector_output = "\n\n".join(parts)

                st.write_stream(stream_text(anomaly_detector_output))
                st.write("---")
                
if __name__ == "__main__":
    main()