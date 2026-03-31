from langchain_openrouter import ChatOpenRouter
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from outputs import Outputs
from functions import process_csv

load_dotenv()

class DataOrchestrator:
    def __init__(self):
        self.model = ChatOpenRouter(model="stepfun/step-3.5-flash:free", temperature=0)

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
            return [final_response, result_of_function[1], file_path]
        else:
            return "I'm just a chatbot to check for data quality! I can't really help with anything else, sorry :("