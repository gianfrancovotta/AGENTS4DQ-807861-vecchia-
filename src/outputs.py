import re
import json

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
            text = content[0].get("text", "")
        else:
            text = str(content)
        
        text = re.sub(r'^```[a-zA-Z]*\n?', '', text)
        text = re.sub(r'\n?```$', '', text)
        
        return text.strip()