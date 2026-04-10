import re
import json

class Outputs:
    def __init__(self, response):
        self.response = response
        
    def __str__(self):
        return self.get_text()
        
    def get_list_out(self):
        return json.loads(self.get_text())
        
    def get_text(self):
        content = self.response
        text = ""
        
        # 1. Handle the new list structure from thinking models
        if isinstance(content, list):
            for block in content:
                # Find the actual text output, ignore the 'thinking' blocks
                if isinstance(block, dict) and block.get("type") == "text":
                    text = block.get("text", "")
                    break
            
            # Fallback just in case it's a list but doesn't have a 'text' block
            if not text and len(content) > 0:
                text = str(content[0])
        else:
            # Handle standard string responses
            text = str(content)
        
        # 2. Strip out Markdown formatting (like ```json or ```)
        # Using a more robust regex to catch backticks anywhere
        text = re.sub(r'```[a-zA-Z]*\n?', '', text)
        text = re.sub(r'\n?```', '', text)
        
        return text.strip()

    def get_json_obj(self):
        """Dedicated method to extract JSON dictionaries or lists."""
        text = self.get_text()
        
        # Look for the first [ or { and the last ] or }
        match = re.search(r'(\[.*\]|\{.*\})', text, re.DOTALL)
        if match:
            text = match.group(1)
            
        return json.loads(text)