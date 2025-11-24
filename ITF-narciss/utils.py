# utils.py

import json
from typing import Dict

def read_json(filepath: str) -> Dict:
    """Đọc và phân tích cú pháp một tệp JSON."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Lỗi: Không tìm thấy tệp {filepath}")
        return {}
    
'''from google import genai
from google.genai import types
import time

client = genai.Client(api_key='AIzaSyCtet6vrHgxb3Z5UQnW_vcVvP3YPc6bGr4')

# Upload the file using the Files API, supply a file name which will be visible in citations
sample_file = client.files.upload(file='/content/NarcissHuth_FeedbackDesign_Erf2004.pdf', config={'name': 'itf-theory'})

# Create the File Search store with an optional display name
file_search_store = client.file_search_stores.create(config={'display_name': 'test'})

# Import the file into the File Search store
operation = client.file_search_stores.import_file(
    file_search_store_name=file_search_store.name,
    file_name=sample_file.name
)

# Wait until import is complete
while not operation.done:
    time.sleep(5)
    operation = client.operations.get(operation)'''