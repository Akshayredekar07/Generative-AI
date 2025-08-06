
import os
from llama_index.core import SimpleDirectoryReader
from exception import CustomError
import sys

def load_data(uploaded_file):
    try:
        temp_file_path = "temp_uploaded_file.txt"
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        loader = SimpleDirectoryReader(input_files=[temp_file_path])
        documents = loader.load_data()
        os.remove(temp_file_path)
        return documents
    except Exception as e:
        raise CustomError(e, sys)