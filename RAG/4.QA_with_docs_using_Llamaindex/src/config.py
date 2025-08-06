import os
from dotenv import load_dotenv
from llama_index.llms.gemini import Gemini
from llama_index.embeddings.gemini import GeminiEmbedding
from exception import CustomError
import sys

def load_env():
    try:
        load_dotenv()
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not found in .env file")
        os.environ["GOOGLE_API_KEY"] = api_key
    except Exception as e:
        raise CustomError(e, sys)

def get_llm():
    try:
        return Gemini(models='gemini-pro')
    except Exception as e:
        raise CustomError(e, sys)

def get_embedding_model():
    try:
        return GeminiEmbedding(model_name="models/embedding-001")
    except Exception as e:
        raise CustomError(e, sys)