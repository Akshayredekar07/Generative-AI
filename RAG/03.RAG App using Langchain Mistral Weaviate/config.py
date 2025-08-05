import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    PGVECTOR_URL = os.getenv("DATABASE_URL")

