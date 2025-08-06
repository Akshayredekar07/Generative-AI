
import os
import shutil
from llama_index.core import VectorStoreIndex, StorageContext, load_index_from_storage
from exception import CustomError
import sys

def build_and_persist_index(documents, persist_dir="storage"):
    try:
        if os.path.exists(persist_dir):
            shutil.rmtree(persist_dir)
        index = VectorStoreIndex.from_documents(documents)
        index.storage_context.persist(persist_dir=persist_dir)
    except Exception as e:
        raise CustomError(e, sys)

def load_query_engine(persist_dir="storage"):
    try:
        storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
        index = load_index_from_storage(storage_context)
        query_engine = index.as_query_engine()
        return query_engine
    except Exception as e:
        raise CustomError(e, sys)