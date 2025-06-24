from langchain_community.document_loaders import TextLoader
# use chatacter text splitter
from langchain_text_splitters import CharacterTextSplitter


loader = TextLoader("speech.txt")
docs = loader.load()


text_splitter = CharacterTextSplitter(separator="\n\n", chunk_size=100, chunk_overlap=20)
split_docs = text_splitter.split_documents(docs)
print(split_docs)