import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

pdf_path = 'attention.pdf'
if not os.path.isfile(pdf_path):
    raise FileNotFoundError(f"PDF file '{pdf_path}' not found.")

# load the pdf document
loader = PyPDFLoader(pdf_path)
docs = loader.load()

# Load the text data
from langchain_community.document_loaders import TextLoader

loader = TextLoader("speech.txt")
txt = loader.load()




text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
texts = [doc.page_content for doc in docs]


# final_document = text_splitter.create_documents(texts)
final_document = text_splitter.split_documents(docs)
# print(final_document)
# print(final_document[0])
# print()
# print(final_document[1])


speech=""
with open('speech.txt', 'r') as f:
    speech = f.read()

split = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
document = split.create_documents([speech])

print(document[0])
print(document[1])


print(type(document[0]))
