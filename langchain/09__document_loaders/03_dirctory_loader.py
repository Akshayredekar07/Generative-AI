# from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
# from pathlib import Path

# # Manually collect and load PDFs using PyPDFLoader
# pdf_files = list(Path('notes').glob("*.pdf"))
# docs = []

# for pdf_file in pdf_files:
#     loader = PyPDFLoader(str(pdf_file))
#     docs.extend(loader.load())

# print(len(docs))


from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader

loader = DirectoryLoader(
    path='notes',
    glob='*.pdf',
    loader_cls=PyPDFLoader  # Force use of PyPDFLoader
)

# docs = loader.load()
docs = loader.lazy_load()

# print(f"Loaded {len(docs)} documents.")

# print(docs[0].page_content)
# print("metadata: ")
# print(docs[0].metadata)

for document in docs:
    print(document.metadata)