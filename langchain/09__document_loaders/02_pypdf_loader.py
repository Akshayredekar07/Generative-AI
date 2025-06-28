
from langchain_community.document_loaders import PyPDFLoader


loader = PyPDFLoader("REACT-Agent.pdf")

# Load and split the PDF into pages
pages = loader.load_and_split()


# print(pages[0].page_content[:100])
# print(pages[0].metadata)

# print(pages[1].page_content)


import pprint

pprint.pp(pages[0].metadata)