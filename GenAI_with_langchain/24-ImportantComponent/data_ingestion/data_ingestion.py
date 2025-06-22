
# Text loaders

from langchain_community.document_loaders import TextLoader

# loader = TextLoader('speech.txt')

# text_document=loader.load()
# print(text_document)


# Reading the pdf files 
from langchain_community.document_loaders import PyPDFLoader
loader=PyPDFLoader('attention.pdf')
docs = loader.load()
# print(type(docs), "\n")
# print(type(docs[0]), "\n")
# print(docs)


## Web based loader
from langchain_community.document_loaders import WebBaseLoader
from bs4.filter import SoupStrainer
loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs=dict(parse_only=SoupStrainer(
        class_=("post-title", "post-content", "post-header")
    ))
)

# web_data = loader.load()
# print(web_data)



## Arxiv
from langchain_community.document_loaders import ArxivLoader

# docs_data = ArxivLoader(query="1706.03762", load_max_docs=2).load()
# import os
# os.environ["USER_AGENT"] = "akshayredekar@gmail.com" 

# print(len(docs_data))
# print(docs_data)


from langchain_community.document_loaders import WikipediaLoader

docs = WikipediaLoader(query="Generative AI", load_max_docs=2).load()

print(docs)