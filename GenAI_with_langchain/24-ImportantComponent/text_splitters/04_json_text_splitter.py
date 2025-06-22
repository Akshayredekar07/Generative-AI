
import json
import requests
from langchain_text_splitters import RecursiveJsonSplitter


json_data=requests.get("https://api.smith.langchain.com/openapi.json").json()

# print(json_data)

json_splitter = RecursiveJsonSplitter(max_chunk_size=300)

json_chunks=json_splitter.split_json(json_data)

# print(json_data)

for chunk in json_chunks[:3]:
    print(chunk)


## The splitter can also output documents
docs=json_splitter.create_documents(texts=[json_data])
for doc in docs[:3]:
    print(doc)



texts=json_splitter.split_text(json_data)
print(texts[0])
print(texts[1])