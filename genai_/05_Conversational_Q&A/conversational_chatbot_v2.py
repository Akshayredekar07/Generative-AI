# conversational_chatbot_v2.py
# Conversational chatbot with RAG capabilities

from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
import bs4
from bs4.filter import SoupStrainer

from dotenv import load_dotenv
import os

def initialize_rag():
    """Initialize RAG components including document loading and vector store."""
    # Load environment variables
    load_dotenv()
    os.environ['HF_HOME'] = 'D:/huggingface_cache'

    # Initialize model and embeddings
    model = ChatOllama(model="llama3.2:1b")
    embeddings = OllamaEmbeddings(model="nomic-embed-text")

    # Load and process documents
    loader = WebBaseLoader(
        web_paths=["https://lilianweng.github.io/posts/2023-06-23-agent/"],
        bs_kwargs={
        "parse_only": SoupStrainer(
        class_=("post-content", "post-title", "post-header")
)
        },
    )
    docs = loader.load()

    # Split documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)

    # Create vector store
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
    
    # Define prompt template
    prompt = ChatPromptTemplate.from_template(
        """You are a helpful AI assistant. Use the following context to answer the question accurately and concisely.
        If the context doesn't provide enough information, use your general knowledge but indicate when you're doing so.
        
        Context: {context}
        
        Question: {input}
        
        Answer:"""
    )
    
    # Create document chain and retrieval chain
    document_chain = create_stuff_documents_chain(model, prompt)
    retriever = vectorstore.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    
    return retrieval_chain

def get_response(chain, user_input):
    """Get response from the RAG chatbot for a given user input."""
    response = chain.invoke({"input": user_input})
    return response["answer"]

def main():
    """Main function to run the RAG chatbot interactively."""
    chain = initialize_rag()
    print("RAG Chatbot (type 'exit' to quit)")
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            print("Goodbye!")
            break
        response = get_response(chain, user_input)
        print(f"Bot: {response}")

if __name__ == "__main__":
    main()