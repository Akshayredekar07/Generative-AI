# conversational_chatbot_v3.py
# Conversational chatbot with RAG and chat message history

from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
import bs4
from bs4.filter import SoupStrainer
from dotenv import load_dotenv
import os

def initialize_rag_with_history():
    """Initialize RAG components with chat history support."""
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
    
    # Define history-aware prompt
    history_aware_prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        ("system", "Given the conversation history, generate a search query to retrieve relevant context.")
    ])
    
    # Define main prompt template
    prompt = ChatPromptTemplate.from_template(
        """You are a helpful AI assistant. Use the following context and conversation history to answer the question accurately and concisely.
        If the context doesn't provide enough information, use your general knowledge but indicate when you're doing so.
        
        Context: {context}
        
        Question: {input}
        
        Answer:"""
    )
    
    # Create chains
    retriever = vectorstore.as_retriever()
    history_aware_retriever = create_history_aware_retriever(model, retriever, history_aware_prompt)
    document_chain = create_stuff_documents_chain(model, prompt)
    retrieval_chain = create_retrieval_chain(history_aware_retriever, document_chain)
    
    # Initialize chat history store
    store = {}
    
    def get_session_history(session_id: str):
        """Get or create chat history for a session."""
        if session_id not in store:
            store[session_id] = ChatMessageHistory()
        return store[session_id]
    
    # Create chain with history
    chain_with_history = RunnableWithMessageHistory(
        retrieval_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history"
    )
    
    return chain_with_history

def get_response(chain, user_input, session_id="abc123"):
    """Get response from the RAG chatbot with history for a given user input."""
    response = chain.invoke(
        {"input": user_input},
        config={"configurable": {"session_id": session_id}}
    )
    return response["answer"]

def main():
    """Main function to run the RAG chatbot with history interactively."""
    chain = initialize_rag_with_history()
    print("RAG Chatbot with History (type 'exit' to quit)")
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            print("Goodbye!")
            break
        response = get_response(chain, user_input)
        print(f"Bot: {response}")

if __name__ == "__main__":
    main()