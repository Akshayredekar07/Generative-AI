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
from bs4.filter import SoupStrainer
from dotenv import load_dotenv
import os
import gradio as gr

def initialize_rag_with_history():
    load_dotenv()
    os.environ['HF_HOME'] = 'D:/huggingface_cache'

    model = ChatOllama(model="llama3.2:1b")
    embeddings = OllamaEmbeddings(model="nomic-embed-text")

    loader = WebBaseLoader(
        web_paths=["https://lilianweng.github.io/posts/2023-06-23-agent/"],
        bs_kwargs={"parse_only": SoupStrainer(class_=("post-content", "post-title", "post-header"))}
    )
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)

    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)

    history_aware_prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        ("system", "Given the conversation history, generate a search query to retrieve relevant context.")
    ])

    prompt = ChatPromptTemplate.from_template(
        """You are a helpful AI assistant. Use the following context and conversation history to answer the question accurately and concisely.
        If the context doesn't provide enough information or the question is unrelated, use your general knowledge and indicate when you're doing so.
        
        Context: {context}
        
        Question: {input}
        
        Answer:"""
    )

    retriever = vectorstore.as_retriever()
    history_aware_retriever = create_history_aware_retriever(model, retriever, history_aware_prompt)
    document_chain = create_stuff_documents_chain(model, prompt)
    retrieval_chain = create_retrieval_chain(history_aware_retriever, document_chain)

    store = {}

    def get_session_history(session_id: str):
        if session_id not in store:
            store[session_id] = ChatMessageHistory()
        return store[session_id]

    chain_with_history = RunnableWithMessageHistory(
        retrieval_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history"
    )

    return chain_with_history, store

def main():
    chain, store = initialize_rag_with_history()
    session_id = "abc123"

    def chatbot_interface(user_input, history):
        if not history:
            history = []
        response = chain.invoke(
            {"input": user_input},
            config={"configurable": {"session_id": session_id}}
        )
        answer = response["answer"]
        history.append((user_input, answer))
        return "", history

    def clear_chat():
        if session_id in store:
            store[session_id].clear()
        return [], ""

    # --- Custom CSS for styling ---
    custom_css = """
    body {
        background-color: #202123;
        color: white;
    }
    .gradio-container {
        max-width: 800px;
        margin: 0 auto;
    }
    #chatbot {
        height: 75vh;
        overflow-y: auto;
        background-color: #343541;
        border-radius: 10px;
        padding: 15px;
    }
    #input-area {
        position: sticky;
        bottom: 0;
        background-color: #40414f;
        padding: 10px;
        border-radius: 10px;
        margin-top: 10px;
    }
    """

    with gr.Blocks(css=custom_css) as demo:
        gr.Markdown("## 💬 Chatbot: RAG + General Knowledge", elem_id="title")

        chatbot = gr.Chatbot(label="Conversation", elem_id="chatbot")

        with gr.Row(elem_id="input-area"):
            user_input = gr.Textbox(
                placeholder="Ask anything here...",
                show_label=False,
                container=False,
                scale=6
            )
            send_btn = gr.Button("Send", scale=1)
            clear_btn = gr.Button("Clear", scale=1)

        send_btn.click(
            fn=chatbot_interface,
            inputs=[user_input, chatbot],
            outputs=[user_input, chatbot]
        )
        user_input.submit(
            fn=chatbot_interface,
            inputs=[user_input, chatbot],
            outputs=[user_input, chatbot]
        )
        clear_btn.click(
            fn=clear_chat,
            outputs=[chatbot, user_input]
        )

    demo.launch()

if __name__ == "__main__":
    main()
