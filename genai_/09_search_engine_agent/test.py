import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun, DuckDuckGoSearchRun
from langchain.agents import initialize_agent, AgentType
from langchain.callbacks import StreamlitCallbackHandler
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize tools
api_wrapper_wiki = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=250)
wiki = WikipediaQueryRun(api_wrapper=api_wrapper_wiki)

api_wrapper_arxiv = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=250)
arxiv = ArxivQueryRun(api_wrapper=api_wrapper_arxiv)

search = DuckDuckGoSearchRun(name="Search")

# Streamlit app setup
st.title("🦜 Langchain Search Agent")
st.sidebar.title("Settings")

# Initialize session state for messages
if "messages" not in st.session_state:
    st.session_state['messages'] = [
        {'role': 'assistant', 'content': "Hi, I'm a chatbot. How can I help you?"}
    ]

# Display chat history
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg['content'])

# Handle user input
if prompt := st.chat_input(placeholder="What is machine learning?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    # Initialize LLM and agent
    llm = ChatGroq(model="llama3-8b-8192", streaming=True)
    tools = [wiki, arxiv, search]  # Prioritize Wikipedia and Arxiv

    search_agent = initialize_agent(
        tools,
        llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        handle_parsing_errors=True,
        # max_iterations=2  # Limit tool calls to avoid rate limits
    )

    # Process the query
    with st.chat_message("assistant"):
        st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
        try:
            response = search_agent.run(prompt, callbacks=[st_cb])
        except Exception as e:
            # Handle any errors, including DuckDuckGo rate limits
            response = "Sorry, I couldn't find an answer. Try asking differently or check back later."

        st.session_state.messages.append({"role": "assistant", "content": response})
        st.write(response)