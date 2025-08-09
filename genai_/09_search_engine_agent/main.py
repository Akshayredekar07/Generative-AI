
import streamlit as st 
from langchain_groq import ChatGroq
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun,DuckDuckGoSearchRun
from langchain.agents import initialize_agent, AgentType
from langchain.callbacks import StreamlitCallbackHandler
import os 

from dotenv import load_dotenv
load_dotenv()

# Tools
api_wrapper_wiki=WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=250, wiki_client=None)
wiki=WikipediaQueryRun(api_wrapper=api_wrapper_wiki)

api_wrapper_arxiv = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=250,arxiv_search=None,arxiv_exceptions=None)
arxiv = ArxivQueryRun(api_wrapper=api_wrapper_arxiv)

search=DuckDuckGoSearchRun(name="Search")

# Set the title
st.title("🦜Langchain-Search-Agent")

# sidebar for setting 
st.sidebar.title("Settings")

if "messages" not in st.session_state:
    st.session_state['messages'] = [
        {'role': 'assistant',
         'content':'Hi, I"m a chatbot. How can i help you?'
         }
    ]


for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg['content'])

if prompt:=st.chat_input(placeholder="What is machine learning?"):
    st.session_state.messages.append({"role":"user","content":prompt})
    st.chat_message("user").write(prompt)

    llm = ChatGroq(model="llama3-8b-8192", streaming=True)
    tools = [search, arxiv, wiki]

    search_agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    handle_parsing_errors=True,
    max_iterations=5  # Limit tool usage
)


    with st.chat_message("assistant"):
        st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
        try:
            response = search_agent.run(st.session_state.messages, callbacks=[st_cb])
        except Exception as e:
            response = "Sorry, I couldn't find an answer to your question."

        st.session_state.messages.append({"role":"assistant", "content":response})
        st.write(response)

