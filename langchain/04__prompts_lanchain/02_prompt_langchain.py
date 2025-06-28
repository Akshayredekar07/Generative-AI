
import streamlit as st
from dotenv import load_dotenv
from langchain_core.prompts import load_prompt
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

# Load environment variables
load_dotenv()

# Load LLM
llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.3",
    model="mistralai/Mistral-7B-Instruct-v0.3",
    task="text-generation"
)
model = ChatHuggingFace(llm=llm)

# Load prompt
template = load_prompt("template.json")

# Streamlit UI
st.title("🧠 Research Paper Summarizer")

paper_input = st.selectbox(
    "Select Research Paper",
    ["Select...", "Attention Is All You Need", "BERT: Pre-training of Deep Bidirectional Transformers", 
     "GPT-3: Language Models are Few-Shot Learners", "Diffusion Models Beat GANs on Image Synthesis"]
)

style_input = st.selectbox(
    "Explanation Style", ["Beginner-Friendly", "Technical", "Code-Oriented", "Mathematical"]
)

length_input = st.selectbox(
    "Explanation Length", ["Short (1-2 paragraphs)", "Medium (3-5 paragraphs)", "Long (detailed explanation)"]
)

if st.button("Summarize"):
    if paper_input == "Select...":
        st.warning("Please select a valid paper.")
    else:
        with st.spinner("Generating explanation..."):
            chain = template | model
            result = chain.invoke({
                "paper_input": paper_input,
                "style_input": style_input,
                "length_input": length_input
            })
            st.markdown("### Response")
            st.write(result.content)
