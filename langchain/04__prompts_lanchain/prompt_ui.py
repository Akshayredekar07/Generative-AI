import streamlit as st
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain_core.prompts import load_prompt
from langchain_groq import ChatGroq
load_dotenv()

model = ChatGroq(model="llama-3.3-70b-versatile")

st.title("Research Paper Explainer")
st.write("Explain research papers in your preferred style.")

paper_input = st.selectbox(
    "Select Research Paper",
    ["Select...",
    "Chain of Thought Prompting(CoT)",
    "Attention is All you Need",
    "BERT: Pretraining of Deep Bidirectional Transformers",
    "GPT-3: Language Models are Few-Shot Learners",
    "Diffusion Models Beats GANs on Image Synthesis"
    ]
)

style_input = st.selectbox(
    "Select Explanation Style",
    ["Beginner-Friendly", "Technical", "Code-Oriented", "Mathematical"]
)

length_input = st.selectbox(
    "Select Explanation Length",
    ["Short (1-2 paragraphs)", "Medium (3-5 paragraphs)", "Long (detailed explanation)"]
)


template = load_prompt("template.json")


prompt = template.format(
    paper_input=paper_input,
    style_input=style_input,
    length_input=length_input
)


if st.button("Generate Explanation"):
    if paper_input=="Select...":
        st.warning("Please select a research paper.")
    else:
        with st.spinner("Generating explanation..."):
            response=model.invoke(prompt)
            st.markdown("### Response")
            st.write(response.content)