import gradio as gr
from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

# Load environment variables (if needed)
load_dotenv()

# Load the Hugging Face endpoint model
llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.3",
    task="text-generation"
)
model = ChatHuggingFace(llm=llm)

# Maintain chat history
chat_history = []

def chat_with_model(user_input, history):
    if user_input.strip().lower() == "exit":
        return history + [[user_input, "Session ended. Type reload to restart."]], gr.update(interactive=False)
    
    response = model.invoke(user_input)
    history.append([user_input, response.content])
    return history, history

# Create the Gradio interface
chatbot = gr.Chatbot()
demo = gr.Interface(
    fn=chat_with_model,
    inputs=[
        gr.Textbox(placeholder="Type your message here...", label="You"),
        gr.State([])
    ],
    outputs=[
        chatbot,
        gr.State([])
    ],
    title="🧠 Chat with Mistral-7B",
    description="Type `exit` to end the session."
)

# Launch the app
demo.launch()
