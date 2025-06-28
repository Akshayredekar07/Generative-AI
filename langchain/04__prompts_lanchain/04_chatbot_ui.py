import gradio as gr
from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

# Load environment variables (if needed)
load_dotenv()

# Load the Hugging Face endpoint model
llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.3",
    model="mistralai/Mistral-7B-Instruct-v0.3",
    task="text-generation"
)
model = ChatHuggingFace(llm=llm)

# Function to handle chat submission
def chat_function(message, history):
    if message.strip().lower() == "exit":
        return history + [[message, "Session ended. Type reload to restart."]], ""
    response = model.invoke(message)
    history.append([message, response.content])
    return history, ""

# Create the interface with Blocks
with gr.Blocks(title="🧠 Chat with Mistral-7B") as demo:
    # Output box (chat history) at the top
    chatbot = gr.Chatbot()
    
    # Input text box below the output
    textbox = gr.Textbox(placeholder="Type your message here...", label="You")
    
    # Row for "Clear" and "Submit" buttons below the text box
    with gr.Row():
        clear_button = gr.Button("Clear")
        submit_button = gr.Button("Submit", variant="primary")  # Changed to orange with primary variant
    
    # Exit info below the buttons
    gr.Markdown("Type exit to end the session.")
    
    # "Flag" button at the bottom
    flag_button = gr.Button("Flag")
    
    # Submit button functionality: update chat and clear input
    submit_button.click(
        fn=chat_function,
        inputs=[textbox, chatbot],
        outputs=[chatbot, textbox]
    )
    
    # Clear button functionality: clear the text box
    clear_button.click(
        fn=lambda: "",
        inputs=[],
        outputs=[textbox]
    )
    
    # Flag button (no functionality yet, can be added if needed)
    # flag_button.click(fn=some_function, inputs=[chatbot], outputs=[])

# Launch the app
demo.launch()