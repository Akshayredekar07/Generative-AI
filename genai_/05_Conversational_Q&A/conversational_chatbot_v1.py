
# conversational_chatbot_v1.py
# Basic conversational chatbot without RAG or chat history

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

def initialize_chatbot():
    """Initialize the chatbot with model and prompt template."""
    model = ChatOllama(model="llama3.2:1b")
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful AI assistant. Provide clear and concise answers to user questions."),
        ("human", "{input}")
    ])
    return prompt | model

def get_response(chain, user_input):
    """Get response from the chatbot for a given user input."""
    response = chain.invoke({"input": user_input})
    return response.content

def main():
    """Main function to run the chatbot interactively."""
    chain = initialize_chatbot()
    print("Basic Chatbot (type 'exit' to quit)")
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            print("Goodbye!")
            break
        response = get_response(chain, user_input)
        print(f"Bot: {response}")

if __name__ == "__main__":
    main()