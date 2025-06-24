
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from pydantic import BaseModel, Field
from typing import Literal
from langchain_core.runnables import RunnableBranch, RunnableLambda, RunnableParallel
import json

# Initialize the model
model = ChatOllama(model='gemma3:1b')

# Define output parser for strings
parser = StrOutputParser()

# Define the Feedback Pydantic model
class Feedback(BaseModel):
    sentiment: Literal['positive', 'negative'] = Field(description='Sentiment of the feedback')

# Define the Pydantic output parser
parser2 = PydanticOutputParser(pydantic_object=Feedback)

# Prompt for sentiment classification
prompt1 = PromptTemplate(
    template=(
        "Classify the sentiment of the following feedback as 'positive' or 'negative'. "
        "Return only a JSON object with a single key 'sentiment' and the value 'positive' or 'negative'. "
        "Do not include any additional text or explanations.\n\n"
        "Feedback: {feedback}\n\n"
        "Example outputs:\n"
        "- Positive: {\"sentiment\": \"positive\"}\n"
        "- Negative: {\"sentiment\": \"negative\"}"
    ),
    input_variables=["feedback"]
)

# Function to clean and fix model output
def clean_output(text):
    try:
        start = text.find('{')
        end = text.rfind('}') + 1
        if start != -1 and end != -1:
            json_str = text[start:end]
            data = json.loads(json_str)
            if 'properties' in data and 'sentiment' in data['properties']:
                return json.dumps({'sentiment': data['properties']['sentiment']})
            return json_str
        if 'negative' in text.lower():
            return '{"sentiment": "negative"}'
        elif 'positive' in text.lower():
            return '{"sentiment": "positive"}'
        return '{"sentiment": "negative"}'  # Default
    except json.JSONDecodeError:
        if 'negative' in text.lower():
            return '{"sentiment": "negative"}'
        return '{"sentiment": "positive"}'

# Classifier chain
classifier_chain = prompt1 | model | RunnableLambda(clean_output) | parser2

# Prompts for responses
prompt2 = PromptTemplate(
    template='Write an appropriate response to this positive feedback:\n{feedback}',
    input_variables=['feedback']
)

prompt3 = PromptTemplate(
    template='Write an appropriate response to this negative feedback:\n{feedback}',
    input_variables=['feedback']
)

# Define branches with feedback extraction
positive_branch = RunnableLambda(lambda x: {'feedback': x['feedback'] if isinstance(x, dict) else getattr(x, 'feedback', None)}) | prompt2 | model | parser
negative_branch = RunnableLambda(lambda x: {'feedback': x['feedback'] if isinstance(x, dict) else getattr(x, 'feedback', None)}) | prompt3 | model | parser

# Define the branch chain
branch_chain = RunnableBranch(
    (lambda x: (x['sentiment'] if isinstance(x, dict) and 'sentiment' in x else Feedback(sentiment='negative')).sentiment == 'positive', positive_branch),
    (lambda x: (x['sentiment'] if isinstance(x, dict) and 'sentiment' in x else Feedback(sentiment='negative')).sentiment == 'negative', negative_branch),
    RunnableLambda(lambda x: "Could not determine sentiment")
)

# Combine chains to pass both sentiment and feedback
parallel_chain = RunnableParallel(
    sentiment=classifier_chain,
    feedback=lambda x: x['feedback']
)

# Final chain with sentiment printing
chain = parallel_chain | RunnableLambda(lambda x: (print(f"Sentiment: {x['sentiment'].sentiment}"), x['sentiment'].sentiment, x['feedback'])[1]) | branch_chain

# Test the chain
response = chain.invoke({"feedback": "This is a terrible smartphone"})
print("Response:", response)