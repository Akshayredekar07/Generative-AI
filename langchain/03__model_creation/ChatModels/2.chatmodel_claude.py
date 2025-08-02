
from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv

load_dotenv()


model = ChatAnthropic(
	model_name="claude-opus-4-20250514",
	timeout=60,  # set your desired timeout in seconds
	stop=None    # or provide a list of stop sequences if needed
)


result = model.invoke("What is the capital of Maharashtra")

print(result.content)
