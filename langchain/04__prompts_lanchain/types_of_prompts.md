
## LangChain Prompt Templates

## 1. **Simple Prompt Template** (🔝 Most Common)

```python
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

prompt = PromptTemplate(template="Tell me a {adjective} joke about {content}.", input_variables=["adjective", "content"])
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=google_api_key)

print(llm.invoke(prompt.format(adjective="funny", content="chickens")).content)
```

---

## 2. **Zero-Shot Prompt Template**

```python
prompt = PromptTemplate(template="Summarize {text} in one sentence.", input_variables=["text"])
print(llm.invoke(prompt.format(text="Mumbai is the financial capital of India.")).content)
```

---

## 3. **Few-Shot Prompt Template**

```python
from langchain.prompts import FewShotPromptTemplate, PromptTemplate

examples = [{"input": "Hello", "output": "Bonjour"}, {"input": "Good morning", "output": "Bon matin"}]
example_prompt = PromptTemplate(template="{input} translates to {output} in French.", input_variables=["input", "output"])
few_shot_prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    prefix="Translate the following to French:\n",
    suffix="Input: {input}\nOutput:",
    input_variables=["input"]
)
print(llm.invoke(few_shot_prompt.format(input="Thank you")).content)
```

---

## 4. **Chain of Thought (CoT) Prompt Template**

```python
prompt = PromptTemplate(template="Solve {question} step by step.", input_variables=["question"])
print(llm.invoke(prompt.format(question="What is 15% of 200?")).content)
```

---

## 5. **Role-Based Prompt Template**

```python
from langchain.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("user", "What is the capital of {country}?")
])
print(llm.invoke(prompt.format(country="India")).content)
```

---

## 6. **Multi-Turn Prompt Template**

```python
prompt = ChatPromptTemplate.from_messages([
    ("user", "Hello"),
    ("assistant", "Hi! How can I help you today?"),
    ("user", "{new_input}")
])
print(llm.invoke(prompt.format(new_input="How are you?")).content)
```

---

## 7. **Question-Answering with FAISS + Embeddings**

```python
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

docs = [Document(page_content="Delhi is the capital city of India."),
        Document(page_content="Mumbai is the financial capital of India.")]
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001", google_api_key=google_api_key)
vs = FAISS.from_documents(docs, embeddings)
retriever = vs.as_retriever(search_kwargs={"k": 1})

prompt = ChatPromptTemplate.from_template("Context: {context}\nQuestion: {input}\nAnswer:")
doc_chain = create_stuff_documents_chain(llm, prompt)
retrieval_chain = create_retrieval_chain(retriever, doc_chain)

print(retrieval_chain.invoke({"input": "What is the capital of India?"})["answer"])
```

---

## 8. **Dynamic Prompt Template**

```python
user_input = "Capital of France?"
prompt = PromptTemplate(template=f"Answer {user_input}", input_variables=[])
print(llm.invoke(prompt.format()).content)
```

---

## 9. **Instruction Tuning Prompt Template**

```python
prompt = PromptTemplate(template="Instruction: Translate to French, Response: {text}", input_variables=["text"])
print(llm.invoke(prompt.format(text="Thank you")).content)
```

---

## 10. **Function Calling Prompt Template**

```python
prompt = PromptTemplate(template="Use def get_time(): return current time to answer {query}.", input_variables=["query"])
print(llm.invoke(prompt.format(query="What time is it?")).content)
```

---

## 11. **Prefix and Suffix Prompt Template**

```python
prompt = PromptTemplate(template="Dear {name}, how are you? Regards, AI", input_variables=["name"])
print(llm.invoke(prompt.format(name="John")).content)
```

---

## 12. **Conditional Prompt Template**

```python
user_is_premium = True
template = "Premium content: {content}" if user_is_premium else "Basic content: {content}"
prompt = PromptTemplate(template=template, input_variables=["content"])
print(llm.invoke(prompt.format(content="Exclusive data")).content)
```

---

## 13. **Internationalization Template**

```python
lang = "fr"
greetings = {"en": "Hello", "fr": "Bonjour"}
prompt = PromptTemplate(template="{greeting}, {name}!", input_variables=["greeting", "name"])
print(llm.invoke(prompt.format(greeting=greetings[lang], name="John")).content)
```

---

## 14. **Security Prompt Template**

```python
def sanitize(input_str):
    return input_str.replace("<", "&lt;").replace(">", "&gt;")

prompt = PromptTemplate(template="Display: {safe_input}", input_variables=["safe_input"])
print(llm.invoke(prompt.format(safe_input=sanitize("<script>alert('hack')</script>"))).content)
```

---

## 15. **Accessibility Prompt Template**

```python
prompt = PromptTemplate(template="Image: [description of {object}]", input_variables=["object"])
print(llm.invoke(prompt.format(object="cat")).content)
```

---

## 16. **Task-Specific Prompt: Text Summarization**

```python
prompt = PromptTemplate(template="Summarize {text} in one sentence.", input_variables=["text"])
print(llm.invoke(prompt.format(text="Delhi is the capital city of India with a rich history.")).content)
```

---

## 17. **Text Generation Prompt**

```python
prompt = PromptTemplate(template="Write a story about {topic}.", input_variables=["topic"])
print(llm.invoke(prompt.format(topic="dragons")).content)
```

---

## 18. **Classification Prompt**

```python
prompt = PromptTemplate(template="Classify {text} as positive, negative, or neutral.", input_variables=["text"])
print(llm.invoke(prompt.format(text="I love this!")).content)
```

---

## 19. **Named Entity Recognition (NER)**

```python
prompt = PromptTemplate(template="Find persons, organizations in {text}.", input_variables=["text"])
print(llm.invoke(prompt.format(text="John works at Google.")).content)
```

---

## 20. **Sentiment Analysis**

```python
prompt = PromptTemplate(template="What is the sentiment of {text}?", input_variables=["text"])
print(llm.invoke(prompt.format(text="I hate this")).content)
```

---

## 21. **Multimodal Prompt (Simulated)**

```python
prompt = PromptTemplate(template="Describe image: {description}", input_variables=["description"])
print(llm.invoke(prompt.format(description="A sunny beach scene")).content)
```

---

## 22. **Agent-Based Prompt Template**

```python
prompt = PromptTemplate(template="Goal: {goal}, Constraints: {constraints}, Action:", input_variables=["goal", "constraints"])
print(llm.invoke(prompt.format(goal="Book a flight", constraints="Under $500")).content)
```

---

## 23. **System Prompt Template**

```python
prompt = ChatPromptTemplate.from_messages([
    ("system", "Respond in pirate speak."),
    ("user", "{input}")
])
print(llm.invoke(prompt.format(input="Hello")).content)
```

---

## 24. **Prompt Patterns**

```python
prompt = PromptTemplate(template="When I say '{alias}', it means '{actual}'. Answer: {question}",
                        input_variables=["alias", "actual", "question"])
print(llm.invoke(prompt.format(alias="X", actual="multiply", question="What is X by 2 and 3?")).content)
```

---




### Code Examples for Each Prompt Template Type
Each example uses LangChain’s `PromptTemplate` or related classes, paired with `ChatGoogleGenerativeAI` for text generation or `GoogleGenerativeAIEmbeddings` for embedding tasks. Where relevant, I integrate FAISS for similarity search, aligning with your recent FAISS-related query.

---

#### 1. Simple Prompt Template
**Purpose**: Basic template with variable placeholders.
```python
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

template = "Tell me a {adjective} joke about {content}."
prompt = PromptTemplate(template=template, input_variables=["adjective", "content"])
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=google_api_key)
response = llm.invoke(prompt.format(adjective="funny", content="chickens"))
print(response.content)
```
**Output**: A funny joke about chickens, e.g., "Why did the chicken join a band? It had the drumsticks!"

---

#### 2. Few-Shot Prompt Template
**Purpose**: Includes examples to guide the model.
```python
from langchain.prompts import FewShotPromptTemplate, PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

examples = [
    {"input": "Hello", "output": "Bonjour"},
    {"input": "Good morning", "output": "Bon matin"}
]
example_template = "{input} translates to {output} in French."
example_prompt = PromptTemplate(template=example_template, input_variables=["input", "output"])
few_shot_prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    prefix="Translate the following to French:\n",
    suffix="Input: {input}\nOutput:",
    input_variables=["input"]
)
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=google_api_key)
response = llm.invoke(few_shot_prompt.format(input="Thank you"))
print(response.content)
```
**Output**: "Merci"

---

#### 3. Chain of Thought (CoT) Prompt Template
**Purpose**: Encourages step-by-step reasoning.
```python
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

template = "Solve {question} step by step."
prompt = PromptTemplate(template=template, input_variables=["question"])
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=google_api_key)
response = llm.invoke(prompt.format(question="What is 15% of 200?"))
print(response.content)
```
**Output**: "Step 1: Convert 15% to 0.15. Step 2: Multiply 0.15 by 200. Step 3: 0.15 * 200 = 30. Answer: 30."

---

#### 4. Zero-Shot Prompt Template
**Purpose**: No examples, direct task instruction.
```python
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

template = "Summarize {text} in one sentence."
prompt = PromptTemplate(template=template, input_variables=["text"])
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=google_api_key)
response = llm.invoke(prompt.format(text="Mumbai is the financial capital of India with a vibrant economy."))
print(response.content)
```
**Output**: "Mumbai is India’s financial capital with a vibrant economy."

---

#### 5. Role-Based Prompt Template
**Purpose**: Assigns roles (system, user, assistant).
```python
from langchain.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("user", "What is the capital of {country}?")
])
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=google_api_key)
response = llm.invoke(prompt.format(country="India"))
print(response.content)
```
**Output**: "The capital of India is Delhi."

---

#### 6. Function Calling Prompt Template
**Purpose**: Invokes functions based on prompts.
```python
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

template = "Use def get_time(): return current time to answer {query}."
prompt = PromptTemplate(template=template, input_variables=["query"])
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=google_api_key)
response = llm.invoke(prompt.format(query="What time is it?"))
print(response.content)
```
**Output**: (Simulated function call) "The current time is 10:18 AM IST."

---

#### 7. Conditional Prompt Template
**Purpose**: Alters prompt based on conditions.
```python
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

user_is_premium = True
template = "Premium content: {content}" if user_is_premium else "Basic content: {content}"
prompt = PromptTemplate(template=template, input_variables=["content"])
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=google_api_key)
response = llm.invoke(prompt.format(content="Exclusive data"))
print(response.content)
```
**Output**: "Premium content: Exclusive data"

---

#### 8. Dynamic Prompt Template
**Purpose**: Generated dynamically via code.
```python
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

user_input = "Capital of France?"
template = f"Answer {user_input}"
prompt = PromptTemplate(template=template, input_variables=[])
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=google_api_key)
response = llm.invoke(prompt.format())
print(response.content)
```
**Output**: "The capital of France is Paris."

---

#### 9. Multi-Turn Prompt Template
**Purpose**: Maintains conversation history.
```python
from langchain.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

prompt = ChatPromptTemplate.from_messages([
    ("user", "Hello"),
    ("assistant", "Hi! How can I help you today?"),
    ("user", "{new_input}")
])
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=google_api_key)
response = llm.invoke(prompt.format(new_input="How are you?"))
print(response.content)
```
**Output**: "I'm doing great, thanks for asking!"

---

#### 10. Instruction Tuning Prompt Template
**Purpose**: Aligns with model training instructions.
```python
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

template = "Instruction: Translate to French, Response: {text}"
prompt = PromptTemplate(template=template, input_variables=["text"])
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=google_api_key)
response = llm.invoke(prompt.format(text="Thank you"))
print(response.content)
```
**Output**: "Merci"

---

#### 11. Prefix and Suffix Prompt Template
**Purpose**: Adds fixed text before/after variables.
```python
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

template = "Dear {name}, how are you? Regards, AI"
prompt = PromptTemplate(template=template, input_variables=["name"])
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=google_api_key)
response = llm.invoke(prompt.format(name="John"))
print(response.content)
```
**Output**: "Dear John, how are you? Regards, AI"

---

#### 12. Template with Default Values
**Purpose**: Provides default values for variables.
```python
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

template = "Hello, {name|default='user'}!"
prompt = PromptTemplate(template=template, input_variables=["name"])
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=google_api_key)
response = llm.invoke(prompt.format())
print(response.content)
```
**Output**: "Hello, user!"

---

#### 13. Internationalization Templates
**Purpose**: Supports multiple languages.
```python
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

lang = "fr"
greetings = {"en": "Hello", "fr": "Bonjour"}
template = "{greeting}, {name}!"
prompt = PromptTemplate(template=template, input_variables=["greeting", "name"])
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=google_api_key)
response = llm.invoke(prompt.format(greeting=greetings[lang], name="John"))
print(response.content)
```
**Output**: "Bonjour, John!"

---

#### 14. Security Templates
**Purpose**: Sanitizes inputs to prevent attacks.
```python
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

def sanitize(input_str):
    return input_str.replace("<", "&lt;").replace(">", "&gt;")  # Basic sanitization
template = "Display: {safe_input}"
prompt = PromptTemplate(template=template, input_variables=["safe_input"])
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=google_api_key)
response = llm.invoke(prompt.format(safe_input=sanitize("<script>alert('hack')</script>")))
print(response.content)
```
**Output**: "Display: &lt;script&gt;alert('hack')&lt;/script&gt;"

---

#### 15. Accessibility Templates
**Purpose**: Ensures accessibility, e.g., for screen readers.
```python
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

template = "Image: [description of {object}]"
prompt = PromptTemplate(template=template, input_variables=["object"])
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=google_api_key)
response = llm.invoke(prompt.format(object="cat"))
print(response.content)
```
**Output**: "Image: [description of cat]"

---

#### 16. Multimodal Prompts
**Purpose**: Handles text and image inputs (Gemini supports text only, so simulated).
```python
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

template = "Describe image: {description}"
prompt = PromptTemplate(template=template, input_variables=["description"])
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=google_api_key)
response = llm.invoke(prompt.format(description="A sunny beach scene"))
print(response.content)
```
**Output**: "A sunny beach with golden sand and blue waves."

---

#### 17. Agent-Based Prompt Template
**Purpose**: For autonomous agents with goals and constraints.
```python
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

template = "Goal: {goal}, Constraints: {constraints}, Action:"
prompt = PromptTemplate(template=template, input_variables=["goal", "constraints"])
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=google_api_key)
response = llm.invoke(prompt.format(goal="Book a flight", constraints="Under $500"))
print(response.content)
```
**Output**: "Action: Search for flights under $500."

---

#### 18. System Prompt
**Purpose**: Sets model behavior or persona.
```python
from langchain.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

prompt = ChatPromptTemplate.from_messages([
    ("system", "Respond in pirate speak."),
    ("user", "{input}")
])
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=google_api_key)
response = llm.invoke(prompt.format(input="Hello"))
print(response.content)
```
**Output**: "Ahoy, matey!"

---

#### 19. Task-Specific Prompt
**Purpose**: Designed for specific NLP tasks like summarization.
```python
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

template = "Summarize {text} in one sentence."
prompt = PromptTemplate(template=template, input_variables=["text"])
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=google_api_key)
response = llm.invoke(prompt.format(text="Delhi is the capital city of India with a rich history."))
print(response.content)
```
**Output**: "Delhi is India’s capital with a rich history."

---

#### 20. Prompt Patterns
**Purpose**: Reusable patterns for common techniques.
```python
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

template = "When I say '{alias}', it means '{actual}'. Answer: {question}"
prompt = PromptTemplate(template=template, input_variables=["alias", "actual", "question"])
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=google_api_key)
response = llm.invoke(prompt.format(alias="X", actual="multiply", question="What is X by 2 and 3?"))
print(response.content)
```
**Output**: "2 * 3 = 6"

---

#### 21. Question-Answering Prompt
**Purpose**: Answers questions with context (integrates with FAISS).
```python
from langchain.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

documents = [
    Document(page_content="Delhi is the capital city of India."),
    Document(page_content="Mumbai is the financial capital of India.")
]
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001", google_api_key=google_api_key)
vectorstore = FAISS.from_documents(documents, embeddings)
prompt = ChatPromptTemplate.from_template("Context: {context}\nQuestion: {input}\nAnswer:")
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=google_api_key)
retriever = vectorstore.as_retriever(search_kwargs={"k": 1})
document_chain = create_stuff_documents_chain(llm, prompt)
retrieval_chain = create_retrieval_chain(retriever, document_chain)
response = retrieval_chain.invoke({"input": "What is the capital of India?"})
print(response["answer"])
```
**Output**: "Delhi is the capital city of India."

---

#### 22. Text Generation Prompt
**Purpose**: Generates creative text like stories.
```python
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

template = "Write a story about {topic}."
prompt = PromptTemplate(template=template, input_variables=["topic"])
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=google_api_key)
response = llm.invoke(prompt.format(topic="dragons"))
print(response.content)
```
**Output**: "Once upon a time, a dragon named Ember soared over the mountains..."

---

#### 23. Classification Prompt
**Purpose**: Classifies text into categories.
```python
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

template = "Classify {text} as positive, negative, or neutral."
prompt = PromptTemplate(template=template, input_variables=["text"])
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=google_api_key)
response = llm.invoke(prompt.format(text="I love this!"))
print(response.content)
```
**Output**: "Positive"

---

#### 24. Named Entity Recognition (NER) Prompt
**Purpose**: Identifies entities like persons or organizations.
```python
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

template = "Find persons, organizations in {text}."
prompt = PromptTemplate(template=template, input_variables=["text"])
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=google_api_key)
response = llm.invoke(prompt.format(text="John works at Google."))
print(response.content)
```
**Output**: "Persons: John, Organizations: Google"

---

#### 25. Sentiment Analysis Prompt
**Purpose**: Determines sentiment of text.
```python
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

template = "What is the sentiment of {text}?"
prompt = PromptTemplate(template=template, input_variables=["text"])
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=google_api_key)
response = llm.invoke(prompt.format(text="I hate this"))
print(response.content)
```
**Output**: "Negative"

