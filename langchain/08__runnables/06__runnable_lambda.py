
from langchain_core.runnables import RunnableLambda

def word_counter(text):
    return len(text.split())


runnable_word_counter = RunnableLambda(word_counter)

count = runnable_word_counter.invoke('Hi there how are you?')
print(count)