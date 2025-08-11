from dotenv import load_dotenv
from llama_index.llms import OpenAI
from llama_index.agent import ReActAgent
from llama_index.tools import FunctionTool
from llama_index.callbacks import LlamaDebugHandler, CallbackManager
import subprocess

load_dotenv()
llm = OpenAI()


def write_haiku(topic: str) -> str:
    """
    Writes a haiku about a given topic
    """
    return llm.complete(f"wrtie me a haiku about {topic}")


def count_characters(text: str) -> int:
    """
    Counts the number of characters in a text
    """
    return len(text)


def open_application(application_name: str) -> str:
    """
    Opens an application in my computer
    """
    try:
        subprocess.Popen(["/usr/bin/open", "-n", "-a", application_name])
        return "succesfully opened application"
    except Exception as e:
        print(f"Error: {str(e)}")


def open_url(url: str) -> str:
    """
    Opens a url in browser (chrome/ safari/ firefox)
    """
    try:
        subprocess.Popen(["/usr/bin/open", "--url", url])
        return "succesfully open url"
    except Exception as e:
        print(f"Error: {str(e)}")


if __name__ == "__main__":
    print("**** Hello Agents LlamaIndex ****")

    tool1 = FunctionTool.from_defaults(fn=write_haiku, name="Write Haiku")
    tool2 = FunctionTool.from_defaults(fn=count_characters, name="Count Chars")
    tool3 = FunctionTool.from_defaults(fn=open_application, name="Open App")
    tool4 = FunctionTool.from_defaults(fn=open_url, name="Open URL")

    llama_debug = LlamaDebugHandler(print_trace_on_end=True)
    callback_manager = CallbackManager(handlers=[llama_debug])
    agent = ReActAgent.from_tools(
        tools=[tool1, tool2, tool3, tool4],
        llm=llm,
        verbose=True,
        callback_manager=callback_manager,
    )
    # res = agent.query("Write me a haiku about tennis and then count the characters in it")
    # res = agent.query("Open discord in my computer")
    res = agent.query(
        "write a haiku about trees and then Open the url https://www.youtube.com/watch?v=dQw4w9WgXcQ in my chrome"
    )
    print(res)
