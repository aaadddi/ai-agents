import os
import getpass
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from typing import List

def set_env_key():
    """Set up Google API key from environment or prompt user"""
    load_dotenv()
    os.environ["GOOGLE_API_KEY"] = os.getenv('GOOGLE_API_KEY')
    
    if "GOOGLE_API_KEY" not in os.environ or not os.environ["GOOGLE_API_KEY"]:
        os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter your Google AI API key: ")

def get_llm(tools:List = None):
    # Ensure API key is set before creating LLM
    set_env_key()
    if tools:
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", 
                                    tools= tools)
    else :
        llm =  ChatGoogleGenerativeAI(model="gemini-2.5-flash")

    return llm