from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.models.huggingface import HuggingFace
from dotenv import load_dotenv
import os

load_dotenv()

token_hf = os.getenv('HF_TOKEN')
model_hf = os.getenv('HF_MODEL')
token_open = os.getenv('OPENAI_TOKEN')
model_open = os.getenv('OPENAI_MODEL')

description = """
    You are a simple assitant to recommend parking spots ITESO located in Tlaquepaque, Jalisco.

    The classes start at 7am and end at 10pm with a break from 3pm to 4pm

    The response should include the following information:
    - 3 options for parking spots
    - Emojis like 🚗
    - The hpus of the class
"""
agent_open = Agent(
    model=OpenAIChat(
        api_key=token_open,
        id=model_open,
    ),
    markdown=True,
    description=description
)


agent_hf = Agent(
    model=HuggingFace(
        api_key=token_hf,
        id=model_hf,
        max_tokens=1000,
    ),
    markdown=False,
    description=description,    
)
try:
    prompt = str(input("Enter a prompt: "))
    print(15*"-" + "Huggingface" + 15*"-")
    agent_hf.print_response(prompt)
    # print(15*"-" + "OpenAI" + 15*"-")
    # agent_open.print_response(prompt)
except Exception as e:
    print(f"error: {e}")
