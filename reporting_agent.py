# snake_case for variables, functions
# camelCase for objects, scripts 

from langchain_openai import ChatOpenAI
from pdf_builder import report 
from analyst import analyze
from dataclasses import dataclass
from deepagents import create_deep_agent
from dcf import find_dcf_tool


llm = ChatOpenAI(model="gpt-4o", temperature=0.2, timeout=30)
tools = [analyze, find_dcf_tool]
tools_by_name = {tool.name: tool for tool in tools}
llm.bind_tools(tools)

SYSTEM_PROMPT = """
                You are a professional financial analyst tasked with answering any relevant prompts given to you.

                You have access to two tools:
                - analyze: use this to get an analysis of financial data pertinent to the prompt given to you
                - find_dcf_tool: use this to perform a Discounted Cash Flow analysis of a given company in a specific year
                

                Make sure that the user gets an accurate, concise response. 
                """

@dataclass
class ResponseFormat:
    """Response format for the agent"""
    answer: str 

agent = create_deep_agent(model=llm, 
                     system_prompt=SYSTEM_PROMPT, 
                     tools=tools, 
                     )


response = agent.invoke(

    {"messages" : [{"role" : "user", "content" : "perform a financial analysis on Apple in 2024 with metrics from the dcf"}]}
)

text = response["messages"][-1].content

print(text)
report(text)