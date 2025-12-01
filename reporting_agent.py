# snake_case for variables, functions
# camelCase for objects, scripts 

from langchain_openai import ChatOpenAI
from analyst import analyze_filings, analyze_financials
from dataclasses import dataclass
from deepagents import create_deep_agent
from dcf import find_dcf_tool
from valuation_agent import valuation_tool


#llm = ChatOpenAI(model="gpt-4o", temperature=0.2, timeout=30)
tools = [analyze_filings, find_dcf_tool, analyze_financials, valuation_tool]
tools_by_name = {tool.name: tool for tool in tools}
#llm.bind_tools(tools)

SYSTEM_PROMPT = """
                You are a professional financial analyst tasked with completely answering any relevant prompts given to you.

                You have access to 4 tools:
                - analyze_filings: find specific financial metrics of a specific company at a given time in its 10-Q and 10-K filings
                - find_dcf_tool: use this to perform a Discounted Cash Flow analysis of a given company in a specific year
                - analyze_financials: find financial ticker data of a company
                - valuation_tool: find a valuation analysis of a company in a given year written by an equity research analyst

                Make sure that the user gets an accurate, concise report with data-driven reasoning. 
                """

@dataclass
class ResponseFormat:
    """Response format for the agent"""
    answer: str 

"""
agent = create_deep_agent(model=llm, 
                     system_prompt=SYSTEM_PROMPT, 
                     tools=tools, 
                     )

response = agent.invoke(

    {"messages" : [{"role" : "user", "content" : "make a prediction on how Apple will perform in 2025 using all the tools you have"}]}
)

text = response["messages"][-1].content

print(text)
"""