from pdf_builder import report_tool
from analyst import analyze_filings, analyze_financials, analyze_news, analyze_parser
from langchain_core.messages import HumanMessage, SystemMessage
from deepagents import create_deep_agent
from dcf import find_dcf_tool
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from valuation_agent import valuation_tool

model = init_chat_model("gpt-4o", model_provider="openai")
llm = ChatOpenAI(model="gpt-4o", temperature=0.2, timeout=30)

MANAGER_PROMPT = """
You are a helpful professional equity analyst. Provide a comprehensive, thorough answer to any user query relevant to the performance
of a given company using the tools at your disposal. 
"""

REPORTING_PROMPT = """
                You are a professional financial analyst tasked with completely answering any relevant prompts given to you.

                You have access to 4 tools:
                - analyze_filings: find specific financial metrics of a specific company at a given time in its 10-Q and 10-K filings
                - find_dcf_tool: use this to perform a Discounted Cash Flow analysis of a given company in a specific year
                - analyze_financials: find financial ticker data of a company
                - valuation_tool: find a valuation analysis of a company in a given year written by an equity research analyst

                Make sure that the user gets an accurate, concise response with data-driven reasoning. 
                """

reporting_tools = [analyze_filings, find_dcf_tool, analyze_financials, valuation_tool]


reporting_agent = create_deep_agent(model=llm, 
                     system_prompt=REPORTING_PROMPT, 
                     tools=reporting_tools, 
                     )


@tool
def create_report(request: str) -> str:
    """
    Create a comprehensive financial report
    Use this when the user wants a financial evaluation of a company 
    """
    res = reporting_agent.invoke({
        "messages" : [{"role": "user", "content" : request}]
    })

    return res["messages"][-1].text


manager_agent = create_deep_agent(
    model=model,
    tools=[create_report],
    system_prompt=MANAGER_PROMPT
)



USER_PROMPT = "make a prediction on how Apple will perform in 2025 using all the tools you have"

messages = [SystemMessage(content="You are a helpful summarizer. Please focus on the key points and avoid writing an answer longer than 50 words."),
            HumanMessage(content=f"Summarize and reformulate the following in a clear, succinct manner: {USER_PROMPT}"),
            ]
        
USER_PROMPT_REDO = model.invoke(messages).content

print(USER_PROMPT_REDO)

print("AAAAAAAAAAAAAAAAAAAAA")

response = manager_agent.invoke(

    {"messages" : [{"role" : "user", "content" : USER_PROMPT_REDO}]}
)

text = response["messages"][-1].content

print(text)

"""
TODO:
manager agent orchestrates everything


MERGE THE FILING/PARSING AGENT!!!!!!!!!!!!!
filings agent just retrieves filings and such, updating the chroma db with information
parsing agent parses through filings and data, extracting revenues and earnings 


the valuation agent is more like a valuation tool rofl rofl rofl rofl 
valuation agent interacts with filings/parsing agent supposedly (to get data????) and uses DCF tool to output valuation

reporting agent writes down report and such
"""
