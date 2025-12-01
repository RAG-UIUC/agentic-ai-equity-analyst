import sys 
from streamlit.web import cli as stcli
from langchain.tools import tool

def report(content):
    file = open("report.txt", "w", encoding="utf-8")
    file.write(content)
    file.close()

    APP_PATH = "agentic-ai-equity-analyst/streamlit_app.py"
    run_streamlit_app(APP_PATH)

    return True 

def run_streamlit_app(app_path: str) -> None:
    """
    Run a Streamlit app programmatically using Streamlit's internal CLI.
    
    Args:
        app_path: Path to the Streamlit app file (e.g., "my_app.py").
    """
    original_argv = sys.argv.copy()
    
    try:
        sys.argv = ["streamlit", "run", app_path]
        stcli.main()        
    finally:
        sys.argv = original_argv
 
@tool
def report_tool(content):
   """
   Write a report containing the content provided as the argument 
   Returns a boolean indicating whether writing the report was successful
   """
   return report(content)
