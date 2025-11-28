import streamlit as st
from langchain.tools import tool

def report(content):
   st.write(content)

@tool
def report_tool(content):
   """
   Write a report containing the content provided as the argument 
   """
   return report(content)
