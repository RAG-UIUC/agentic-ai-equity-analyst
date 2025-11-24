import streamlit as st
from langchain.tools import tool

#@tool 
def report(content):
   """Write a report containing the content provided as the argument 
   
   """

   st.write(content)

