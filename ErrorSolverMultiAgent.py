# streamlit_app.py
import streamlit as st
from langchain_groq import ChatGroq
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from openai import OpenAI
from dotenv import load_dotenv
from typing import Annotated, TypedDict

load_dotenv()

# ------------------ Model setup ------------------
error_finder_model = ChatGroq(model="openai/gpt-oss-20b", temperature=0.4,max_tokens=950)
error_solver_model = ChatGroq(model="openai/gpt-oss-20b", temperature=0.4,max_tokens=950)

# ------------------ State definition ------------------
class State(TypedDict):
    code_snippet: str
    message: Annotated[list, add_messages]
    error_analysis: str
    corrected_code: str

# ------------------ Nodes ------------------
def error_analysis_node(state: State):
    code_snippet = state["code_snippet"]
    prompt = (
        "Analyze the following code snippet and identify any potential errors, "
        "issues, or areas for improvement. Provide a detailed explanation of the "
        "problems found.\n\n"
        "Do not include any additional text or explanations."
        f"Code Snippet:\n{code_snippet}\n\n"
        "Return your analysis in a clear and structured format and avoid any unnecessary commentary and analysis within the token limit. "
    )
    response = error_finder_model.invoke(prompt)
    state["error_analysis"] = response.content
    return state

def error_solver_node(state: State):
    code_snippet = state["code_snippet"]
    error_analysis = state["error_analysis"]

    prompt = (
        "You are a code debugging analyzer.\n\n"
        "Analyze the following code and identify ONLY actual or highly probable "
        "syntax errors, runtime errors, and logical errors.\n\n"
        "STRICT RULES:\n"
        "1. Do not invent missing values.\n"
        "2. Do not assume the programmer's intention when it cannot be determined.\n"
        "3. Clearly distinguish between syntax errors, runtime errors, and logical errors.\n"
        "4. Do not treat PEP 8 style issues as errors unless they affect execution.\n"
        "5. Do not suggest unnecessary refactoring.\n"
        "6. If a variable is used before being defined, explicitly identify it.\n"
        "7. Explain what causes the error and what information is required to fix it.\n\n"
        f"Code Snippet:\n{code_snippet}\n\n"
        "Return ONLY the corrected code. "
        "Do not include explanations or markdown."
    )

    response = error_solver_model.invoke(prompt)
    corrected_code = response.content

    # Clean up the response
    if "<think>" in corrected_code:
        corrected_code = corrected_code.split("</think>")[-1].strip()
    corrected_code = corrected_code.replace("```python", "").replace("```", "").strip()

    state["corrected_code"] = corrected_code
    return state

def general_node(state: State):
    state["error_analysis"] = "No specific errors identified."
    state["corrected_code"] = state["code_snippet"]
    return state

def response_node(state: State):
    return {
        "error_analysis": state["error_analysis"],
        "corrected_code": state["corrected_code"],
    }

def route_node(state: State):
    if "error" in state["error_analysis"].lower():
        return "error_solver"
    else:
        return "general"

# ------------------ Build graph ------------------
graph = StateGraph(State)
graph.add_node("error_analysis", error_analysis_node)
graph.add_node("error_solver", error_solver_node)
graph.add_node("general", general_node)
graph.add_node("response", response_node)

graph.add_edge(START, "error_analysis")
graph.add_conditional_edges("error_analysis", route_node)
graph.add_edge("error_solver", "response")
graph.add_edge("general", "response")
graph.add_edge("response", END)

app = graph.compile()

# ------------------ Helper function ------------------
def process_code(code_snippet: str) -> dict:
    state = {"code_snippet": code_snippet, "message": []}
    result = app.invoke(state)
    return {
        "error_analysis": result["error_analysis"],
        "corrected_code": result["corrected_code"],
    }

# ------------------ Streamlit UI ------------------
st.set_page_config(page_title="Code Error Solver", page_icon="🐞", layout="wide")
st.title("🐞 Code Error Solver")
st.markdown("Paste your Python code below and get a detailed analysis and suggested corrections.")

# Input area
code_input = st.text_area(
    "Code Snippet",
    height=300,
    placeholder="Enter your code here...",
    key="code_input"
)

# Submit button
if st.button("Analyze Code", type="primary"):
    if not code_input.strip():
        st.warning("Please enter some code first.")
    else:
        with st.spinner("Analyzing code..."):
            try:
                result = process_code(code_input)
                
                # Display results in two columns
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("Error Analysis")
                    st.markdown(result["error_analysis"])
                with col2:
                    st.subheader("Corrected Code")
                    # Use code block for better formatting
                    st.code(result["corrected_code"], language="python")
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

# Footer
st.markdown("---")
st.caption("Powered by LangChain, LangGraph, Mistral AI, and Groq.")