import os
import re
from dotenv import load_dotenv
from typing import Annotated, TypedDict

import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END

load_dotenv()

# ----------------------------------------------------------------------
# Page config
# ----------------------------------------------------------------------
st.set_page_config(page_title="College Assistant", page_icon="🎓", layout="centered")


# ----------------------------------------------------------------------
# Cached resources — built once per server process, not per rerun
# ----------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading embedding model...")
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


@st.cache_resource(show_spinner="Indexing college documents...")
def build_retrievers():
    embeddings = get_embeddings()

    def retriever_node(path_url: str):
        loader = PyPDFLoader(path_url)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_documents(documents)
        vectorstore = FAISS.from_documents(chunks, embeddings)
        return vectorstore.as_retriever(search_kwargs={"k": 4})

    academic_retriever = retriever_node("academics_handbook.pdf")
    fee_retriever = retriever_node("fee_structure.pdf")
    return academic_retriever, fee_retriever


@st.cache_resource(show_spinner=False)
def get_model():
    return ChatGroq(model="openai/gpt-oss-20b", temperature=0.4, api_key=os.getenv("GROQ_API_KEY"))


academic_retriever, fee_retriever = build_retrievers()
model = get_model()


# ----------------------------------------------------------------------
# Graph state + nodes (unchanged logic, just module-level so they can
# see the cached retriever/model objects above)
# ----------------------------------------------------------------------
class State(TypedDict):
    program: str
    message: Annotated[list, add_messages]
    query_type: str
    retrieved_context: str


def classifier_node(state: State):
    last_message = state["message"][-1].content
    prompt = (
        "Classify the following student query into exactly one category: "
        "'academic', 'fee', or 'general'.\n\n"
        "Use 'academic' for questions about attendance, exams, grading, credits, "
        "promotion, course structure, summer training, or degree requirements.\n"
        "Use 'fee' for questions about tuition, payment, refund, late charges, "
        "scholarships, or any money-related topic.\n"
        "Use 'general' for greetings, casual talk, or anything not related to "
        "the college rules or fee.\n\n"
        f"Query: {last_message}\n\n"
        "Return only one word: academic, fee, or general."
    )
    response = model.invoke(prompt)
    labels = re.findall(r"\b(academic|fee|general)\b", response.content.lower())
    category = labels[-1] if labels else "general"
    return {"query_type": category}


def academic_rag_node(state: State):
    query = state["message"][-1].content
    docs = academic_retriever.invoke(query)
    context = "\n\n".join([doc.page_content for doc in docs])
    return {"retrieved_context": context}


def fee_rag_node(state: State):
    query = state["message"][-1].content
    docs = fee_retriever.invoke(query)
    context = "\n\n".join([doc.page_content for doc in docs])
    return {"retrieved_context": context}


def general_node(state: State):
    return {"retrieved_context": "NO_RETRIEVAL_NEEDED"}


def response_node(state: State):
    query = state["message"][-1].content
    programme = state.get("program", "unknown")
    context = state["retrieved_context"]

    if context == "NO_RETRIEVAL_NEEDED":
        prompt = (
            f"You are a friendly college assistant talking to a {programme} student. "
            f"Answer this question using your own general knowledge:\n\n{query}"
        )
    else:
        prompt = (
            f"You are a college assistant helping a {programme} student. "
            f"Use the following context from the official college documents to answer "
            f"the question accurately. If the context mentions specific figures for "
            f"different programmes, highlight the one relevant to {programme} if possible.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {query}\n\n"
            f"Give a clear, friendly, and precise answer."
        )

    answer = model.invoke(prompt)
    content = answer.content

    if "</think>" in content:
        content = content.split("</think>", 1)[1].strip()
    elif "<think>" in content:
        content = content.replace("<think>", "").strip()

    return {"message": [answer.model_copy(update={"content": content})]}


def router_query(state: State):
    query_type = state["query_type"]
    if query_type == "academic":
        return "academic_rag_node"
    elif query_type == "fee":
        return "fee_rag_node"
    else:
        return "general_node"


@st.cache_resource(show_spinner=False)
def build_graph():
    graph = StateGraph(State)
    graph.add_node("classifier_node", classifier_node)
    graph.add_node("academic_rag_node", academic_rag_node)
    graph.add_node("fee_rag_node", fee_rag_node)
    graph.add_node("general_node", general_node)
    graph.add_node("response_node", response_node)

    graph.add_edge(START, "classifier_node")
    graph.add_conditional_edges("classifier_node", router_query)
    graph.add_edge("academic_rag_node", "response_node")
    graph.add_edge("fee_rag_node", "response_node")
    graph.add_edge("general_node", "response_node")
    graph.add_edge("response_node", END)

    return graph.compile()


app = build_graph()


# ----------------------------------------------------------------------
# Streamlit UI
# ----------------------------------------------------------------------
st.title("🎓 College Assistant")

# --- Programme selection (once, kept in session_state) ---
if "program" not in st.session_state:
    st.session_state.program = None

if st.session_state.program is None:
    st.subheader("Welcome! Let's get you set up.")
    choice = st.radio(
        "Please select your programme of study:",
        options=["BCA", "B.Com", "BBA"],
        index=None,
    )
    if st.button("Continue", disabled=choice is None):
        st.session_state.program = choice
        st.session_state.chat_history = []
        st.rerun()
    st.stop()

# --- Sidebar: current programme + reset ---
with st.sidebar:
    st.markdown(f"**Programme:** {st.session_state.program}")
    if st.button("Change programme / Reset chat"):
        st.session_state.program = None
        st.session_state.chat_history = []
        st.rerun()

st.caption(f"You're set as a **{st.session_state.program}** student. Ask me about academics, fees, or anything else!")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# --- Render chat history ---
for role, text in st.session_state.chat_history:
    with st.chat_message(role):
        st.markdown(text)

# --- Chat input ---
user_query = st.chat_input("Type your question here...")

if user_query:
    st.session_state.chat_history.append(("user", user_query))
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            state = {
                "program": st.session_state.program,
                "message": [{"role": "human", "content": user_query}],
            }
            response = app.invoke(state)
            answer_text = response["message"][-1].content
        st.markdown(answer_text)

    st.session_state.chat_history.append(("assistant", answer_text))