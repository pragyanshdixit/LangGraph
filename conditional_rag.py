import os
import re
from unicodedata import category
from dotenv import load_dotenv
from typing import Annotated, TypedDict
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph , START, END
from sympy import content
load_dotenv()

embeddings=HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

def retriever_node(path_url:str):
    """state 1: Loads a PDF document from the given path or URL, splits it into chunks, and creates a FAISS vector store for retrieval."""
    
    loader=PyPDFLoader(path_url)
    documents=loader.load()
    
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
    
    chunks=text_splitter.split_documents(documents)
    
    vectorstore=FAISS.from_documents(chunks,embeddings)
    
    return vectorstore.as_retriever(search_kwargs={"k":4})

academic_retriever=retriever_node("academics_handbook.pdf")
fee_retriever=retriever_node("fee_structure.pdf")

model = ChatGroq(model="openai/gpt-oss-20b", temperature=0.4,api_key=os.getenv("GROQ_API_KEY"))

class State(TypedDict):
    program: str
    message: Annotated[list,add_messages]
    query_type:str
    retrieved_context:str
    
def classifier_node(state:State):
    """"Look at the latest user query and decide which path to take"""
    last_message=state["message"][-1].content
    
    prompt=(
        "Classify the following student query into exactly one category: " "'academic', 'fee', or 'general'.\n\n" 
        "Use 'academic' for questions about attendance, exams, grading, credits, " 
        "promotion, course structure, summer training, or degree requirements.\n" 
        "Use 'fee' for questions about tuition, payment, refund, late charges, "
        "scholarships, or any money-related topic.\n" 
        "Use 'general' for greetings, casual talk, or anything not related to " 
        "the college rules or fee.\n\n" 
        f"Query: {last_message}\n\n" 
        "Return only one word: academic, fee, or general."
    )
    response=model.invoke(prompt)
    labels = re.findall(r"\b(academic|fee|general)\b", response.content.lower())
    category = labels[-1] if labels else "general"
        
    return {"query_type": category}

def academic_rag_node(state:State):
    """Retrieves relevant chunks from the academics handbook."""
    query=state["message"][-1].content
    docs=academic_retriever.invoke(query)
    context="\n\n".join([doc.page_content for doc in docs])
    print("Academic content retrieved\n")
    return {r"retrieved_context": context}

def fee_rag_node(state:State):
    """Retrieves relevant chunks from the fee structure document."""
    query=state["message"][-1].content
    docs=fee_retriever.invoke(query)
    context="\n\n".join([doc.page_content for doc in docs])
    print("Fee content retrieved\n")
    return {r"retrieved_context": context}
def general_node(state:State):
    """Handles general queries that do not require retrieval."""
    print("General query handled\n")
    return {r"retrieved_context": "NO_RETRIEVAL_NEEDED"}

def response_node(state:State):
    """Generates the final answer, personalized using the student's programme."""
    
    query=state["message"][-1].content
    programme=state.get("program","unknown")
    context=state["retrieved_context"]
    
    if context=="NO_RETRIEVAL_NEEDED":
        prompt=(
            f"You are a friendly college assistant talking to a {programme} student. "
            f"Answer this question using your own general knowledge:\n\n{query}" 
            )
    else:
        prompt = ( f"You are a college assistant helping a {programme} student. " 
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

def router_query(state:State):
    """It will select the appropriate retrieval path based on the classified query type."""
    
    query_type=state["query_type"]
    
    if query_type=="academic":
        return "academic_rag_node"
    elif query_type=="fee":
        return "fee_rag_node"
    else:
        return "general_node"
    
    
graph=StateGraph(State)

graph.add_node("classifier_node",classifier_node)
graph.add_node("academic_rag_node",academic_rag_node)
graph.add_node("fee_rag_node",fee_rag_node)
graph.add_node("general_node",general_node)
graph.add_node("response_node",response_node)

graph.add_edge(START,"classifier_node")

graph.add_conditional_edges("classifier_node",router_query)

graph.add_edge("academic_rag_node","response_node")
graph.add_edge("fee_rag_node","response_node")
graph.add_edge("general_node","response_node")

graph.add_edge("response_node",END)

app=graph.compile()

print("Welcome to the college assistant! how can I help you today?\n")
category=input("Please enter your programme of study[1:BCA, 2:MCA, 3:BBA]: ")

if category=="1":
    category="BCA"
elif category=="2":
    category="B.Com"
elif category=="3":
    category="BBA"
    
print(f"\nGreat! You're set as a {category} student.\n")

while True:
    user_query=input("You: ")
    
    if user_query.lower() in ["exit","quit"]:
        print("Goodbye! Have a great day!")
        break
    
    state={"program":category,"message":[{"role":"human","content":user_query}]}
    
    response=app.invoke(state)
    
    print(f"Assistant:\n{response['message'][-1].content}")
    
    
    
    
    


