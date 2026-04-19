import os
from typing import Dict, TypedDict, List
from pydantic import BaseModel, Field

from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.output_parsers import PydanticOutputParser


# --- Structured Output Schema ---
class RetentionAction(BaseModel):
    action: str = Field(description="The specific retention action to take.")
    reasoning: str = Field(description="Why this action is recommended based on the customer profile.")

class RetentionReport(BaseModel):
    risk_summary: str = Field(description="High-level summary of the customer's churn risk and profile.")
    contributing_factors: List[str] = Field(description="Key factors contributing to the churn risk.")
    recommended_actions: List[RetentionAction] = Field(description="Actionable retention recommendations.")
    supporting_sources: List[str] = Field(description="Relevant best practices or guidelines retrieved from the knowledge base.")
    business_ethical_disclaimers: str = Field(description="Important business and ethical disclaimers regarding the suggested actions.")


# --- State Management ---
class AgentState(TypedDict):
    customer_id: str
    customer_data: Dict
    churn_probability: float
    risk_level: str
    feature_importance: Dict
    missing_data_flags: List[str]
    retrieved_context: str
    final_report: RetentionReport


# --- Base Path Setup for KB ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
KB_PATH = os.path.join(BASE_DIR, "retention_kb.txt")

# Cache vector store globally
_VECTOR_STORE = None

def get_retriever():
    global _VECTOR_STORE
    if _VECTOR_STORE is None:
        if not os.path.exists(KB_PATH):
            raise FileNotFoundError(f"Knowledge base file not found at {KB_PATH}")
        
        loader = TextLoader(KB_PATH, encoding="utf-8")
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        splits = text_splitter.split_documents(docs)
        
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        _VECTOR_STORE = FAISS.from_documents(splits, embeddings)
        
    return _VECTOR_STORE.as_retriever(search_kwargs={"k": 2})


# --- Agent Nodes ---

def data_validator_node(state: AgentState) -> AgentState:
    """Explicitly checks for noisy or missing data and flags them."""
    missing_flags = []
    data = state["customer_data"]
    
    # Common telecom dataset checks
    if pd.isna(data.get("MonthlyCharges")):
        missing_flags.append("MonthlyCharges is missing. High risk of inaccurate billing predictions.")
    if pd.isna(data.get("tenure")) or data.get("tenure") == 0:
        missing_flags.append("Tenure is missing or zero. Customer may be brand new.")
        
    if "TotalCharges" in data:
        tc = data["TotalCharges"]
        if pd.isna(tc) or (isinstance(tc, str) and str(tc).strip() == ""):
            missing_flags.append("TotalCharges is blank or missing. Treating as new customer.")
            
    state["missing_data_flags"] = missing_flags
    return state


def rag_retriever_node(state: AgentState) -> AgentState:
    """Retrieves relevant best practices based on the customer profile."""
    retriever = get_retriever()
    
    # Formulate a search query based on important features and risk
    important_features = list(state["feature_importance"].keys())[:3]
    query = f"Customer has high risk due to {', '.join(important_features)}"
    
    docs = retriever.invoke(query)
    retrieved_text = "\n\n".join([d.page_content for d in docs])
    
    state["retrieved_context"] = retrieved_text
    return state


def generator_node(state: AgentState) -> AgentState:
    """Generates the structured retention strategy."""
    
    # Ensure Groq API key is set
    llm = ChatGroq(model="llama3-70b-8192", temperature=0.2)
    
    parser = PydanticOutputParser(pydantic_object=RetentionReport)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert AI Retention Strategy Assistant for a telecom company. "
                   "Your job is to reason about customer data, predicted risk, and retrieved best practices to "
                   "generate a highly targeted, structured customer retention report. "
                   "\n\n{format_instructions}"),
        ("user", "Customer ID: {cust_id}\n"
                 "Churn Probability: {prob:.2%}\n"
                 "Risk Level: {risk}\n"
                 "Customer Data: {data}\n"
                 "Top Contributing Factors: {factors}\n"
                 "Missing/Noisy Data Flags: {flags}\n\n"
                 "--- Retrieved Knowledge Base Best Practices ---\n{context}\n\n"
                 "Generate the Retention Report exactly as structured.")
    ])
    
    chain = prompt | llm | parser
    
    result = chain.invoke({
        "format_instructions": parser.get_format_instructions(),
        "cust_id": state["customer_id"],
        "prob": state["churn_probability"],
        "risk": state["risk_level"],
        "data": state["customer_data"],
        "factors": state["feature_importance"],
        "flags": "\n".join(state["missing_data_flags"]) if state["missing_data_flags"] else "None",
        "context": state["retrieved_context"]
    })
    
    state["final_report"] = result
    return state


# --- Agent Builder ---

def run_retention_agent(
    customer_id: str,
    customer_data: Dict,
    churn_probability: float,
    risk_level: str,
    feature_importance: Dict
) -> RetentionReport:
    """Runs the LangGraph-style agent loop directly."""
    import pandas as pd # Needed in data_validator_node
    
    # Initialize state
    state: AgentState = {
        "customer_id": customer_id,
        "customer_data": customer_data,
        "churn_probability": churn_probability,
        "risk_level": risk_level,
        "feature_importance": feature_importance,
        "missing_data_flags": [],
        "retrieved_context": "",
        "final_report": None
    }
    
    # Execute explicit workflow graph
    state = data_validator_node(state)
    state = rag_retriever_node(state)
    state = generator_node(state)
    
    return state["final_report"]
