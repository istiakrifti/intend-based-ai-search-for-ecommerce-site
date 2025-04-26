from inputValidatorAgent import InputValidator
from sqlAgent import SQLAgent
import final_retriever
from langgraph.graph import StateGraph, MessagesState, END, START
from langchain_core.runnables.graph import CurveStyle, MermaidDrawMethod, NodeStyles
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.runnables import RunnableParallel, RunnableLambda
import os
import lrucache

# ---------- ENV SETUP ----------
dbname = os.environ.get("DB_NAME")
user = os.environ.get("DB_USER")    
password = os.environ.get("DB_PASSWORD")
host = os.environ.get("DB_HOST")    
port = os.environ.get("DB_PORT")

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["LANGSMITH_TRACING"] = os.getenv("LANGSMITH_TRACING")
os.environ["LANGSMITH_ENDPOINT"] = os.getenv("LANGSMITH_ENDPOINT")
os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY")
os.environ["LANGSMITH_PROJECT"] = os.getenv("LANGSMITH_PROJECT")
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

# ---------- STATE ----------
class AgentState(MessagesState):
    question: str
    # context: str
    db_url: str
    validation_result: str
    sql_result: str
    rag_result: str
    final_result: str

# ---------- AGENT NODES ----------
def validate_input_wrapper(state: AgentState):
    question = state["question"]
    # context = state["context"]
    validator = InputValidator(question)
    value = validator.validate()
    return {"validation_result": value}

def parallel_sql_and_retriever():
    return RunnableParallel({
        "sql_result": RunnableLambda(lambda state: SQLAgent(state["db_url"], state["question"]).execute_query()),
        "rag_result": RunnableLambda(lambda state: final_retriever.final_retrieval(state["question"]))
    })

def rerank_wrapper(state: AgentState):
    question = state["question"]
    documents = state["sql_result"]
    retrieved_docs = state["rag_result"]
    reranked_docs = final_retriever.rerank(question, documents, retrieved_docs) 
    return {"final_result": reranked_docs}

# ---------- WORKFLOW ----------
def build_workflow():
    workflow = StateGraph(AgentState)

    workflow.add_node("inputValidatorAgent", validate_input_wrapper)
    workflow.add_node("parallel_node", parallel_sql_and_retriever())
    workflow.add_node("rerank", rerank_wrapper)

    workflow.add_edge(START, "inputValidatorAgent")
    workflow.add_conditional_edges(
        "inputValidatorAgent",
        lambda state: state["validation_result"],
        {
            "Yes": "parallel_node",
            "No": END,
        }
    )
    workflow.add_edge("parallel_node", "rerank")
    workflow.add_edge("rerank", END)

    return workflow.compile()

# ---------- PIPELINE RUNNER ----------
def run_search_pipeline(query: str):
    # embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
    # vector_db = FAISS.load_local("vector_db", embedding_model, allow_dangerous_deserialization=True)
    # results = vector_db.similarity_search(query, k=2)
    # context = "\n\n".join([doc.page_content for doc in results])

    inputs = {
        "question": query,
        # "context": context,
        "db_url": f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{dbname}",
    }

    if query in lrucache.cache:
        print("Cache hit!")
        return lrucache.cache[query].get("final_result", "No relevant products found.")

    app_workflow = build_workflow()
    final_result = {}
    for output in app_workflow.stream(inputs):
        final_result.update(output)

    result_only = final_result.get("rerank", {})
    lrucache.cache[query] = result_only

    return result_only.get("final_result", "No relevant products found.")
