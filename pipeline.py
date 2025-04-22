from inputValidatorAgent import InputValidator
from sqlAgent import SQLAgent
import final_retriever
from langgraph.graph import StateGraph, MessagesState, END, START
from langgraph.prebuilt import tools_condition, ToolNode
from IPython.display import Image, display
from langchain_core.runnables.graph import CurveStyle, MermaidDrawMethod, NodeStyles
from pprint import pprint
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
import os
import lrucache
from cachetools import LRUCache
import lrucache

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

class AgentState(MessagesState):
    question: str
    context: str
    db_url: str
    validation_result: str
    sql_result: str
    final_result: str

def validate_input_wrapper(state: AgentState):
    """
    Validates the input question against the provided context.
    """
    question = state["question"]
    context = state["context"]
    validator = InputValidator(question, context)
    value = validator.validate()
    return {"validation_result": value}

def execute_sql_wrapper(state: AgentState):
    """
    Executes the SQL query based on the input question.
    """
    question = state["question"]
    agent = SQLAgent(state["db_url"], question)
    return {"sql_result": agent.execute_query()}

def retrieve_wrapper(state: AgentState):
    """
    Retrieves the SQL query based on the input question.
    """
    question = state["question"]
    documents = state["sql_result"]
    
    result = final_retriever.final_retrieval(question, documents)
    
    return {"final_result": result}


def build_workflow():
    workflow = StateGraph(AgentState)

    workflow.add_node('inputValidatorAgent', validate_input_wrapper)
    workflow.add_node('sqlAgent', execute_sql_wrapper)
    workflow.add_node('retrieve', retrieve_wrapper)

    workflow.add_edge(START, 'inputValidatorAgent')
    workflow.add_conditional_edges(
        'inputValidatorAgent',
        lambda state: state["validation_result"],
        {
            "Yes": 'sqlAgent',
            "No": END,
        }
    )

    workflow.add_edge('sqlAgent', 'retrieve')
    workflow.add_edge("retrieve",END)

    return workflow.compile()

def run_search_pipeline(query: str):
    embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
    vector_db = FAISS.load_local("vector_db", embedding_model, allow_dangerous_deserialization=True)
    results = vector_db.similarity_search(query, k=2)
    context = "\n\n".join([doc.page_content for doc in results])


    inputs = {
        "question": query,
        "context": context,
        "db_url": f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{dbname}",
    }

    if query in lrucache.cache:
        print("Cache hit!")
        return lrucache.cache[query].get("final_result", "No relevant products found.")
    
    app_workflow = build_workflow()
    final_result = {}
    for output in app_workflow.stream(inputs):
        final_result.update(output)

    final_result = final_result.get("retrieve", {})
    lrucache.cache[query] = final_result
    
    return final_result.get("final_result", "No relevant products found.")