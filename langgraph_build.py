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
    return result

# def generate_wrapper(state: AgentState):
#     """
#     Generates the SQL query based on the input question.
#     """
#     question = state["question"]
#     documents = state["documents"]
#     agent = state["ragagent"]
#     result = agent.generate(question, documents)
#     print("-------------------------------------------------")
#     print("result", result)
#     return {"generation": result['generation']}

# def grade_generation_v_documents_and_question_wrapper(state: AgentState):
#     """
#     Grades the generated SQL query against the input question and context.
#     """
#     question = state["question"]
#     generation = state["generation"]
#     documents = state["documents"]
#     agent = state["ragagent"]
#     print("-------------------------------------------------")
#     result = agent.grade_generation_v_documents_and_question(generation, documents, question)
#     print("result", result)
#     return result


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
# workflow.add_conditional_edges(
#     "generate",
#     grade_generation_v_documents_and_question_wrapper,
#     {
#         "not supported": "generate",
#         "useful": END,
#         "not useful": "generate",
#     }
# )

question = "gaming laptop with good graphics within price 88500 taka"
embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
vector_db = FAISS.load_local("vector_db", embedding_model, allow_dangerous_deserialization=True)
results = vector_db.similarity_search(question, k=2)

# Extract page_content from each document and create the context
context = "\n\n".join([doc.page_content for doc in results])
inputs = {
    "question": question,
    "context": context,
    "db_url": f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{dbname}",
}

app = workflow.compile()

for output in app.stream(inputs):
    for key, value in output.items():
        print(f"Finished running: {key}:")
        print(value)

# mermaid_code = app.get_graph().draw_mermaid()

# # Save the Mermaid code to a .mmd file
# with open("workflow_graph.mmd", "w") as f:
#     f.write(mermaid_code)