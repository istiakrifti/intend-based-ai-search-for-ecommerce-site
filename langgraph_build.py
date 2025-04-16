from inputValidatorAgent import InputValidator
from sqlAgent import SQLAgent
from langgraph.graph import StateGraph, MessagesState, END, START
from langgraph.prebuilt import tools_condition, ToolNode
from IPython.display import Image, display
from langchain_core.runnables.graph import CurveStyle, MermaidDrawMethod, NodeStyles
from pprint import pprint
from langchain_core.messages import BaseMessage, HumanMessage
import os

dbname = os.environ.get("DB_NAME")
user = os.environ.get("DB_USER")    
password = os.environ.get("DB_PASSWORD")
host = os.environ.get("DB_HOST")    
port = os.environ.get("DB_PORT")

class AgentState(MessagesState):
    question: str
    context: str
    db_url: str
    validation_result: str
    sql_result: str


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


workflow = StateGraph(AgentState)

workflow.add_node('inputValidatorAgent', validate_input_wrapper)
workflow.add_node('sqlAgent', execute_sql_wrapper)


workflow.add_edge(START, 'inputValidatorAgent')
workflow.add_conditional_edges(
    'inputValidatorAgent',
    lambda state: state["validation_result"],
    {
        "Yes": 'sqlAgent',
        "No": END,
    }
)

workflow.add_edge('sqlAgent', END)

inputs = {
    "question": "Which products have the highest price?",
    "context": "This is a product database. It contains information about various products, including their names, prices, and categories.",
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