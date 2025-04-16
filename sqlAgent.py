from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatOllama
from langchain import SQLDatabase
from langchain.chains import create_sql_query_chain
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain.chains import LLMChain
from dotenv import load_dotenv
import os
import json

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

class SQLAgent:
    def __init__(self, db_url, question):
        self.db_url = db_url
        self.question = question

    def execute_query(self):
        db = SQLDatabase.from_uri(self.db_url)

        llm = ChatOpenAI(model='gpt-4o', response_format={"type": "json_object"}, temperature=0)

        prompt = PromptTemplate(
            input_variables=["question", "dialect", "top_k", "table_info"],
            template="""You are a {dialect} expert. Given an input question, create a syntactically correct {dialect} query to run.
            Unless the user specifies in the question a specific number of examples to obtain, query for at most {top_k} results using the LIMIT clause as per {dialect}. You can order the results to return the most informative data in the database.
            Try to understand what kind of products the user is looking for and write query to fetch all the columns related to the products. I am mentioning the columns below for your reference. 

            The columns are:
            - id
            - name
            - base_price
            - discount
            - rating        
            - category
            - subcategory   
            - brand
            - stock
            - image_urls

            Pay attention to use only the column names you can see in the tables below. Be careful to not query for columns that do not exist. Also, pay attention to which column is in which table.
            Pay attention to use date('now') function to get the current date, if the question involves "today".

            Only use the following tables:
            {table_info}

            Question: {input}

            Write an initial draft of the query. Then double check the {dialect} query for common mistakes, including:
            - Using NOT IN with NULL values
            - Using UNION when UNION ALL should have been used
            - Using BETWEEN for exclusive ranges
            - Data type mismatch in predicates
            - Properly quoting identifiers
            - Using the correct number of arguments for functions
            - Casting to the correct data type
            - Using the proper columns for joins
            - LIMIT clause for the number of results

            Use format:

            Please return the initial draft and final query in JSON format with keys 'initial query' and 'final query'.

            """
        )

        chain = create_sql_query_chain(llm, db, prompt=prompt) | JsonOutputParser()
        prompt.pretty_print()

        query = chain.invoke(
            {
                "question": self.question,
                "dialect": db.dialect,
                "top_k": 5,
                "table_info": db.get_table_info(),
            }
        )

        query_result = query['final query']
        result = db.run(query_result)

        llm_for_formation = ChatOpenAI(model="gpt-3.5-turbo", response_format={"type": "json_object"},temperature=0.7)

        formation_prompt = PromptTemplate(
            input_variables=["sql_query", "result"],
            template="""
            
            You are a helpful assistant. Given the following SQL query result, format it into a JSON object with the necessary column names as keys and their corresponding values, following the exact order of the columns as they appear in the SQL query.
            SQL Query: {sql_query}
            Result: {result}
            Ensure the JSON object is well-structured and includes all relevant columns from the result in the same order as in the SQL query. The columns are:
            - id
            - name
            - base_price
            - discount
            - rating        
            - category
            - subcategory   
            - brand
            - stock
            - image_urls
            """

            
        )

        chain_for_formation = LLMChain(
            llm=llm_for_formation,
            prompt=formation_prompt,
            output_parser=JsonOutputParser(),
        )

        generated_result = chain_for_formation.invoke(
            {
                "sql_query": query_result,
                "result": result,
            }
        )
        
        # formatted_result = "\n".join(
        #     [f"{key}: {value}" for item in generated_result for key, value in item.items()]
        # )

        return generated_result


# if __name__ == "__main__":
#     db_url = "sqlite:///Product.db"
#     question = "Which products are in stock?"

#     sql_agent = SQLAgent(db_url, question)
#     result = sql_agent.execute_query() 
#     print(result)