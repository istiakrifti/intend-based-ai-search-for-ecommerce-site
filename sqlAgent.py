from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatOllama
from langchain import SQLDatabase
from langchain.chains import create_sql_query_chain
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain.chains import LLMChain
from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from sqlalchemy import create_engine
import pandas as pd
import psycopg2
import os
import json

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
dbname = os.environ.get("DB_NAME")
user = os.environ.get("DB_USER")    
password = os.environ.get("DB_PASSWORD")
host = os.environ.get("DB_HOST")    
port = os.environ.get("DB_PORT")

class SQLAgent:
    def __init__(self, db_url, question):
        self.db_url = db_url
        self.question = question

    def format_generator(self, result):
        # llm_for_formation = ChatOpenAI(model="gpt-4o", response_format={"type": "json_object"},temperature=0.7)

        # formation_prompt = PromptTemplate(
        #     input_variables=["sql_query", "result"],
        #     template="""

        #     You are a helpful assistant. Given the following SQL query result, format it into a JSON object with the necessary column names as keys and their corresponding values, following the exact order of the columns as they appear in the SQL query.
        #     SQL Query: {sql_query}
        #     Result: {result}
        #     Ensure the JSON object is well-structured and includes all relevant columns from the result in the same order as in the SQL query. The columns are:
        #     - id
        #     - name
        #     - base_price
        #     - discount
        #     - rating        
        #     - category
        #     - subcategory   
        #     - brand
        #     - stock
        #     - image_urls
        #     All the produc's info will be a json object with the keys as mentioned above.
        #     """    
        # )

        # chain_for_formation = LLMChain(
        #     llm=llm_for_formation,
        #     prompt=formation_prompt,
        #     output_parser=JsonOutputParser(),
        # )

        # generated_result = chain_for_formation.invoke({
        #     "sql_query": query_result,
        #     "result": result,
        # })

        
        # final_result = None
        # try:
        #     final_result = generated_result['result']
        # except Exception as e:
        #     return None

        formatted_result = []

        for idx, row in result.iterrows():
            # Generate the 'id' in the format sql_1, sql_2, ...
            doc_id = f"sql_{idx + 1}"
        
            content = (
                f"The {row['name']} by {row['brand']} is a premium offering in the {row['category']} > "
                f"{row['subcategory']} segment. Priced at ${row['base_price']}, it is currently available "
                f"at a discount of {row['discount']}%. This product has received an average customer rating "
                f"of {row['rating']} stars.\n\n"
                f"Product Specifications:\n{row['specs']}\n\n"
                f"Availability Status: {'In stock' if row['stock'] > 0 else 'Temporarily unavailable'}."
            )

            # Prepare the metadata (current row)
            metadata = row.to_dict()  # Use the current dictionary as metadata

            # Set the distance to 1 (as per the requirement)
            distance = 1

            # Create the formatted document structure
            formatted_result.append({
                'id': doc_id,
                'document': content,
                'metadata': metadata,
                'distance': distance
            })
        
        return formatted_result

    def execute_query(self):
        db = SQLDatabase.from_uri(self.db_url)

        conn = psycopg2.connect(
            dbname=dbname,
            user=user,
            password=password,
            host=host,
            port=port
        )

        llm = ChatOpenAI(model='gpt-4o', response_format={"type": "json_object"}, temperature=0)
        
        template="""You are a {dialect} expert. Given an input question, create a syntactically correct {dialect} query to run.
            Unless the user specifies in the question a specific number of examples to obtain, query for at most {top_k} results using the LIMIT clause as per {dialect}. You can order the results to return the most informative data in the database.
            Try to understand what kind of products the user is looking for and write query to fetch all the columns related to the products. To fetch all the products you can use this at the first
            SELECT p.id, p.name, p.base_price, p.discount, p.rating, 
            p.category, p.subcategory, p.brand, p.stock,
    
            COALESCE(
                json_agg(DISTINCT jsonb_build_object(s.attr_name, s.attr_value)) 
                FILTER (WHERE s.attr_name IS NOT NULL), '[]'
            ) AS specs,

            COALESCE(
                array_agg(DISTINCT i.img_url) FILTER (WHERE i.img_url IS NOT NULL), 
                ARRAY[]::VARCHAR[]
            ) AS image_urls

            FROM products p
            LEFT JOIN spec_table s ON p.id = s.product_id
            LEFT JOIN images i ON p.id = i.product_id
            GROUP BY p.id

            then you can use the WHERE clause to filter the products based on the user query.
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
        # prompt.pretty_print()
        
        retry_attempts = 2
        query_result = None
        result = None
        error_message = None

        for attempt in range(retry_attempts):
            template_with_error = template

            if error_message:
                template_with_error = f"{error_message} \n\n {template}"
            prompt = ChatPromptTemplate.from_template(template_with_error)
            prompt.pretty_print()
            chain = create_sql_query_chain(llm, db, prompt=prompt) | JsonOutputParser()
            query = chain.invoke(
                {
                    "question": self.question,
                    "dialect": db.dialect,
                    "top_k": 10,
                    "table_info": db.get_table_info(),
                }
            )
            
            try:
                if not query['final query']:
                    raise ValueError("The final query is missing in the json response.")
                query_result = query['final query']
                print(f"Query: {query_result}")
                # result = db.run(query_result)
                df = pd.read_sql(query_result, conn)
                # print(df.head(1))
                # result = df.to_json(orient='records') 
                # parsed_json = json.loads(result)
                # result = json.dumps(parsed_json, ensure_ascii=False)
                result = df
                # print(result)
                break
            except Exception as e:
                error_message = f"Please solve the error. The error message is: {str(e)}"
                print(error_message)

        if result.empty:
            return None
        
        conn.close()
        formatted_result = self.format_generator(result)

        print("-----------------------------------")
        print(f"Formatted Result: {formatted_result}")
        print("-----------------------------------")
        return formatted_result  
    
# returning None if the query is not valid or the result is empty, otherwise returning the result in JSON format.