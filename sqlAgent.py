import logging
from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatOllama
from langchain_community.utilities import SQLDatabase
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
import lrucache

# ------------------- SETUP -------------------
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
dbname = os.environ.get("DB_NAME")
user = os.environ.get("DB_USER")
password = os.environ.get("DB_PASSWORD")
host = os.environ.get("DB_HOST")
port = os.environ.get("DB_PORT")

# ------------------- LOGGER SETUP -------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    filename="sql_agent.log",
    filemode="a"
)

logger = logging.getLogger(__name__)

# ------------------- AGENT CLASS -------------------
class SQLAgent:
    def __init__(self, db_url, question):
        self.db_url = db_url
        self.question = question
        logger.info(f"SQLAgent initialized with question: '{self.question}'")

    def format_generator(self, result):
        logger.info("Formatting query results into structured documents.")

        formatted_result = []
        for idx, row in result.iterrows():
            doc_id = f"sql_{idx + 1}"

            content = (
                f"The {row['name']} by {row['brand']} is a premium offering in the {row['category']} > "
                f"{row['subcategory']} segment. Priced at ${row['base_price']}, it is currently available "
                f"at a discount of {row['discount']}%. This product has received an average customer rating "
                f"of {row['rating']} stars.\n\n"
                f"Product Specifications:\n{row['specs']}\n\n"
                f"Availability Status: {'In stock' if row['stock'] > 0 else 'Temporarily unavailable'}."
            )

            metadata = row.to_dict()

            formatted_result.append({
                'id': doc_id,
                'document': content,
                'metadata': metadata,
                'distance': 1
            })

        logger.info(f"Generated {len(formatted_result)} formatted results.")
        return formatted_result

    def execute_query(self):
        logger.info("Starting query execution process.")

        try:
            db = SQLDatabase.from_uri(self.db_url)
            conn = psycopg2.connect(
                dbname=dbname,
                user=user,
                password=password,
                host=host,
                port=port
            )
            logger.info("Database connection established.")
        except Exception as e:
            logger.error(f"Failed to connect to the database: {e}")
            return None

        llm = ChatOpenAI(model='gpt-4o', response_format={"type": "json_object"}, temperature=0)

        template = """You are a {dialect} expert. Given an input question, create a syntactically correct {dialect} query to run.
            Unless the user specifies in the question a specific number of examples to obtain, query for at most {top_k} results using the LIMIT clause as per {dialect}. You can order the results to return the most informative data in the database.
            Try to understand what kind of products the user is looking for and write query to fetch all the columns related to the products.

            The query structure should be based on:
            SELECT p.id, p.name, p.base_price, p.discount, p.rating,
                p.category, p.subcategory, p.brand, p.stock,
                COALESCE(json_agg(DISTINCT jsonb_build_object(s.attr_name, s.attr_value)) 
                            FILTER (WHERE s.attr_name IS NOT NULL), '[]') AS specs,
                COALESCE(array_agg(DISTINCT i.img_url) FILTER (WHERE i.img_url IS NOT NULL), ARRAY[]::VARCHAR[]) AS image_urls
            FROM products p
            LEFT JOIN spec_table s ON p.id = s.product_id
            LEFT JOIN images i ON p.id = i.product_id
            WHERE ...
            GROUP BY p.id
            ORDER BY ...
            LIMIT {top_k};

            Pay attention to use only the column names you can see in the tables below. Be careful to not query for columns that do not exist. Also, pay attention to which column is in which table.
            Pay attention to use date('now') function to get the current date, if the question involves "today".

            Only use the following tables:
            {table_info}

            Question: {input}

            Write a query and check the {dialect} query for common mistakes, including:
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

            Please return final query in JSON format with key'final query'.
        """

        

        retry_attempts = 2
        query_result = None
        result = None
        error_message = None

        table_info = db.get_table_info(table_names=["products", "spec_table", "images"])
        # logger.info(f"Table information retrieved: {table_info}")
        for attempt in range(retry_attempts):
            logger.info(f"Query generation attempt {attempt + 1}/{retry_attempts}")

            template_with_error = template
            if error_message:
                template_with_error = f"{error_message} \n\n {template}"

            try:
                prompt = ChatPromptTemplate.from_template(template_with_error)
    
                chain = create_sql_query_chain(llm, db, prompt=prompt) | JsonOutputParser()

                query = chain.invoke({
                    "question": self.question,
                    "dialect": db.dialect,
                    "top_k": 6,
                    "table_info": table_info
                })

                # Log initial and final SQL queries
                logger.info(f"Initial SQL query draft:\n{query.get('initial query', 'N/A')}")
                logger.info(f"Final SQL query:\n{query.get('final query', 'N/A')}")

                if not query['final query']:
                    raise ValueError("Missing 'final query' in the response.")

                query_result = query['final query']

                if query_result in lrucache.cache:
                    print("Cache hit! Returning cached result.")
                    return lrucache.cache[query_result]

                df = pd.read_sql(query_result, conn)

                if df.empty:
                    logger.warning("Query executed successfully but returned no results.")
                    return None

                result = df
                break

            except Exception as e:
                error_message = f"Error during query generation or execution: {str(e)}"
                logger.error(error_message)

        conn.close()
        logger.info("Database connection closed.")

        if result is not None:
            formatted_result = self.format_generator(result)
            # key = lrucache.get_normalized_cache_key(query_result)
            lrucache.cache[query_result] = formatted_result
            logger.info("Returning formatted result.")
            logger.info("Formatted Result:\n%s", json.dumps(formatted_result, indent=2))
            return formatted_result
        else:
            logger.warning("No result to format; returning None.")
            return None
