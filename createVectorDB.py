import os
import json
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.schema import Document
from dotenv import load_dotenv
import psycopg2
import pandas as pd
from constants import *

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# dbname = os.environ.get("DB_NAME")
# user = os.environ.get("DB_USER")    
# password = os.environ.get("DB_PASSWORD")
# host = os.environ.get("DB_HOST")    
# port = os.environ.get("DB_PORT")

# conn = psycopg2.connect(
#     dbname=dbname,
#     user=user,
#     password=password,
#     host=host,
#     port=port
# )

# query = """
# SELECT 
#     p.id, p.name, p.base_price, p.discount, p.rating, 
#     p.category, p.subcategory, p.brand, p.stock,
    
#     COALESCE(
#         json_agg(DISTINCT jsonb_build_object(s.attr_name, s.attr_value)) 
#         FILTER (WHERE s.attr_name IS NOT NULL), '[]'
#     ) AS specs,

#     COALESCE(
#         array_agg(DISTINCT i.img_url) FILTER (WHERE i.img_url IS NOT NULL), 
#         ARRAY[]::VARCHAR[]
#     ) AS image_urls

# FROM products p
# LEFT JOIN spec_table s ON p.id = s.product_id
# LEFT JOIN images i ON p.id = i.product_id
# GROUP BY p.id;
# """

# df = pd.read_sql(query, conn)
# conn.close()

# docs = []
# for _, row in df.iterrows():
#     specs = row['specs']
#     spec_text = "; ".join(
#         f"{k}: {v}" for d in specs for k, v in d.items()
#     )

#     content = (
#         f"{row['name']} by {row['brand']} - "
#         f"{row['category']}/{row['subcategory']}, "
#         f"Price: ${row['base_price']} (Discount: {row['discount']}%), "
#         f"Rating: {row['rating']} stars. "
#         f"Specs: {spec_text}"
#     )

#     metadata = {
#         "id": row["id"],
#         "name": row["name"],
#         "brand": row["brand"],
#         "price": float(row["base_price"]),
#         "discount": float(row["discount"]),
#         "rating": float(row["rating"]),
#         "category": row["category"],
#         "subcategory": row["subcategory"],
#         "stock": row["stock"],
#         "image_urls": row["image_urls"] 
#     }

#     docs.append(Document(page_content=content, metadata=metadata))
#     docs.append(Document(page_content=content))

embedding_model = OpenAIEmbeddings(model=OPENAI_EMBEDDING_MODEL)
# vector_db = FAISS.from_documents(docs, embedding_model)
# vector_db.save_local("vector_db")
vector_db = FAISS.load_local("vector_db", embedding_model, allow_dangerous_deserialization=True)

query = "289000"
results = vector_db.similarity_search(query, k=5)
for idx, doc in enumerate(results, 1):
    print(doc)
    print(f"Result {idx} Price: {doc.metadata['price']}")

