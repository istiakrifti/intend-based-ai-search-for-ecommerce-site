import os
import json
import numpy as np
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.schema import Document
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_community.docstore.in_memory import InMemoryDocstore
import psycopg2
import faiss
import pandas as pd

load_dotenv()
# os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

dbname = os.environ.get("DB_NAME")
user = os.environ.get("DB_USER")    
password = os.environ.get("DB_PASSWORD")
host = os.environ.get("DB_HOST")    
port = os.environ.get("DB_PORT")

conn = psycopg2.connect(
    dbname=dbname,
    user=user,
    password=password,
    host=host,
    port=port
)
def prepare_document(product_id):
    query = f"""
    SELECT 
        p.id, p.name, p.base_price, p.discount, p.rating, 
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
    WHERE p.id = {product_id}
    GROUP BY p.id;
    """

    df = pd.read_sql(query, conn)

    if df.empty:
        print(f"No product found with ID: {product_id}")
        return None

    row = df.iloc[0]
    specs = row['specs']
    spec_text = "; ".join(
        f"{k}: {v}" for d in specs for k, v in d.items()
    )
    content = (
        f"The {row['name']} by {row['brand']} is a premium offering in the {row['category']} > "
        f"{row['subcategory']} segment. Priced at ${row['base_price']}, it is currently available "
        f"at a discount of {row['discount']}%. This product has received an average customer rating "
        f"of {row['rating']} stars.\n\n"
        
        f"Product Specifications:\n{spec_text}\n\n"

        f"Availability Status: {'In stock' if row['stock'] > 0 else 'Temporarily unavailable'}."
    )

    metadata = {
        "id": int(row["id"]) if isinstance(row["id"], (np.integer, int)) else row["id"],
        "name": row["name"],
        "brand": row["brand"],
        "price": float(row["base_price"]),
        "discount": float(row["discount"]),
        "rating": float(row["rating"]),
        "category": row["category"],
        "subcategory": row["subcategory"],
        "stock": int(row["stock"]) if isinstance(row["stock"], (np.integer, int)) else row["stock"],
        "specs": row['specs'],
        "image_urls": row["image_urls"] 
    }

    print(metadata)

    return Document(page_content=content, metadata=metadata)

async def upsert_product(batch):
    embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
    vector_db = FAISS.load_local("vector_db", embedding_model, allow_dangerous_deserialization=True)

    new_docs = []
    new_ids = []
    to_delete = []

    existing_ids = set() 
    try:
        existing_ids = set(vector_db.index_to_docstore_id.values())
    except Exception as e:
        print("Could not load existing vector DB index:", e)

    for item in batch:
        action = item["action"]
        product_id = str(item["product_id"])

        if action == "DELETE":
            to_delete.append(product_id)

        elif action == "INSERT":
            if product_id in existing_ids:
                print(f"Skipping INSERT for product {product_id} - already exists in vector DB.")
                continue
            doc = prepare_document(product_id)
            if doc:
                new_docs.append(doc)
                new_ids.append(product_id)
            else:
                print(f"INSERT failed: no data found for {product_id}")

        elif action == "UPDATE":
            to_delete.append(product_id)
            doc = prepare_document(product_id)
            if doc:
                new_docs.append(doc)
                new_ids.append(product_id)
            else:
                print(f"UPDATE failed: no data found for {product_id}")

    # Delete old entries
    if to_delete:
        try:
            vector_db.delete(ids=to_delete)
            print(f"Deleted {len(to_delete)} product(s)")
        except Exception as e:
            print(f"Failed to delete product(s): {e}")

    # Add new documents
    if new_docs:
        try:
            vector_db.add_documents(documents=new_docs, ids=new_ids)
            print(f"Added/Updated {len(new_docs)} product(s)")
        except Exception as e:
            print(f"Failed to add documents: {e}")

    # Save final state
    vector_db.save_local("vector_db")
    conn.close()
