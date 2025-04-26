import os
import json
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

query = """
SELECT * FROM products;
"""

df = pd.read_sql(query, conn)
# df = pd.read_csv('cleaned_products')
conn.close()

# # print(df.columns)
# # print(df.shape)
# # df = df.head(5000)

# # df.to_csv("products.csv", index=False)

docs = []
for _, row in df.iterrows():
    
    content = (
        f"Title: {row['title']}\n"
        f"Brand: {row['brand']}\n"
        f"Description: {row['description']}\n"
        f"specs: {row['specs']}\n"
    )

    metadata = {
        "id": row["product_id"],
        "title": row["title"],
        "brand": row["brand"],
        "price": float(row["price"]),
        "description": row["description"],
        "specs": row["specs"],
    }

    docs.append(Document(page_content=content, metadata=metadata))
#     # docs.append(Document(page_content=content))

# llm = ChatGroq(
#     model="llama3-70b-8192",
#     temperature=0,
# )

# template = """
# Product details: {product_text}

# You are an expert product analyst. Based on the provided product data, write a detailed and unique product description that highlights its ideal use cases, technical capabilities, and target users. Focus on specific scenarios where this product performs well.

# Incorporate terms and phrases users are likely to search for — such as:

# "low configuration laptop for students"

# "best phone for content creators"

# "budget-friendly DSLR for beginners"

# "gaming PC under budget"

# "laptop for programming and coding"

# "mobile with good battery backup and camera"

# Use your domain knowledge to identify important features and constraints. Do not use generic or repetitive descriptions across products. Instead, tailor the description uniquely to the product's specs, performance range, price point, and category (e.g., laptop, phone, PC, camera, etc.).

# Do not include any headings, labels, or markdown — only output the product description text. Write naturally but clearly so the description can be stored in a vector database and matched via semantic search.
# """

# prompt = ChatPromptTemplate.from_messages([
#     ("system", "You are a product analyst assistant writing detailed, SEO-friendly descriptions."),
#     ("human", template),
# ])

# chain = prompt | llm

# # Invoke the chain with a sample doc
# response = chain.invoke({"product_text": docs[1].page_content})
# print(response.content)
# print(docs[0].page_content)


embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
index = faiss.IndexFlatL2(len(OpenAIEmbeddings().embed_query("hello world")))
vector_db = FAISS(
    embedding_function=OpenAIEmbeddings(),
    index=index,
    docstore= InMemoryDocstore(),
    index_to_docstore_id={}
)

ids = [str(row["product_id"]) for _, row in df.iterrows()]

vector_db.add_documents(documents=docs, ids=ids)
# # vector_db = FAISS.from_documents(docs, embedding_model)

vector_db.save_local("vector_db2")

# vector_db = FAISS.load_local("vector_db_temp", embedding_model, allow_dangerous_deserialization=True)

# new_metadata = {
#     "id": "123456",  # this is the unique product id
#     "name": "Example Laptop",
#     "brand": "CoolBrand",
#     "price": 599.99,
#     "discount": 10,
#     "rating": 4.2,
#     "category": "Laptops",
#     "subcategory": "Student Laptops",
#     "stock": 20,
#     "specs": [{"RAM": "8GB"}, {"Storage": "256GB SSD"}],
#     "image_urls": ["http://example.com/image.jpg"]
# }

# spec_text = "; ".join(f"{k}: {v}" for d in new_metadata["specs"] for k, v in d.items())

# new_doc = Document(
#     page_content=(
#         f"The {new_metadata['name']} by {new_metadata['brand']} is a premium offering in the {new_metadata['category']} > "
#         f"{new_metadata['subcategory']} segment. Priced at ${new_metadata['price']}, it is currently available "
#         f"at a discount of {new_metadata['discount']}%. This product has received an average customer rating "
#         f"of {new_metadata['rating']} stars.\n\n"
#         f"Product Specifications:\n{spec_text}\n\n"
#         f"Availability Status: {'In stock' if new_metadata['stock'] > 0 else 'Temporarily unavailable'}."
#     ),
#     metadata=new_metadata
# )

# make sure 'embedding_model' and 'vector_db' are already defined
# vector_db.add_documents(documents=[new_doc], ids=[new_metadata["id"]])

# vector_db.save_local("vector_db_temp")

# Now get the actual Document from the in-memory docstore
# vector_db.delete(ids=["123456"])
# vector_db.save_local("vector_db_temp")
# doc = vector_db.docstore._dict['123456']

# Print the content and metadata
# print("--- Document at Index 2 ---")
# print("Content:\n", doc.page_content)
# print("\nMetadata:\n", json.dumps(doc.metadata, indent=2))
# for i, doc in enumerate(vector_db.docstore._dict.values()):
#     if i >= 1:
#         break
#     print(f"--- Document {i+1} ---")
#     print("Content:", doc.page_content[:500])  # First 500 characters
#     print("Metadata:", json.dumps(doc.metadata, indent=2))
#     print("\n")
# query = "289000"
# results = vector_db.similarity_search(query, k=5)
# for idx, doc in enumerate(results, 1):
#     print(f"Result {idx} Price: {doc.metadata['price']}")