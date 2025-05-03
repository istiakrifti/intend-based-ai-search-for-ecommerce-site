from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import sqlToVectordb
from pipeline import run_search_pipeline
import asyncio
import asyncpg
import os
from dotenv import load_dotenv

load_dotenv()

dbname = os.environ.get("DB_NAME")
user = os.environ.get("DB_USER")
password = os.environ.get("DB_PASSWORD")
host = os.environ.get("DB_HOST")
port = os.environ.get("DB_PORT")

app = FastAPI()

# Serve static files (images, CSS, etc.)
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

class QueryRequest(BaseModel):
    query: str

# ---------- Background Task: Process Change Logs ----------
async def process_change_logs():
    while True:
        conn = await asyncpg.connect(
            user=user,
            password=password,
            database=dbname,
            host=host,
            port=port
        )

        rows = await conn.fetch("""
            SELECT id, product_id, action 
            FROM product_change_log 
            WHERE processed = FALSE 
            ORDER BY created_at 
            LIMIT 100;
        """)

        if rows:
            batch = []
            log_ids = []

            for row in rows:
                batch.append({
                    "product_id": row["product_id"],
                    "action": row["action"]
                })
                log_ids.append(row["id"])

            try:
                # Batch call to your upsert function
                await sqlToVectordb.upsert_product(batch)
                print(f"Processed {len(batch)} product(s)")

                # Batch mark processed
                await conn.executemany("""
                    UPDATE product_change_log 
                    SET processed = TRUE 
                    WHERE id = $1;
                """, [(log_id,) for log_id in log_ids])

            except Exception as e:
                print(f"Failed to process batch: {e}")

        await conn.close()
        await asyncio.sleep(120)

# ---------- FastAPI Events ----------
@app.on_event("startup")
async def startup_event():
    asyncio.create_task(process_change_logs())

@app.get("/", response_class=HTMLResponse)
async def serve_page(request: Request):
    return templates.TemplateResponse("search.html", {"request": request})


@app.get("/search")
async def search_products(query):
    results = run_search_pipeline(query)
    return JSONResponse(content={"products": results})
