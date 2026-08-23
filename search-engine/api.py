import asyncio
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse

from indexer import build_search_index, search_publications

REINDEX_INTERVAL_SECONDS = 60 * 60

INDEX = build_search_index()
print(f"Indexed {len(INDEX['docs'])} publications.")

async def reindex_periodically():
    while True:
        await asyncio.sleep(REINDEX_INTERVAL_SECONDS)
        global INDEX
        try:
            new_index = await asyncio.to_thread(build_search_index)
            INDEX = new_index
            print(f"Reindexed {len(INDEX['docs'])} publications.")
        except Exception as e:
            print(f"Reindex failed, keeping previous index: {e}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    task = asyncio.create_task(reindex_periodically())
    yield
    task.cancel()

app = FastAPI(title="Coventry CHCT Publication Search", lifespan=lifespan)

HOME_PAGE = (Path(__file__).parent / "templates" / "index.html").read_text()

@app.get("/search")
def search(q: str = Query("", description="Search query")):
    if not q.strip():
        return {"query": q, "results": []}
    return {"query": q, "results": search_publications(INDEX, q)}

@app.get("/", response_class=HTMLResponse)
def home():
    return HOME_PAGE
