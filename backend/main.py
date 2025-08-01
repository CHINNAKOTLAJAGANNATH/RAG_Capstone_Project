from fastapi import FastAPI
from backend.debug_index import combined

app = FastAPI()

@app.get("/debug/chunks")
async def get_chunks():
    return {"chunks": combined}
