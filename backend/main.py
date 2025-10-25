
from fastapi import FastAPI

app = FastAPI()

@app.get("/api/nn-framework/status")
async def chess_engine_status():
    return {"status": "OK"}
