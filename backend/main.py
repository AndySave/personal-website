
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .nn_api.router import router as nn_router
from .chess_api.router import router as chess_router


app = FastAPI()
app.include_router(nn_router)
app.include_router(chess_router)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://10.0.0.18:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)
