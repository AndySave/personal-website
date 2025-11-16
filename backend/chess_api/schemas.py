
from pydantic import BaseModel

class Moves(BaseModel):
    moves: list[str]
