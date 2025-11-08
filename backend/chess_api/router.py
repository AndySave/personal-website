
from fastapi import APIRouter


router = APIRouter()


@router.get("/api/chess/test")
async def test():
    return {"msg": "This is just a test"}
