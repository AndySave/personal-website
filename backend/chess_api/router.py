
from fastapi import APIRouter, HTTPException
import chess.engine

from .constants import STARTPOS
from .schemas import Moves


router = APIRouter()

@router.post("/api/chess/get-engine-move")
async def get_engine_move(moves: Moves):
    with chess.engine.SimpleEngine.popen_uci('stockfish.exe') as engine:
        board = chess.Board(STARTPOS)
        for move in moves.moves:
            if move not in board.generate_legal_moves():
                raise HTTPException(status_code=422, detail="Illegal move")
            board.push_san(move)

        limit = chess.engine.Limit(time=0.1)
        analysis_result = engine.analyse(board, limit)

        best_move = analysis_result.get('pv')[0]

    return {"best_move": str(best_move)}
