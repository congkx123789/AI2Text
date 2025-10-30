from __future__ import annotations
from builtins import str
from pydantic import BaseModel
from typing import List, Any


class TranscribeRequest(BaseModel):
    audio_path: str


class AskRequest(BaseModel):
    query: str


class RAGResponse(BaseModel):
    answer: str
    contexts: List[Any]