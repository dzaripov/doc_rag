from pydantic import BaseModel, Field
from enum import Enum
from datetime import datetime

class QueryInput(BaseModel):
    question: str
    session_id: str = Field(default=None)

class QueryResponse(BaseModel):
    answer: str
    session_id: str

class DocumentInfo(BaseModel):
    id: int
    filename: str

class DeleteFileRequest(BaseModel):
    file_id: int
