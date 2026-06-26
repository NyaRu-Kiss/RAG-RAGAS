from pydantic import BaseModel, Field


class MessageRequest(BaseModel):
    message: str = Field(min_length=1)


class Citation(BaseModel):
    file_name: str
    file_path: str | None = None
    page_label: str | None = None
    score: float | None = None
    snippet: str


class RetrievedContext(BaseModel):
    file_name: str
    file_path: str | None = None
    page_label: str | None = None
    score: float | None = None
    text: str


class ChatResponse(BaseModel):
    answer: str
    citations: list[Citation] = []


class ActionResponse(BaseModel):
    message: str
    count: int = 0
