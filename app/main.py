from pathlib import Path

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool

from app.config import Settings, get_settings
from app.rag import RagService, get_rag_service
from app.schemas import ActionResponse, ChatResponse, MessageRequest


BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"

app = FastAPI(title="LlamaIndex RAG")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/favicon.ico", include_in_schema=False)
async def favicon() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/documents/upload", response_model=ActionResponse)
async def upload_documents(
    files: list[UploadFile] = File(...),
    settings: Settings = Depends(get_settings),
) -> ActionResponse:
    saved = 0
    for file in files:
        if not file.filename:
            continue
        target = settings.upload_dir / Path(file.filename).name
        target.write_bytes(await file.read())
        saved += 1
    return ActionResponse(message=f"已上传 {saved} 个文件", count=saved)


@app.post("/api/index/rebuild", response_model=ActionResponse)
async def rebuild_index(
    settings: Settings = Depends(get_settings),
    rag: RagService = Depends(get_rag_service),
) -> ActionResponse:
    try:
        count = await run_in_threadpool(rag.rebuild_index)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"重建索引失败: {exc}") from exc
    message = f"索引完成，读取 {count} 个文档"
    if settings.pdf_parser == "docling":
        message += "（docling 解析耗时较长，属预期行为）"
    return ActionResponse(message=message, count=count)


@app.post("/api/index/update", response_model=ActionResponse)
async def update_index(
    rag: RagService = Depends(get_rag_service),
) -> ActionResponse:
    try:
        stats = await run_in_threadpool(rag.update_index)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"更新索引失败: {exc}") from exc
    message = (
        f"增量更新完成：新增 {stats.added}，更新 {stats.updated}，"
        f"删除 {stats.removed}，未变化 {stats.unchanged}"
    )
    return ActionResponse(message=message, count=stats.total_changed)


@app.post("/api/chat", response_model=ChatResponse)
async def chat(
    request: MessageRequest,
    rag: RagService = Depends(get_rag_service),
) -> ChatResponse:
    try:
        answer, citations = await run_in_threadpool(rag.chat, request.message)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"问答失败: {exc}") from exc
    return ChatResponse(answer=answer, citations=citations)


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")
