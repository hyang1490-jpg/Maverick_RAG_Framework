"""
Project Icarus - FastAPI 网关 (api_server.py)
=============================================
连接前端 UI 与底层 RAG 引擎 (icarus_core.py)。
流式返回审判结果，Content-Type: text/event-stream。

# NOTE: 参数红线
# -------------------------------------------------------
#   端口         = 8000
#   CORS         = 允许所有来源 (前端本地调试)
#   流式响应     = StreamingResponse + text/event-stream
#   后端引擎     = icarus_core.generate_judgment_stream
# -------------------------------------------------------
"""

import logging
import sys

import ollama
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from icarus_core import (
    SYSTEM_PROMPT,
    LLM_MODEL_NAME,
    LLM_NUM_CTX,
    generate_judgment_stream,
    getCollection,
    retrieveCases,
)

# ─── 日志配置 ───────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# ─── FastAPI 实例 ──────────────────────────────────────
app = FastAPI(
    title="Project Icarus - Tribunal API",
    description="RAG-powered business failure analysis engine",
    version="2.0.0",
)

# ─── CORS 跨域配置 (允许所有来源，方便前端本地调试) ────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── 请求体模型 ────────────────────────────────────────
class JudgeRequest(BaseModel):
    """
    审判请求体 (SSE 流式端点)。
    idea: 用户的商业想法（纯文本）
    """
    idea: str


class AnalyzeRequest(BaseModel):
    """
    分析请求体 (非流式端点，适配前端接口契约)。
    danger_thought: 用户的危险商业想法（纯文本）
    """
    danger_thought: str


# ─── 流式响应生成器 ────────────────────────────────────
def streamJudgmentResponse(user_idea: str):
    """
    包装 icarus_core 的生成器，将每个 chunk 编码为 UTF-8 bytes 流。
    StreamingResponse 要求 iterable 产出 bytes 或 str。
    """
    try:
        for chunk in generate_judgment_stream(user_idea):
            yield chunk
    except Exception as e:
        logger.error("[STREAM ERROR] %s", e)
        yield f"\n\n[ERROR] Pipeline execution failed: {e}"


def extractFailureReasons(document: str) -> str:
    """
    从 ChromaDB document 文本中提取失败原因。
    document 格式: "行业: XXX。失败原因: YYY。认知偏差原型: ZZZ。"
    """
    try:
        # 在 "失败原因: " 和 "。认知偏差原型:" 之间截取
        start_marker = "失败原因: "
        end_marker = "。认知偏差原型:"
        start_idx = document.index(start_marker) + len(start_marker)
        end_idx = document.index(end_marker)
        return document[start_idx:end_idx]
    except ValueError:
        # 解析失败则返回整个 document 作为兜底
        return document


# ─── API 路由 ──────────────────────────────────────────
@app.post("/api/judge")
async def judgeEndpoint(request: JudgeRequest):
    """
    POST /api/judge
    接收 {"idea": "用户的商业想法"}，
    流式返回伊卡洛斯审判结果 (SSE)。
    """
    if not request.idea or not request.idea.strip():
        raise HTTPException(status_code=400, detail="idea field cannot be empty")

    logger.info("[API] Received judgment request: %s", request.idea[:100])

    return StreamingResponse(
        streamJudgmentResponse(request.idea.strip()),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/api/analyze")
async def analyzeEndpoint(request: AnalyzeRequest):
    """
    POST /api/analyze
    接收 {"danger_thought": "..."}，非流式调用，返回结构化 JSON。
    适配前端已有接口契约。
    """
    user_input = request.danger_thought.strip()
    if not user_input:
        raise HTTPException(status_code=400, detail="danger_thought field cannot be empty")

    logger.info("[API] Received analyze request: %s", user_input[:100])

    try:
        # ─── 阶段一: 三路并行召回 ─────────────────────
        collection = getCollection()
        results = retrieveCases(collection, user_input)

        # top-1 用于现有 UI 字段（向后兼容）
        top_doc  = results["documents"][0][0]
        top_meta = results["metadatas"][0][0]
        top_dist = results["distances"][0][0]

        company_name   = top_meta.get("company_name", "Unknown")
        archetype      = top_meta.get("archetype", "Unknown")
        funding_amount = top_meta.get("funding_amount", "Unknown")

        failure_reasons = extractFailureReasons(top_doc)
        outcome = f"融资{funding_amount}后破产: {failure_reasons}"

        # 所有匹配案例，供前端多卡片渲染
        matched_cases = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            matched_cases.append({
                "company_name":   meta.get("company_name", "Unknown"),
                "archetype":      meta.get("archetype", "Unknown"),
                "industry":       meta.get("industry", "Unknown"),
                "distance":       round(dist, 4),
                "failure_reasons": extractFailureReasons(doc),
            })

        # ─── 阶段二: Prompt 组装（所有案例注入） ───────
        case_blocks = []
        for i, (doc, meta) in enumerate(
            zip(results["documents"][0], results["metadatas"][0]), start=1
        ):
            case_blocks.append(
                f"--- 案例 {i}: {meta.get('company_name')}(融资额: {meta.get('funding_amount')}) ---\n"
                f"核心认知偏差: {meta.get('archetype')}\n"
                f"案件详情: {doc}"
            )

        user_message = (
            f"【历史死亡档案】\n" + "\n\n".join(case_blocks) +
            f"\n\n【待审判的商业想法】\n{user_input}"
        )

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ]

        # ─── 阶段三: 非流式生成 ──────────────────────
        logger.info("[ANALYZE] calling %s (non-stream) ...", LLM_MODEL_NAME)
        response = ollama.chat(
            model=LLM_MODEL_NAME,
            messages=messages,
            options={"num_ctx": LLM_NUM_CTX},
            stream=False,
        )
        diagnosis = response["message"]["content"]

        return {
            "target_id":     company_name,
            "archetype":     archetype,
            "outcome":       outcome,
            "distance":      round(top_dist, 4),
            "diagnosis":     diagnosis,
            "matched_cases": matched_cases,
        }

    except Exception as e:
        logger.error("[ANALYZE ERROR] %s", e)
        raise HTTPException(status_code=500, detail=f"Analysis failed: {e}")


# ─── 健康检查 ──────────────────────────────────────────
@app.get("/api/health")
async def healthCheck():
    """
    GET /api/health
    服务健康检查。
    """
    return {"status": "alive", "service": "Project Icarus Tribunal API"}


# ─── 启动入口 ──────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info",
    )
