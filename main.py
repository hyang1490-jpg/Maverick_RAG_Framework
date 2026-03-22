"""
Project Icarus — 神经中枢 (FastAPI 服务端) v2
适配新 JSON 架构：archetype 替代 peak_status。
"""

import logging
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from icarus_radar import runAnalysis

# NOTE: 日志配置
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("icarus_main")

# ─── 常量 ───────────────────────────────────────────────
STATIC_DIR = Path(__file__).parent / "maverick_icarus_db"

app = FastAPI(
    title="Project Icarus",
    description="本地化 RAG 心理侧写分析系统",
)

# NOTE: 挂载静态资源目录
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ─── 请求 / 响应 Schema ─────────────────────────────────
class AnalyzeRequest(BaseModel):
    """前端提交的危险思维分析请求"""
    danger_thought: str


class AnalyzeResponse(BaseModel):
    """分析结果响应体（v2：archetype 替代 peak_status）"""
    target_id: str
    archetype: str
    outcome: str
    distance: float
    diagnosis: str


# ─── 路由 ───────────────────────────────────────────────
@app.get("/")
async def serveFrontend():
    """返回前端 index.html 页面"""
    html_path = STATIC_DIR / "index.html"
    return FileResponse(str(html_path), media_type="text/html")


@app.post("/api/analyze", response_model=AnalyzeResponse)
async def analyze(req: AnalyzeRequest):
    """
    核心分析接口。
    接收前端的危险思维文本 → 调用雷达引擎 → 返回结构化分析结果。
    """
    logger.info("收到分析请求，输入长度: %d 字符", len(req.danger_thought))
    result = runAnalysis(req.danger_thought)
    logger.info("分析完成 → 匹配目标: %s, 距离: %.6f", result["target_id"], result["distance"])
    return AnalyzeResponse(**result)
