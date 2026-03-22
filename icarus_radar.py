"""
Project Icarus — 雷达引擎模块 (v4)
适配新 JSON 架构：archetype / core_cognitive_bias / fatal_action / trigger_condition
对外暴露 runAnalysis() 函数，供 FastAPI 调用。
内部逻辑：SentenceTransformer 向量检索 + JSON 查表补全 + Ollama 本地 LLM 批判式诊断。
"""

import json
import logging
from pathlib import Path

import chromadb
import requests
import torch
from sentence_transformers import SentenceTransformer

# NOTE: 日志配置
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("icarus_radar")

# ─── 常量 ───────────────────────────────────────────────
DATA_FILE = Path(__file__).parent / "grok_data.json"
DB_PATH = "./maverick_icarus_db"
COLLECTION_NAME = "fatal_decisions_archive"
EMBEDDING_MODEL = "BAAI/bge-large-zh-v1.5"
OLLAMA_ENDPOINT = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "qwen2.5:14b"

# ─── 延迟加载容器 ───────────────────────────────────────
_model: SentenceTransformer | None = None
_collection: chromadb.Collection | None = None
# NOTE: 全量 JSON 记录缓存，按 target_id 索引，用于 Prompt 拼接时查表获取长文本字段
_recordsIndex: dict[str, dict] | None = None


def _detectDevice() -> str:
    """
    安全检测 CUDA 可用性。
    RTX 5080 (Blackwell) 与某些 CUDA toolkit 版本存在兼容性问题，
    因此需要实际创建 tensor 进行端到端验证。
    """
    if not torch.cuda.is_available():
        return "cpu"
    try:
        _test = torch.zeros(1, device="cuda")
        del _test
        return "cuda"
    except Exception as e:
        logger.warning("CUDA 可用性检测失败 (%s)，回退至 CPU 模式", e)
        return "cpu"


def _ensureInit() -> None:
    """
    延迟初始化 SentenceTransformer、ChromaDB Collection 和 JSON 记录索引。
    首次调用 runAnalysis() 时触发，后续调用直接复用。
    """
    global _model, _collection, _recordsIndex
    if _model is not None and _collection is not None and _recordsIndex is not None:
        return

    device = _detectDevice()
    logger.info("初始化降维引擎: %s (device=%s)", EMBEDDING_MODEL, device)
    _model = SentenceTransformer(EMBEDDING_MODEL, device=device)
    if device == "cuda":
        logger.info("CUDA 加速已启用 🚀")
    else:
        logger.info("当前运行于 CPU 模式")

    logger.info("连接 ChromaDB → %s", DB_PATH)
    client = chromadb.PersistentClient(path=DB_PATH)
    _collection = client.get_collection(name=COLLECTION_NAME)
    logger.info("Collection [%s] 就绪，记录数: %d", COLLECTION_NAME, _collection.count())

    # NOTE: 加载全量 JSON 并按 target_id 建索引
    #       Prompt 拼接需要 core_cognitive_bias / fatal_action 等长文本字段，
    #       这些字段未存入 ChromaDB metadata（过长），而是从 JSON 直接查表获取
    logger.info("加载 JSON 记录索引: %s", DATA_FILE)
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        records = json.load(f)
    _recordsIndex = {r["target_id"]: r for r in records}
    logger.info("JSON 索引就绪，共 %d 条记录", len(_recordsIndex))


def _callOllama(prompt: str) -> str:
    """
    向本地 Ollama 服务发送同步推理请求。
    使用 qwen2.5:14b，stream 关闭以获取完整响应。
    """
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
    }
    try:
        logger.info("正在向 Ollama (%s) 发送推理请求...", OLLAMA_MODEL)
        resp = requests.post(OLLAMA_ENDPOINT, json=payload, timeout=120)
        resp.raise_for_status()
        result = resp.json()
        return result.get("response", "[ERROR] Ollama 返回体中未找到 response 字段")
    except requests.exceptions.ConnectionError:
        logger.error("无法连接 Ollama 服务，请确认 ollama serve 已启动")
        return "[ERROR] Ollama 服务连接失败 — 请执行 ollama serve 启动服务"
    except requests.exceptions.Timeout:
        logger.error("Ollama 推理超时 (>120s)")
        return "[ERROR] Ollama 推理超时，模型可能资源不足"
    except Exception as e:
        logger.error("Ollama 调用异常: %s", e)
        return f"[ERROR] Ollama 调用异常: {e}"


def _buildMaverickPrompt(
    danger_thought: str,
    target_id: str,
    archetype: str,
    core_cognitive_bias: str,
    fatal_action: str,
    ultimate_outcome: str,
) -> str:
    """
    拼接 Maverick 批判式 Prompt（v2：适配新架构字段）。
    将用户危险想法与 ChromaDB 检索到的历史死神全量档案注入 LLM 人设指令。
    """
    return (
        f"你现在是系统内置的冷酷心理侧写师（代号 Maverick）。\n"
        f"以下是用户当前的危险想法：\n"
        f"「{danger_thought}」\n\n"
        f"经过向量数据库语义检索，用户的思维模式与历史上的 {target_id} 高度吻合。\n"
        f"此人的原型分类为：{archetype}。\n\n"
        f"以下是 {target_id} 的完整档案：\n"
        f"- 核心认知偏差：{core_cognitive_bias}\n"
        f"- 致命行为：{fatal_action}\n"
        f"- 最终结局：{ultimate_outcome}\n\n"
        f"用户的当前想法极度危险，且与 {target_id} 走向了相同的深渊。\n"
        f"请结合此人的惨痛结局，用极具攻击性、冰冷且一针见血的语言，"
        f"对用户进行 200 字以内的无情批判，粉碎他的傲慢，并给出最后一条保命建议。"
    )


def runAnalysis(dangerThought: str) -> dict:
    """
    对外暴露的核心分析函数。

    @param dangerThought 用户输入的高危思维文本
    @returns 包含 target_id, archetype, outcome, distance, diagnosis 的字典
    """
    _ensureInit()

    # ── 1. 向量化 + ChromaDB 语义检索（取最匹配的 1 个）──
    logger.info("高危思维样本已锁定，正在向量化...")
    query_embedding = _model.encode([dangerThought], normalize_embeddings=True)

    results = _collection.query(
        query_embeddings=query_embedding.tolist(),
        n_results=1,
        include=["metadatas", "distances", "documents"],
    )

    target_id = results["ids"][0][0]
    metadata = results["metadatas"][0][0]
    distance = float(results["distances"][0][0])

    # ── 2. 从 JSON 索引查表获取完整记录（长文本字段）───
    full_record = _recordsIndex.get(target_id, {})
    core_cognitive_bias = full_record.get("core_cognitive_bias", "")
    fatal_action = full_record.get("fatal_action", "")
    archetype = metadata.get("archetype", full_record.get("archetype", ""))

    # ── 3. 调用 Ollama 生成批判式诊断书 ──────────────────
    prompt = _buildMaverickPrompt(
        danger_thought=dangerThought,
        target_id=target_id,
        archetype=archetype,
        core_cognitive_bias=core_cognitive_bias,
        fatal_action=fatal_action,
        ultimate_outcome=metadata["ultimate_outcome"],
    )
    diagnosis = _callOllama(prompt)

    # ── 4. 返回结构化结果 ────────────────────────────────
    return {
        "target_id": target_id,
        "archetype": archetype,
        "outcome": metadata["ultimate_outcome"],
        "distance": distance,
        "diagnosis": diagnosis,
    }
